!pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
!pip install evaluate
!pip install rouge_score

import os
import gc
import torch
from peft import PeftModel
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported
from evaluate import load as load_metric
import numpy as np
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

# Define a function to apply templates to conversations
def apply_template(examples):
    messages = examples["conversations"]
    text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
    return {"text": text}

# Define a function for prompting the model
def prompt_model(prompt, model, tokenizer):
    inputs = tokenizer.apply_chat_template(
        [prompt],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    generation = model.generate(input_ids=inputs, max_new_tokens=400, use_cache=True)
    decoded_seq = tokenizer.decode(generation[0],
                                   skip_special_tokens=True,
                                   do_sample=False)
    return decoded_seq

def validation_epoch(model, tokenizer, eval_dataset, metric):
    with torch.no_grad():
        FastLanguageModel.for_inference(model)
        tokenizer.padding_side = "left"
        print("\n\n Validation epoch...")
        scores = []
        for i in tqdm(range(len(eval_dataset))):
            prompt = eval_dataset[i]["conversations"][0]
            output = prompt_model(prompt, model, tokenizer)
            prediction = output.split("assistant")[1]
            reference = tokenizer.apply_chat_template(
                    [eval_dataset[i]["conversations"][1]],
                    tokenize=False
                ).split("assistant")[1].replace("<|im_end|>","")
            score = metric.compute(predictions=[prediction], references=[reference])["rouge2"]
            scores.append(score)
        model.train()
        current_score = np.mean(scores)
    return current_score

if __name__ == "__main__":

    model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"

    # Set parameters
    max_seq_length = 1024
    batch_size = 1
    learning_rate = 5e-5

    # load validation metric
    metric = load_metric("rouge")

    # Load model and tokenizer (iteratively reloading to avoid OOM)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    # Configure the tokenizer with chat templates
    tokenizer = get_chat_template(
        tokenizer,
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
        chat_template="chatml",
    )

    # Load the dataset and split into train and validation
    dataset = load_dataset('arrow', data_files='training_data.arrow', split="train")
    dataset = dataset.map(apply_template, batched=True)
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    print(train_test_split)

    # Configure the model with PEFT and LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
        use_rslora=True,
        use_gradient_checkpointing="unsloth"
    )

    # Create training arguments
    training_args = TrainingArguments(
        learning_rate=learning_rate,
        lr_scheduler_type="linear",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=16,
        num_train_epochs=1,  # We train for one epoch at a time
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        optim="adamw_8bit",
        weight_decay=0.01,
        output_dir="output",
        seed=42,
    )

    # Early stopping parameters
    best_score = float('-inf')
    current_score = 0
    patience = 3
    max_epochs = 10
    eps = 1e-4
    epoch = 0

    # Early stopping logic
    while True:

        tokenizer.padding_side = "right"

        # Create the optimizer
        optimizer = AdamW(model.parameters(), lr=learning_rate)

        # Calculate the total number of training steps
        num_training_steps = (len(train_dataset) * max_epochs) // (batch_size * training_args.gradient_accumulation_steps)

        # Create the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

        if epoch > 0:
            # Load optimizer state
            optimizer_state_path = os.path.join(training_args.output_dir, "optimizer.pt")
            optimizer_state = torch.load(optimizer_state_path)
            optimizer.load_state_dict(optimizer_state)

            # Load scheduler state
            scheduler_state_path = os.path.join(training_args.output_dir, "scheduler.pt")
            scheduler_state = torch.load(scheduler_state_path)
            scheduler.load_state_dict(scheduler_state)

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=2,
            args=training_args,
            optimizers=(optimizer, scheduler)  # Pass the optimizer and scheduler
        )
        
        # Train for one epoch
        trainer.train()

        # Save lora model and states
        model.save_pretrained(f"LoRA_epochs={epoch+1}")
        # Save optimizer and scheduler states
        torch.save(trainer.optimizer.state_dict(), os.path.join(training_args.output_dir, "optimizer.pt"))
        torch.save(trainer.lr_scheduler.state_dict(), os.path.join(training_args.output_dir, "scheduler.pt"))


       
        # Validation epoch to calculate ROUGE
        current_score = validation_epoch(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            metric=metric
        )
        print(f"\n\n Score after {epoch+1} epochs = {current_score*100}")

        if current_score > best_score + eps:
            best_score = current_score
            print(f"\n\n New best score = {best_score}")
            print(f"\n\n BEST EPOCH = {epoch+1}")
            patience = 0

        else:
            patience += 1
            print(f"\n\n Patience = {patience}")
            if patience == 3:
                break

        epoch += 1
        if epoch >= max_epochs:
            break
        
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n\n Training complete after {epoch} epochs. Final score = {best_score}")
