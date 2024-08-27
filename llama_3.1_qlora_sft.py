!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
!pip install evaluate
!pip install rouge_score

import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments, EarlyStoppingCallback
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported
from evaluate import load as load_metric

# Load ROUGE metric
rouge_metric = load_metric('rouge')

# Define the compute function for ROUGE
def compute_metrics(eval_pred):
    predictions, references = eval_pred
    decoded_preds = [pred.strip() for pred in predictions]
    decoded_labels = [ref.strip() for ref in references]
    
    # Compute ROUGE score
    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    # Optional: format the result dictionary as needed
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    return result

# Set max sequence length
max_seq_length = 1024

# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

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

# Configure the tokenizer with chat templates
tokenizer = get_chat_template(
    tokenizer,
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    chat_template="chatml",
)

# Define a function to apply templates to conversations
def apply_template(examples):
    messages = examples["conversations"]
    text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
    return {"text": text}

# Load the dataset and split into train and validation
dataset = load_dataset('arrow', data_files='training_data.arrow',split="train")
dataset = dataset.map(apply_template, batched=True)

# Split dataset into train and validation
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']
print(train_test_split)


# Define training arguments
training_args = TrainingArguments(
    learning_rate=3e-4,
    lr_scheduler_type="linear",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=5,  # Increase epochs to allow early stopping to work
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    warmup_steps=10,
    output_dir="output",
    seed=42,
    evaluation_strategy="steps",  # Enable evaluation during training
    eval_steps=50,                # Evaluate every 50 steps
    load_best_model_at_end=True,  # Load best model at the end of training
    metric_for_best_model="rougeL",  # Metric to monitor (e.g., rougeL)
    greater_is_better=True,       # Higher ROUGE is better
    save_steps=50,                # Save model every 50 steps
    save_total_limit=3,           # Only keep the last 3 models
)

# Set up the trainer with early stopping callback
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    args=training_args,
    compute_metrics=compute_metrics,  # Pass the custom compute function
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Add early stopping
)

# Train the model with early stopping based on ROUGE
trainer.train()

model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
