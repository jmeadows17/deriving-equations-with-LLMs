!pip install sentencepiece
!pip install transformers
!pip install torch
!pip install rich[jupyter]
!pip install datasets
!pip install git+https://github.com/google-research/bleurt.git

# Importing libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os
from tqdm.auto import tqdm
import json
from datasets import Dataset, concatenate_datasets

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console

# define a rich console logger
console = Console(record=True)

# to display dataframe in ASCII format
def display_df(df):
    """display dataframe in ASCII format"""

    console = Console()
    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    console.print(table)

# training logger to log training progress
training_logger = Table(
    Column("Epoch", justify="center"),
    Column("Steps", justify="center"),
    Column("Loss", justify="center"),
    title="Training Status",
    pad_edge=False,
    box=box.ASCII,
)

# Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


class YourDataSetClass(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        #source_text = " ".join(source_text.split())
        #target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }



def train(epoch, tokenizer, model, device, loader, optimizer):

    """
    Function to be called for training with the parameters passed from main function

    """

    model.train()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]

        if _ % 200 == 0:
            training_logger.add_row(str(epoch), str(_), str(loss))
            console.print(training_logger)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


import torchtext
from torchtext.data import get_tokenizer
#tokenizer_english = get_tokenizer("basic_english")
#from torchtext.data.metrics import bleu_score


def validate(epoch, tokenizer, model, device, loader):

    """
    Function to evaluate model for predictions

    """
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_length=512, 
              num_beams=2,
              repetition_penalty=2.5, 
              length_penalty=1.0, 
              early_stopping=True
              )

            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]

            if _%10==0:
                console.print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)

    return predictions, actuals


from torchtext.data.metrics import bleu_score
from transformers.optimization import Adafactor, AdafactorSchedule
import numpy as np
from datasets import load_metric

def T5Trainer(
    train_set, dev_set, source_text, target_text, model_params, output_dir=OUTPUT_DIR
):

    """
    T5 trainer

    """

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

    new_tokens = ["\\","{","}","_","^"]

    # add the tokens to the tokenizer vocabulary
    tokenizer.add_tokens(list(new_tokens))

    # add new, random embeddings for the new tokens
    model.resize_token_embeddings(len(tokenizer))

    # logging
    console.log(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    train_set = train_set[[source_text, target_text]]
    display_df(train_set.head(2))

    # Importing the raw dataset
    dev_set = dev_set[[source_text, target_text]]
    display_df(dev_set.head(2))

    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation.
    #train_size = 0.8
    #train_dataset = dataframe.sample(frac=train_size, random_state=model_params["SEED"])
    #val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
    #train_dataset = train_dataset.reset_index(drop=True)

    train_dataset = train_set
    val_dataset = dev_set

    #console.print(f"FULL Dataset: {dataframe.shape}")
    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    console.print(f"TEST Dataset: {val_dataset.shape}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = YourDataSetClass(
        train_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    val_set = YourDataSetClass(
        val_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    
    #optimizer = torch.optim.Adam(
    #    params=model.parameters(), lr=model_params["LEARNING_RATE"]
    #)

    optimizer = Adafactor(
    model.parameters(),
    lr=model_params["LEARNING_RATE"],
    eps=(1e-30, model_params["LEARNING_RATE"]),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=0.0,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False)

    # Training loop
    console.log(f"[Initiating Fine Tuning]...\n")

    # Load evaluation metric
    best_val_score = -1.0    
    metric = load_metric('bleurt', "bleurt-large-512")
    
    loop = tqdm(range(model_params["TRAIN_EPOCHS"]))
    for epoch in loop:
        train(epoch, tokenizer, model, device, training_loader, optimizer)
        # evaluating test dataset at the end of each epoch
        console.log(f"[Initiating Validation]...\n")
        for epoch in range(model_params["VAL_EPOCHS"]):
            predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
            print(predictions, actuals)
            metric.add_batch(predictions=predictions, references=actuals)
            eval_score = np.mean(metric.compute()["scores"])
            final_df = pd.DataFrame({"Questions":val_dataset[source_text], "Generated Text": predictions, "Actual Text": actuals})
            if eval_score > best_val_score:
                best_val_score = eval_score              
                final_df.to_csv(os.path.join(output_dir, "predictions.csv"))
                console.log(f"[Saving Model]...\n")
                # Saving the model after training
                path = os.path.join(output_dir, "model_files")
                model.save_pretrained(path)
                tokenizer.save_pretrained(path)
                console.save_text(os.path.join(output_dir, "logs.txt"))
                console.print(
                    f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
                )
                console.print(
                    f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n"""
                )
                console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")
            console.log(f"[Validation Completed. Best Bleurt score: {best_val_score}. Current Bleurt score: {eval_score}]\n")






model_params = {
    "MODEL": "google/flan-t5-base"#"/content/drive/MyDrive/outputs/model_files",
    "TRAIN_BATCH_SIZE": 4,  # training batch size
    "VALID_BATCH_SIZE": 4,  # validation batch size
    "TRAIN_EPOCHS": 10,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 5e-5,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 512,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
}


df_train = pd.DataFrame(list(dataset["train"]))
print(df_train.head())

df_dev = pd.DataFrame(list(dataset["test"]))
print(df_dev.head())


OUTPUT_DIR = "/content/drive/MyDrive/outputs/"
DATA_DIR = "data.json"


with open(DATA_DIR,"r") as f:
    d_json = json.load(f)

dataset = Dataset.from_list(d_json).train_test_split(test_size=0.1)

T5Trainer(
    train_set=df_train,
    dev_set=df_dev,
    source_text="prompt",
    target_text="derivation",
    model_params=model_params,
    output_dir=OUTPUT_DIR,
)
