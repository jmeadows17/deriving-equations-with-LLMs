import json
import torch
import pandas as pd
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from evaluate import load
from tqdm.auto import tqdm
import os

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, sp.Integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def evaluate(df_1, metric_name, df_2=None, static=True):
    # df_1 = unperturbed
    m_list = []
    if  metric_name == "gleu":
        metric = load("google_bleu")
    elif metric_name == "bleurt":
        metric = load("bleurt", "bleurt-large-512")
    else:
        metric = load(metric_name)
    for i in tqdm(range(len(df_1))):
        if static:
            ref_1 = df_1.iloc[i]["Actual Text"]
            pred_1 = df_1.iloc[i]["Generated Text"]
            m_1 = metric.compute(predictions=[pred_1], references=[ref_1])
            if metric_name == "rouge":
                m_1 = m_1['rouge2']
            if metric_name == "bleu":
                m_1 = m_1['bleu']
            if metric_name == "gleu":
                m_1 = m_1['google_bleu']
            if metric_name == "bleurt":
                m_1 = m_1['scores'][0]
            m_list.append(m_1)
        else:
            ref_1, ref_2 = df_1.iloc[i]["Actual Text"], df_2.iloc[i]["p Actual Text"]
            pred_1, pred_2 = df_1.iloc[i]["Generated Text"], df_2.iloc[i]["p Generated Text"]
            m_1, m_2 = metric.compute(predictions=[pred_1], references=[ref_1]), metric.compute(predictions=[pred_2], references=[ref_2])
            if metric_name == "rouge":
                m_1, m_2 = m_1["rouge2"], m_2["rouge2"]
            if metric_name == "bleu":
                m_1, m_2 = m_1["bleu"], m_2["bleu"]
            if metric_name == "gleu":
                m_1, m_2 = m_1["google_bleu"], m_2["google_bleu"]
            if metric_name == "bleurt":
                m_1, m_2 = m_1["scores"][0], m_2["scores"][0]
            m_list.append(m_1 - m_2)
    return m_list

def df2json(df):
    data = []
    for i in range(len(df)):
        entry = {k: df.iloc[i][k] for k in df.columns}
        data.append(entry)
    return data
    
model_id = os.getcwd() + "\\models\\t5-base_25_epochs_5e-5"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = T5Tokenizer.from_pretrained(model_id)
model = T5ForConditionalGeneration.from_pretrained(model_id).to(device)

evaluation = "alt_goal"

with open("static_prompts_%s.json"%(evaluation), "r") as f:
    data = json.load(f)

with open("%s_prompts.json"%(evaluation), "r") as f:
    p_data = json.load(f)
    
    
# for testing
data = data[:5]
p_data = p_data[:5]

print("\n Generating perturbed derivations...")
static_outputs = []
perturbed_outputs = []
loop = tqdm(range(len(data)))
for i in loop:
    input_ids = tokenizer.encode(data[i]["prompt"], return_tensors='pt', max_length=512, truncation=True).to(device)
    output = model.generate(input_ids=input_ids, max_length=512, early_stopping=True)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    if data[i]['derivation'][0] == "\\":
        generated_text = "\\ " + generated_text
        
    static_outputs.append({
        "Actual Text":data[i]["derivation"],
        "Generated Text":generated_text,
        "srepr_derivation":data[i]['srepr_derivation']
    })
    
    input_ids = tokenizer.encode(p_data[i]["prompt"], return_tensors='pt', max_length=512, truncation=True).to(device)
    output = model.generate(input_ids=input_ids, max_length=512, early_stopping=True)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    if p_data[i]['derivation'][0] == "\\":
        generated_text = "\\ " + generated_text
        
    perturbed_outputs.append({
        "p Actual Text":p_data[i]["derivation"],
        "p Generated Text":generated_text,
        "p_srepr_derivation":p_data[i]['srepr_derivation']
    })

print("\n")
    
df_1, df_2 = pd.DataFrame(static_outputs), pd.DataFrame(perturbed_outputs)
df = pd.concat([df_1, df_2], axis=1)

for metric_name in ["rouge","bleu","bleurt","gleu"]:
    print(metric_name)
    df[metric_name] = evaluate(df_1, metric_name, df_2=df_2, static=False)
    print("\n")

data = df2json(df)

file_name = evaluation + "_" + model_id.split("models\\")[1] + ".json"

with open(file_name, "w") as f:
    json.dump(data, f, cls=NpEncoder)
