#%%
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import pipeline


#loading sample data
df = pd.read_csv("./data/tone_responses.csv")


#%%
#cleaning 
df.replace('"', '', inplace=True)
df.iloc[63]


#%%

df["tone"] = df["tone"].map({
    "neutral": 0,
    "curious": 1,
    "happy": 2,
    "excited": 3,
    "bored": 4,
    "confused": 5})

#renaming tone as label because huggingface expects that
df["label"] = df["tone"]


#this is converting the dataframe into a huggingface dataset
#this will make it easier to split into train and test sets

dataset = Dataset.from_pandas(df[["comment", "label"]])
dataset = dataset.train_test_split(test_size=0.2)


#splitting into train and test sets. we are using 
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=6)

#tokenizing the comments
def tokenize(batch):
    return tokenizer(
        batch["comment"],
        padding="max_length",
        truncation=True,
        max_length=128)

tokenized = dataset.map(tokenize, batched=True)



#training the model
training_args = TrainingArguments(
    output_dir="tone_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"])

trainer.train()


#saving the model
trainer.save_model("tone_model")
tokenizer.save_pretrained("tone_model")



#loading the model
classifier = pipeline(
    "text-classification",
    model="tone_model",
    tokenizer="tone_model",
    return_all_scores=True)


classifier("I'm confused about fractions")




# %%
