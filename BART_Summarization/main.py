import pandas as pd 
from datasets import load_dataset
from datasets import Dataset as data_set
from datasets import DatasetDict 

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# dataset = load_dataset("cnn_dailymail",'1.0.0')

df = pd.read_csv('summarization.csv')

articles_list = df['articles'] # Your list of articles
summaries_list = df['highlights']  # Your list of summaries

class CustomDataset(Dataset):
    def __init__(self, articles, summaries):
        self.articles = articles
        self.summaries = summaries
        self.length = len(articles)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        article = self.articles[idx]
        summary = self.summaries[idx]
        return {'article': article, 'summary': summary}


# Split the data into train, test, and validation sets
total_length = len(articles_list)
train_length = int(0.8 * total_length)
test_length = int(0.1 * total_length)
val_length = total_length - train_length - test_length

train_data, test_data, val_data = random_split(
    list(zip(articles_list, summaries_list)),
    [train_length, test_length, val_length]
)

# Unzip the datasets
train_articles, train_summaries = zip(*train_data)
test_articles, test_summaries = zip(*test_data)
val_articles, val_summaries = zip(*val_data)

# Create instances of the CustomDataset for train, test, and validation
train_dataset = CustomDataset(train_articles, train_summaries)
test_dataset = CustomDataset(test_articles, test_summaries)
val_dataset = CustomDataset(val_articles, val_summaries)

# Create DataLoader instances for train, test, and validation
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Create DatasetDict
dataset_dict = {
    'train': train_dataset,
    'test': test_dataset,
    'val': val_dataset
}

# Create DataLoaderDict
dataloaders_dict = {
    'train': train_loader,
    'test': test_loader,
    'val': val_loader
}

def convert_to_hf_dataset(dataset):
    hf_dataset = data_set.from_dict({
        'input': dataset.articles, 
        'output': dataset.summaries
    })
    return hf_dataset

train_hf_dataset = convert_to_hf_dataset(train_dataset)
test_hf_dataset = convert_to_hf_dataset(test_dataset)
val_hf_dataset = convert_to_hf_dataset(val_dataset)

samsum = DatasetDict({
    'train': train_hf_dataset,
    'test': test_hf_dataset,
    'val': val_hf_dataset
})

print(samsum) 

# print(samsum['train'][0])

device = 'gpu'
model_ckpt = 't5-small'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)

def get_feature(batch):
    encodings = tokenizer(
        batch['input'], 
        text_target=batch['output'],
        max_length=1024, 
        truncation=True
    )
    encodings = {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': encodings['labels']
    }
    return encodings

samsum_pt = samsum.map(get_feature, batched=True)
columns = ['input_ids', 'labels', 'attention_mask']
samsum_pt.set_format(type='torch', columns=columns)

from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir = 'bart_samsum',
    num_train_epochs=2,
    warmup_steps = 500,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay = 0.01,
    logging_steps = 10,
    evaluation_strategy = 'steps',
    eval_steps=500,
    save_steps=1e6,
    gradient_accumulation_steps=16
)

trainer = Trainer(
    model=model, 
    args=training_args, 
    tokenizer=tokenizer, 
    data_collator=data_collator,
    train_dataset = samsum_pt['train'], 
    eval_dataset = samsum_pt['val']
)

trainer.train()

trainer.save_model('bart_samsum_model')

