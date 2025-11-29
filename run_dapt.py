"""
Domain-adaptive pretraining of bert-large-cased on public CTI tweets.
"""
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

# 1) Public CTI tweet corpus
dataset = load_dataset("CyberDan/security_tweets", split="train")  # each row has "text"

# 2) Tokenizer & MLM model
checkpoint = "bert-large-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForMaskedLM.from_pretrained(checkpoint)


def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256)


tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)

# 3) TrainingArguments
args = TrainingArguments(
    output_dir="bert-large-cti-dapt",
    num_train_epochs=1,
    per_device_train_batch_size=16,  # max_len=256 => fits comfortably on modern GPUs
    fp16=True,
    learning_rate=5e-5,
    logging_steps=200,
    save_total_limit=1,
    report_to="none",
)

# 4) Trainer
Trainer(model=model, args=args, train_dataset=tokenized, data_collator=data_collator).train()
model.save_pretrained("bert-large-cti-dapt")
tokenizer.save_pretrained("bert-large-cti-dapt")
print("DAPT complete: bert-large-cti-dapt/ is ready.")
