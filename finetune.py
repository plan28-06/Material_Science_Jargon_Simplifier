import json
from datasets import Dataset
from transformers import (
    BartTokenizerFast,
    BartForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import evaluate
import numpy as np

MODEL_NAME = "facebook/bart-base"
MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 256

# --- 1. Load data ---
def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

train_data = load_jsonl("train.jsonl")
val_data = load_jsonl("val.jsonl")

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

# --- 2. Load tokenizer and model ---
tokenizer = BartTokenizerFast.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

# --- 3. Preprocess: tokenize inputs and targets ---
def preprocess(examples):
    inputs = examples["complex"]
    targets = examples["simple"]

    model_inputs = tokenizer(
        inputs, max_length=MAX_INPUT_LEN, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        text_target=targets, max_length=MAX_TARGET_LEN, truncation=True, padding="max_length"
    )

    # Replace padding token id with -100 so it's ignored in loss calculation
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_tokenized = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
val_tokenized = val_dataset.map(preprocess, batched=True, remove_columns=val_dataset.column_names)

# --- 4. Data collator (handles batching/padding automatically) ---
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# --- 5. Metrics (ROUGE) ---
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return {k: round(v, 4) for k, v in result.items()}

# --- 6. Training arguments ---
training_args = Seq2SeqTrainingArguments(
    output_dir="./bart-simplifier-checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,      # small batch size to fit 8GB VRAM
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,       # effective batch size = 4*4 = 16
    weight_decay=0.01,
    save_total_limit=2,                  # only keep last 2 checkpoints (saves disk space)
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=True,                           # mixed precision, faster + less VRAM on RTX 4060
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    report_to="none",                    # disable wandb/etc logging
)

# --- 7. Trainer ---
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# --- 8. Train ---
trainer.train()

# --- 9. Save final model ---
trainer.save_model("./bart-simplifier-final")
tokenizer.save_pretrained("./bart-simplifier-final")

print("Training complete. Model saved to ./bart-simplifier-final")