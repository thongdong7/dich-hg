import os
from transformers import (
    Trainer,
    TrainingArguments,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
)
from datasets import load_dataset

CHECKPOINT = "facebook/m2m100_418M"

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

tokenizer = M2M100Tokenizer.from_pretrained(CHECKPOINT, src_lang="zh", tgt_lang="vi")

# Dataset dir
dataset_dir = os.environ["DATASET_ROOT"]

# Load file train.csv and validation.csv from dataset_dir
train_file = os.path.join(dataset_dir, "train.csv")
val_file = os.path.join(dataset_dir, "validation.csv")

# Load dataset
ds = load_dataset("csv", data_files={"train": train_file, "validation": val_file})


# Tokenize and encode the dataset
def preprocess_function(examples):
    inputs = [ex for ex in examples["src"]]
    targets = [ex for ex in examples["target"]]
    # model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    # Inputs could be string or string[]
    # text_target could be string or string[]
    model_inputs = tokenizer(
        inputs,
        text_target=targets,
        padding="max_length",
        max_length=235,
        truncation=True,
        return_tensors="pt",
    )

    # The model_inputs object contains the input_ids, attention_mask, and labels now

    return model_inputs


tokenized_dataset = ds.map(preprocess_function, batched=True)

# Print the first example of the training set
print("-" * 50)
print("Example of the first training set:")
print(tokenized_dataset["train"][0])
print("-" * 50)

model = M2M100ForConditionalGeneration.from_pretrained(CHECKPOINT)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="none",
    disable_tqdm=True,
    save_strategy="epoch",
    save_total_limit=1,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# Fine-tune the model
trainer.train()
