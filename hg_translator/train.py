from genericpath import exists
import json
import os
from posixpath import abspath, dirname, join
import tempfile
from datasets import load_dataset, DatasetDict, Dataset
import shutil
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    pipeline,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

import evaluate
import numpy as np

# Save the start time to measure execution time of the script
import time

from hg_translator.dataset import get_full_dataset, get_sample_dataset
from .model_config import (
    model_name,
    num_train_epochs,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    checkpoint,
    prefix,
    source_lang,
    target_lang,
)

start = time.time()

# Config
# number_of_books = 100

print("Number of train epochs:", num_train_epochs)

repo_dir = dirname(dirname(abspath(__file__)))
print("Repo directory:", repo_dir)
os.chdir(repo_dir)

label_dir = join(repo_dir, "tien_hiep/label_new")
if not exists(label_dir):
    label_dir = join(repo_dir, "label")

if not exists(label_dir):
    label_dir = "/kaggle/input/dich-hg1/label"

print("Label directory:", label_dir)


def gen():
    # Get all .json files in the label directory
    for root, dirs, files in os.walk(label_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                file_id = os.path.relpath(file_path, label_dir).split(".")[0]
                print("Processing file:", file_id, flush=True)
                with open(file_path, "r") as f:
                    data = json.load(f)

                for i, line in enumerate(data["lines"]):
                    if (
                        line.get("approved")
                        and line.get("convert")
                        and line.get("dich")
                    ):
                        yield {
                            "id": f"{file_id}_{i}",
                            "translation": {
                                "convert": line["convert"],
                                "dich": line["dich"],
                            },
                        }


# ds = get_sample_dataset()
ds = get_full_dataset()
train_test_data = ds.train_test_split(
    test_size=0.2
)  # 80% for training, 20% for testing
train_val_data = train_test_data["train"].train_test_split(
    test_size=0.25
)  # 75% of 80% for training, 25% of 80% for validation


train_data = train_val_data["train"]
val_data = train_val_data["test"]
test_data = train_test_data["test"]

print("Sample train data:", train_data[0])

print("Number of train data:", len(train_data))
# {'id': '3_34', 'translation': {'convert': 'Buổi sáng về sau, Vương hộ pháp cũng không có nhường mọi người ăn điểm tâm, trực tiếp bả nhiều người người tới dưới núi một mảng lớn đủ loại cây trúc sườn dốc trước mặt', 'dich': 'Hôm sau trời vừa sáng, Vương hộ pháp cũng không để cho mọi người được ăn điểm tâm, trực tiếp mang cả bọn xuống núi, tới trước một khu rừng trúc rậm rạp'}}

tokenizer_model = checkpoint


def load_tokenizer():
    print(f"Loading Tokenizer {tokenizer_model}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    return tokenizer


tokenizer = load_tokenizer()


def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, truncation=False, padding=True
    )
    # print("model inputs:", model_inputs)
    # model inputs: {'input_ids': [[13959, 1566, 12, 2379, 10, 27, 1], [13959, 1566, 12, 2379, 10, 1853, 10842, 10327, 3316, 1]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 'labels': [[3, 20891, 4111, 20371, 22694, 329, 18988, 1], [9132, 276, 18433, 9215, 5999, 14132, 1]]}
    return model_inputs


# Tokenizing the data
print("Tokenizing train data...")
tokenized_train_data = train_data.map(preprocess_function, batched=True)

print("Tokenizing validation data...")
tokenized_val_data = val_data.map(preprocess_function, batched=True)

print("Tokenizing test data...")
tokenized_test_data = test_data.map(preprocess_function, batched=True)

print("Sample tokenized data:", tokenized_train_data[0])
print("Done")
exit(1)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

# evaluate

metric = evaluate.load("sacrebleu")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


# train
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# raw_inputs = [
#     "Viên Minh nhìn một cái còn tại nhắm mắt điều tức Cáp Cống, không có lên tiếng quấy rầy, ngẩng đầu nhìn về phía hẻm núi phía trên.",
#     "Hắn cũng không phải là đang lo lắng Ô Bảo bọn người sẽ truy vào đến, trong hạp cốc khu vực rộng rãi, chớ nói Thanh Lang Bang cùng Liệp Cẩu Đường tổng cộng bất quá mấy chục người quy mô, chính là nhiều gấp bội, muốn tìm được bọn hắn cũng khó như lên trời.",
# ]
# inputs = tokenizer(raw_inputs, padding=True, truncation=False, return_tensors="pt")
# print(inputs)

# outputs = model(**inputs)
# print(outputs.logits.shape)
# exit(1)

temp_dir = tempfile.gettempdir()
print("Temp dir:", temp_dir)
output_dir = join(temp_dir, "output")
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    learning_rate=2e-5,
    # Reduce batch size to 8 to
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,  # 15
    predict_with_generate=True,
    fp16=False,
    push_to_hub=False,
    disable_tqdm=True,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_val_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# Evaluation
print("Evaluating on test data...")
eval_result = trainer.evaluate(tokenized_test_data)
print(f"Test results: {eval_result}")


model_dir = join(repo_dir, model_name)
if exists(model_dir):
    print(f"Removing {model_dir}...")
    shutil.rmtree(model_dir)

trainer.save_model(model_dir)

# Remove model zip if exists
if exists(f"{model_dir}.zip"):
    print(f"Removing {model_dir}.zip...")
    os.remove(f"{model_dir}.zip")

# Zip the model_dir and save to repo_dir
print(f"Saving model to {model_dir}.zip...")
shutil.make_archive(model_dir, "zip", model_dir)

# Get 10 test examples
test_examples = tokenized_test_data[:10]

# Use the trained model to translate them
translations = [
    trainer.model.generate(example, num_return_sequences=1) for example in test_examples
]

# Print the translations
for i, (input, translation) in enumerate(zip(test_examples, translations)):
    print(f"Input {i+1}: {tokenizer.decode(input)}")
    print(f"Translation {i+1}: {tokenizer.decode(translation[0])}")
    print("-" * 100)

# print("Remove run directly to save space...")
# if exists(join(model_dir, "runs")):
#     shutil.rmtree(join(model_dir, "runs"))

# print("Remove all checkpoint-xxx folders in the model directory...")
# for file in os.listdir(model_dir):
#     if file.startswith("checkpoint-"):
#         print(f"Removing {model_name}/{file}...")
#         shutil.rmtree(join(model_dir, file))

print("Training time: {:.2f} seconds".format(time.time() - start))
print(f"Test results: {eval_result}")

translator = pipeline("translation", model=model_name)


def _test(text: str, expected: str):
    print("Input    : ", prefix + text)
    # print(
    #     "Output   : ", translator(prefix + text)[0]["translation_text"]
    # )
    print(
        "Output   : ", translator(prefix + text, max_length=100)[0]["translation_text"]
    )
    print("Expected : ", expected)
    print("-" * 100)


# _test(
#     "translate English to French: Legumes share resources with nitrogen-fixing bacteria."
# )
_test(
    "说着，他就扬了扬拎在手中的两个深红色的小酒埕。",
    "Vừa nói, hắn vừa vung vẩy hai chai rượu đỏ thẫm trong tay.",
)


_test(
    "“嘿，你真带酒来了？”鱼翁一看到有酒，眼睛里顿时有了光亮。",
    "“Hắc, ngươi thật mang rượu tới tới?” cá ông vừa nhìn thấy có rượu, con mắt trong tức khắc có rồi ánh sáng.",
)

_test(
    "他忙起身相迎。",
    "Hắn vội vàng đứng dậy nghênh tiếp.",
)

_test(
    "“答应过您的事，怎么可能会忘？”袁铭笑着把酒递了过去，说道。",
    "“Đã đáp ứng chuyện của ngài, làm sao lại quên?” Viên Minh cười nâng cốc đưa tới, nói.",
)
