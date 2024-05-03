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

start = time.time()

# Config
# number_of_books = 100
model_name = "convert_dich"
num_train_epochs = 30

per_device_train_batch_size = 16
per_device_eval_batch_size = 16

checkpoint = "VietAI/vit5-base"

print("Number of train epochs:", num_train_epochs)

repo_dir = dirname(dirname(abspath(__file__)))
print("Repo directory:", repo_dir)
os.chdir(repo_dir)

label_dir = join(repo_dir, "tien_hiep/label")
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
                print("Processing file:", file_id)
                with open(file_path, "r") as f:
                    data = json.load(f)
                    try:
                        approve_lines = data.get("approvedLines", [])
                        convert_lines = data["convert"]["data"]["lines"]
                        dich_lines = data["dich"]["data"]["lines"]
                    except Exception as e:
                        print("Error processing file:", file)
                        print(e)
                        exit(1)

                    for line in approve_lines:
                        yield {
                            "id": f"{file_id}_{line}",
                            "translation": {
                                "convert": convert_lines[line],
                                "dich": dich_lines[line],
                            },
                        }


ds = Dataset.from_generator(gen)
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

# load dataset
# all_books = load_dataset("opus_books", "en-fr")


# # Create a small dataset
# books = DatasetDict({"train": all_books["train"].select(range(number_of_books))})

# books = books["train"].train_test_split(test_size=0.2)
# print(books["train"][0])
# exit(0)
# books["train"][0]

# preprocess


# checkpoint = "google-t5/t5-small"
tokenizer_model = checkpoint
# tokenizer_model = "vinai/phobert-base-v2"


def load_tokenizer():
    print(f"Loading Tokenizer {tokenizer_model}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    return tokenizer


tokenizer = load_tokenizer()

# source_lang = "en"
# target_lang = "fr"

source_lang = "convert"
target_lang = "dich"

if "google-t5" in tokenizer_model:
    prefix = "translate English to French: "
else:
    prefix = "dịch sang tiếng Việt: "


def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, truncation=True)
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
# print("Done")
# exit(1)

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

temp_dir = tempfile.gettempdir()
print("Temp dir:", temp_dir)
output_dir = join(temp_dir, "output")
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
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

print("Remove run directly to save space...")
if exists(join(model_dir, "runs")):
    shutil.rmtree(join(model_dir, "runs"))

print("Remove all checkpoint-xxx folders in the model directory...")
for file in os.listdir(model_dir):
    if file.startswith("checkpoint-"):
        print(f"Removing {model_name}/{file}...")
        shutil.rmtree(join(model_dir, file))

print("Training time: {:.2f} seconds".format(time.time() - start))
print(f"Test results: {eval_result}")

translator = pipeline("translation", model="convert_dich")


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
    "Hàn Lập đem thể nội trong kinh mạch năng lượng chảy chậm rãi địa thu về đan điền",
    "Hàn Lập đưa toàn bộ năng lượng trong cơ thể theo kinh mạch chảy chậm về đan điền",
)


_test(
    "Hàn Lập đem thể nội trong kinh mạch năng lượng chảy chậm rãi địa thu về đan điền, đây là hắn hôm nay liên tiếp vận hành đệ thất cái đại chu thiên tuần hoàn, hắn biết rõ chính mình thân thể đã đạt đến có thể thừa nhận được cực hạn,",
    "Hàn Lập đưa toàn bộ năng lượng trong cơ thể theo kinh mạch trở về đan điền, vậy là hôm nay hắn đã vận hành được bảy chu thiên tuần hoàn, hắn cũng tự biết rằng đây đã là giới hạn của bản thân hắn tại thời điểm này",
)

_test(
    "Nhị Lăng Tử mở to hai mắt, thẳng tắp nhìn cỏ tranh cùng bùn nhão dán thành đen nóc nhà,",
    "‘Anh ngố’ nhìn chằm chằm vào nóc nhà được tạo thành từ cỏ dại và bùn trộn lẫn.",
)

_test(
    "Một đạo gầy yếu bóng dáng nhảy vào mặt sông, tóe lên một mảnh bọt nước.",
    "Một đạo bóng dáng gầy yếu nhảy vào mặt sông, tóe lên một mảnh bọt nước.",
)
