from genericpath import exists
import json
import os
from posixpath import abspath, dirname, join
from datasets import Dataset


# Save the start time to measure execution time of the script


repo_dir = dirname(dirname(abspath(__file__)))
print("Repo directory:", repo_dir)
os.chdir(repo_dir)

label_dir = join(repo_dir, "tien_hiep/label_new")
if not exists(label_dir):
    label_dir = join(repo_dir, "label")

if not exists(label_dir):
    label_dir = "/kaggle/input/dich-hg1/label"

print("Label directory:", label_dir)


def get_full_dataset():
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
                            # Ignore too long line
                            convert_words = len(line["convert"].split(" "))
                            if convert_words > 100:
                                print(
                                    f"Line {file_id}_{i} is too long ({convert_words} words), ignoring...",
                                    flush=True,
                                )
                                continue

                            yield {
                                "id": f"{file_id}_{i}",
                                "translation": {
                                    "convert": line["convert"],
                                    "dich": line["dich"],
                                },
                            }

    return Dataset.from_generator(gen)


def get_sample_dataset():
    return get_full_dataset().select(range(100))
