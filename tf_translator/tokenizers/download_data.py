from time import time
from datasets import load_dataset
import sys
from .config import small
import os

ignore_prefix = {"http://www.alrage.net"}


def _is_ignore_url(url: str):
    for prefix in ignore_prefix:
        if url.startswith(prefix):
            return True
    return False


def download_data(file_path: str, max_size: int = 1000000):
    print("Load dataset")
    vi = load_dataset("allenai/c4", "vi", streaming=True)
    zh = load_dataset("allenai/c4", "zh", streaming=True)

    ds = {"vi": vi, "zh": zh}

    start = time()
    with open(file_path, "w", encoding="utf-8") as f:
        for lang in ["vi", "zh"]:
            print("lang: ", lang)
            count = 0
            for type_ in ["train", "validation"]:
                print("type: ", type_)

                if count >= max_size:
                    break

                for sample in ds[lang][type_]:
                    url = sample["url"]

                    if _is_ignore_url(url):
                        continue

                    text = sample["text"]
                    f.write(text + "\n")
                    count += 1
                    print(f"\r{lang}/{type_}:{count}", end="", flush=True)
                    sys.stdout.flush()

                    if count >= max_size:
                        break

    print(f"\nDownload Data Time: {time() - start:.2f}s")

    # Print file size

    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / 1024 / 1024
    print(f"File size: {file_size_mb:.2f}MB")


def download_data_for_config(config: dict):
    max_size = config["ds_size"]

    download_data(file_path=f"vi_zh_{max_size}.txt", max_size=max_size)


if __name__ == "__main__":
    # download_data("vi_zh_1k.txt", max_size=1000)

    download_data_for_config(small)
