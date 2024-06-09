from genericpath import exists
import sys
from time import time
from typing import List
import sentencepiece as spm
from .config import Config, vi_small_config, zh_small_config
from datasets import load_dataset


def _train(
    file_path: str, model_prefix: str, vocab_size: int, model_type: str = "unigram"
):
    print("Vocab size: ", vocab_size)
    print("Model type: ", model_type)

    start = time()
    spm.SentencePieceTrainer.train(
        input=file_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )

    print("Vocab size: ", vocab_size)
    print("Model type: ", model_type)
    print(f"Train Tokenizer Time: {time() - start:.2f}s")


def _train_for_config(config: Config):
    if exists(config.model_file):
        return

    print(
        f"Train tokenizer for {config.lang} ({config.ds_size/1000}k, {config.vocab_size/1000}k)"
    )
    _train(
        file_path=config.source_file,
        model_prefix=config.model_prefix,
        vocab_size=config.vocab_size,
    )


ignore_prefix = {"http://www.alrage.net"}


def _is_ignore_url(url: str):
    for prefix in ignore_prefix:
        if url.startswith(prefix):
            return True
    return False


def _load_data(configs: List[Config]):
    for config in configs:
        if exists(config.source_file):
            continue

        start = time()
        print(f"Load dataset for {config.lang} ({config.ds_size_str})")
        ds = load_dataset("allenai/c4", config.lang, streaming=True)
        with open(config.source_file, "w", encoding="utf-8") as f:
            count = 0
            for type_ in ["train", "validation"]:
                print("type: ", type_)

                if count >= config.ds_size:
                    break

                for sample in ds[type_]:
                    url = sample["url"]

                    if _is_ignore_url(url):
                        continue

                    text = sample["text"]
                    f.write(text + "\n")
                    count += 1
                    print(f"\r{config.lang}/{type_}:{count}", end="", flush=True)
                    sys.stdout.flush()

                    if count >= config.ds_size:
                        break

        print(f"\nDownload Data Time: {time() - start:.2f}s")


if __name__ == "__main__":
    # _train("vi_zh_1k.txt", model_prefix="spm_vi_zh_1k", vocab_size=8000)

    _load_data([vi_small_config, zh_small_config])

    _train_for_config(vi_small_config)
    _train_for_config(zh_small_config)
