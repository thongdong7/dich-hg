import csv
from typing import List, Tuple

from tf_translator.train import TokenizerConfig, train
from tf_translator.train_config import TrainConfig


def _write_to_csv_file(data: List[Tuple[str, str]], file_path: str):
    with open(file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["src", "target"])
        for src, target in data:
            writer.writerow([src, target])


if __name__ == "__main__":
    data_folder = ""
    sample_train = [
        [
            "hello",
            "xin chao",
        ]
    ]
    sample_validation = [
        [
            "good bye",
            "tam biet",
        ]
    ]
    src_vocab_size = 10
    target_vocab_size = 10

    _write_to_csv_file(
        data=sample_train,
        file_path=f"{data_folder}/train.csv",
    )
    _write_to_csv_file(
        data=sample_validation,
        file_path=f"{data_folder}/validation.csv",
    )

    train(
        data_folder=data_folder,
        config=TrainConfig(
            src_vocab_size=1,  # we dont use this number
            target_vocab_size=1,  # we dont use this number
        ),
        tokenizer_config=TokenizerConfig(
            src_vocab_size=src_vocab_size,
            target_vocab_size=src_vocab_size,
        ),
    )
