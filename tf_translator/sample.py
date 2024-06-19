import csv
import os
from subprocess import run
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
    data_folder = os.getcwd()
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
            num_layers=4,
            d_model=128,
            dff=512,
            num_heads=8,
            dropout_rate=0.1,
            batch_size=128,
            epochs=1,
        ),
        tokenizer_config=TokenizerConfig(
            src_vocab_size=src_vocab_size,
            target_vocab_size=src_vocab_size,
        ),
    )

    # Show the output
    run(["ls", "-lh", data_folder], check=True)
