from dataclasses import dataclass
from posixpath import join
from time import time
from tf_translator.tokenizers.config import TranslateConfig
from tf_translator.train_config import TrainConfig
import tensorflow as tf

import csv
from typing import AnyStr
import sentencepiece as spm


def extract_column_to_text(input_csv: AnyStr, output_dir: AnyStr) -> None:
    """
    Extract a specific column from a CSV file and save to a text file.

    :param input_csv: Path to the input CSV file.
    :param output_txt: Path to the output text file.
    :param column: Name of the column to extract.
    """
    src_file = join(output_dir, "src.txt")
    target_file = join(output_dir, "target.txt")
    with open(input_csv, "r") as csv_file, open(
        join(output_dir, "src.txt"), "w"
    ) as txt_src_file, open(join(output_dir, "target.txt"), "w") as txt_target_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            txt_src_file.write(row["src"] + "\n")
            txt_target_file.write(row["target"] + "\n")

    return src_file, target_file


def _train_tokenizer2(
    vocab_size: int, model_type: str, file_path: str, model_prefix: str
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


@dataclass
class TokenizerConfig:
    src_vocab_size: int
    target_vocab_size: int


def _train_tokenizer(data_folder: str, output_folder: str, config: TokenizerConfig):
    src_file, target_file = extract_column_to_text(
        join(data_folder, "train.csv"), output_dir=data_folder
    )

    _train_tokenizer2(
        file_path=src_file,
        model_prefix=join(output_folder, "spm_src"),
        vocab_size=config.src_vocab_size,
        model_type="unigram",
    )

    _train_tokenizer2(
        file_path=target_file,
        model_prefix=join(output_folder, "spm_target"),
        vocab_size=config.target_vocab_size,
        model_type="unigram",
    )


def train(data_folder: str, config: TrainConfig, tokenizer_config: TokenizerConfig):
    """
    Train the model from csv folder
    """
    _train_tokenizer(
        data_folder=data_folder, output_folder=data_folder, config=tokenizer_config
    )

    # Load the dataset
    # train_ds = tf.data.experimental.CsvDataset(
    #     join(data_folder, "train.csv"), [tf.string, tf.string], header=True
    # )
    # val_ds = tf.data.experimental.CsvDataset(
    #     join(data_folder, "validation.csv"), [tf.string, tf.string], header=True
    # )

    # # Train the model
    # train_model(train_ds, val_ds, config)
