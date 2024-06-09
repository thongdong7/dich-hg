# -*- coding: utf-8 -*-
import json
import logging
import os
from posixpath import abspath, dirname, join
from time import time
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text as text
from .util import (
    ExportTranslator,
    Translator,
    masked_accuracy,
    masked_loss,
    positional_encoding,
    CustomSchedule,
)
from .transformer import Transformer
from .tokenizers.spm_utils import load_translate_tokenizer
from .tokenizers.config import zh_vi_small_config
from .train_config import MAX_TOKENS

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

repo_dir = dirname(dirname(abspath(__file__)))


def detect_label_dir():
    options = ["label", "tien_hiep/label_new"]
    for option in options:
        label_dir_ = join(repo_dir, option)
        if os.path.exists(label_dir_):
            return label_dir_

    raise FileNotFoundError(f"Cannot find label directory in {options}")


label_dir = detect_label_dir()
print("Label directory:", label_dir)


num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1


def _load_dataset():
    examples, metadata = tfds.load(
        "ted_hrlr_translate/pt_to_en", with_info=True, as_supervised=True
    )

    train_examples, val_examples = examples["train"], examples["validation"]

    return train_examples, val_examples


def gen(folder_contain: str = None, folder_not_contain: str = None):
    # Get all .json files in the label directory
    print(
        f"Start gen with folder_contain={folder_contain}, folder_not_contain={folder_not_contain}"
    )
    print("Label directory:", label_dir)
    for root, dirs, files in os.walk(label_dir):
        if folder_contain and folder_contain not in root:
            continue

        if folder_not_contain and folder_not_contain in root:
            continue

        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                file_id = os.path.relpath(file_path, label_dir).split(".")[0]

                # print("Processing file:", file_id, flush=True)
                with open(file_path, "r") as f:
                    data = json.load(f)

                for i, line in enumerate(data["lines"]):
                    if (
                        line.get("approved")
                        and line.get("chinese")
                        and line.get("dich")
                    ):
                        # Ignore too long line
                        chinese_words = len(line["chinese"]["raw"])
                        if chinese_words > 100:
                            print(
                                f"Line {file_id}_{i} is too long ({chinese_words} words), ignoring...",
                                flush=True,
                            )
                            continue

                        yield line["chinese"]["raw"], line["dich"]


def _get_ds_train():
    # Get all .json files in the label directory which folder not contain pham-nhan-tu-tien
    return tf.data.Dataset.from_generator(
        lambda: gen(folder_not_contain="pham-nhan-tu-tien"),
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ),
    )


def _get_ds_validation():
    # Get all .json files in the label directory which folder contain pham-nhan-tu-tien
    return tf.data.Dataset.from_generator(
        lambda: gen(folder_contain="pham-nhan-tu-tien"),
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ),
    )


def _load_dataset_from_generator():
    train_ds = _get_ds_train()
    val_ds = _get_ds_validation()

    return train_ds, val_ds


def _train(
    save_as: str,
    epochs: int = 1,
    ds_shuffle_buffer_size: int = 20000,
    ds_batch_size: int = 64,
):
    tokenizers = load_translate_tokenizer(zh_vi_small_config)

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=tokenizers.src.get_vocab_size().numpy(),
        target_vocab_size=tokenizers.target.get_vocab_size().numpy(),
        dropout_rate=dropout_rate,
    )

    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    # train_examples, val_examples = _load_dataset()
    train_examples, val_examples = _load_dataset_from_generator()

    print("Train size: ", len(list(train_examples)))
    print("Validation size: ", len(list(val_examples)))

    print(train_examples)
    for src_examples, target_examples in train_examples.batch(3).take(1):
        print("> Examples in source:")
        for pt in src_examples.numpy():
            print(pt.decode("utf-8"))
        print()
    print("> Examples in target:")
    for target in target_examples.numpy():
        print(target.decode("utf-8"))

    print("> This is a batch of strings:")
    for target in target_examples.numpy():
        print(target.decode("utf-8"))

    encoded = tokenizers.target.tokenize(target_examples)

    print("> This is a padded-batch of token IDs:")
    for row in encoded.to_list():
        print(row)

    round_trip = tokenizers.target.detokenize(encoded)

    print("> This is human-readable text:")
    for line in round_trip.numpy():
        print(line.decode("utf-8"))
    # exit(1)

    def prepare_batch(src, target):
        src = tokenizers.src.tokenize(src)  # Output is ragged.
        src = src[:, :MAX_TOKENS]  # Trim to MAX_TOKENS.
        src = src.to_tensor()  # Convert to 0-padded dense Tensor

        target = tokenizers.target.tokenize(target)
        target = target[:, : (MAX_TOKENS + 1)]
        target_inputs = target[:, :-1].to_tensor()  # Drop the [END] tokens
        target_labels = target[:, 1:].to_tensor()  # Drop the [START] tokens

        return (src, target_inputs), target_labels

    def make_batches(ds):
        return (
            ds.shuffle(ds_shuffle_buffer_size)
            .batch(ds_batch_size)
            .map(prepare_batch, tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

    # Create training and validation set batches.
    train_batches = make_batches(train_examples)
    val_batches = make_batches(val_examples)

    # for (src, target), target_labels in train_batches.take(1):
    #     break

    # print(src.shape)
    # print(target.shape)
    # print(target_labels.shape)

    # output = transformer((src, target))
    # print(output.shape)

    # attn_scores = transformer.decoder.dec_layers[-1].last_attn_scores
    # print(attn_scores.shape)  # (batch, heads, target_seq, input_seq)

    # print(transformer.summary())
    # exit(1)

    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    print("Compile...")
    transformer.compile(
        loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy]
    )

    print("Fit...")
    start = time()
    transformer.fit(train_batches, epochs=epochs, validation_data=val_batches)
    print(f"Time to train: {time() - start:.2f}s")

    print("Create translator...")
    translator = Translator(tokenizers, transformer)

    def print_translation(sentence, tokens, ground_truth):
        print(f'{"Input:":15s}: {sentence}')
        print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
        print(f'{"Ground truth":15s}: {ground_truth}')

    sentence = "韩立三叔一见这人，立刻恭恭敬敬的上前施了一个礼。"
    ground_truth = (
        "Tam thúc của Hàn Lập vừa nhìn thấy người mới đến, lập tức cung kính làm lễ."
    )

    translated_text, translated_tokens, attention_weights = translator(
        tf.constant(sentence)
    )
    print_translation(sentence, translated_text, ground_truth)

    print("Export translator")
    translator = ExportTranslator(translator)

    tf.saved_model.save(translator, export_dir=save_as)
    print(f"Translator saved to {save_as}")


if __name__ == "__main__":
    _train(save_as="zh_vi_transformer", epochs=10)
