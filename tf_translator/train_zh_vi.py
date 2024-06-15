# -*- coding: utf-8 -*-
from genericpath import exists
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
from .train_config import MAX_TOKENS, base_config, TrainConfig


repo_dir = dirname(dirname(abspath(__file__)))


def detect_label_dir():
    dataset_root = os.environ.get("DATASET_ROOT")
    if dataset_root and exists(dataset_root):
        return dataset_root

    options = ["label", "tien_hiep/label_new"]
    for option in options:
        label_dir_ = join(repo_dir, option)
        if os.path.exists(label_dir_):
            return label_dir_

    raise FileNotFoundError(f"Cannot find label directory in {options}")


label_dir = detect_label_dir()
print("Label directory:", label_dir)


# def _load_dataset():
#     examples, metadata = tfds.load(
#         "ted_hrlr_translate/pt_to_en", with_info=True, as_supervised=True
#     )

#     train_examples, val_examples = examples["train"], examples["validation"]

#     return train_examples, val_examples


# def gen(name: str):
#     file = f"{name}.txt"
#     is_sample = os.environ.get("SAMPLE") == "1"

#     file_path = os.path.join(label_dir, file)

#     data = []
#     count = 0
#     with open(file_path, "r") as f:
#         for line in f:
#             if not line.strip():
#                 continue

#             parts = json.loads(line.strip())
#             data.append((parts[0], parts[1]))
#             count += 1
#             if is_sample and count > 100:
#                 break
#     return data


# def split_src_target(example):
#     return example[0], example[1]


def parse_line(line):
    # return json.loads(line.strip())
    line = tf.strings.strip(line)
    return tf.io.decode_json_example(line)


def _get_ds_train():
    return tf.data.TextLineDataset(join(label_dir, "train.txt")).map(parse_line)


def _get_ds_validation():
    return tf.data.TextLineDataset(join(label_dir, "validation.txt")).map(parse_line)


def _load_dataset_from_generator():
    train_ds = _get_ds_train()
    val_ds = _get_ds_validation()

    return train_ds, val_ds


def _train(
    save_as: str,
    config: TrainConfig,
    ds_shuffle_buffer_size: int = 20000,
):
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
    exit(1)
    tokenizers = load_translate_tokenizer(zh_vi_small_config)

    print("-" * 20)
    print(f"Input vocab size: {tokenizers.src.get_vocab_size()}")
    print(f"Target vocab size: {tokenizers.target.get_vocab_size()}")
    print(f"Number layers: {config.num_layers}")
    print(f"d_model: {config.d_model}")
    print(f"dff: {config.dff}")
    print(f"num_heads: {config.num_heads}")
    print(f"dropout_rate: {config.dropout_rate}")
    print(f"batch_size: {config.batch_size}")
    print("-" * 20)
    transformer = Transformer(
        num_layers=config.num_layers,
        d_model=config.d_model,
        num_heads=config.num_heads,
        dff=config.dff,
        input_vocab_size=tokenizers.src.get_vocab_size().numpy(),
        target_vocab_size=tokenizers.target.get_vocab_size().numpy(),
        dropout_rate=config.dropout_rate,
    )

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
            .batch(config.batch_size)
            .map(prepare_batch, tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

    # Create training and validation set batches.
    train_batches = make_batches(train_examples)
    val_batches = make_batches(val_examples)

    for (src, target), target_labels in train_batches.take(3):
        # print("src shape", src.shape)
        # print("src ", src)
        # print("src ", src[0])
        # for item in src:
        #     print("item", item)
        break

    print("src", src.shape)
    print("target", target.shape)
    print("target label", target_labels.shape)

    output = transformer((src, target))
    print("output", output.shape)

    attn_scores = transformer.decoder.dec_layers[-1].last_attn_scores
    print("attn_scores", attn_scores.shape)  # (batch, heads, target_seq, input_seq)
    # exit(1)

    learning_rate = CustomSchedule(config.d_model)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    print("Compile...")
    transformer.compile(
        loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy]
    )

    print(transformer.summary())

    print(f"Fit (epochs={config.epochs})...")
    start = time()
    transformer.fit(train_batches, epochs=config.epochs, validation_data=val_batches)
    print("-" * 20)
    print(f"Config: {config}")
    print(transformer.summary())
    print(f"Time to train: {time() - start:.2f}s", flush=True)

    print("Create translator...")
    start = time()
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
    print(f"Time to create translator: {time() - start:.2f}s")

    # loaded_translator = tf.saved_model.load(save_as)
    # print("Loaded translator")


if __name__ == "__main__":
    _train(save_as="zh_vi_transformer", config=base_config)
