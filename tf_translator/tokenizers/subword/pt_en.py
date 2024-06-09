import collections
import os
import pathlib
import re
import string
import sys
import tempfile
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
from .utils import (
    add_start_end,
    cleanup_text,
    CustomTokenizer,
    reserved_tokens,
    export_tokenizer,
)

tf.get_logger().setLevel("ERROR")
pwd = pathlib.Path.cwd()


def _load_data():
    # Download dataset
    print("Load dataset")
    start = time()
    examples, metadata = tfds.load(
        "ted_hrlr_translate/pt_to_en", with_info=True, as_supervised=True
    )
    train_examples, val_examples = examples["train"], examples["validation"]

    for pt, en in train_examples.take(1):
        print("Portuguese: ", pt.numpy().decode("utf-8"))
        print("English:   ", en.numpy().decode("utf-8"))

    train_en = train_examples.map(lambda pt, en: en)
    train_pt = train_examples.map(lambda pt, en: pt)

    print(f"Time to load dataset: {time() - start:.2f}s")

    return train_en, train_pt, train_examples


bert_tokenizer_params = dict(lower_case=True)

bert_vocab_args = dict(
    # The target vocabulary size
    vocab_size=8000,
    # Reserved tokens that must be included in the vocabulary
    reserved_tokens=reserved_tokens,
    # Arguments for `text.BertTokenizer`
    bert_tokenizer_params=bert_tokenizer_params,
    # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
    learn_params={},
)


def _train():
    train_en, train_pt, train_examples = _load_data()

    print("Build tokenizer")

    def _build_vocab(train):
        return bert_vocab.bert_vocab_from_dataset(
            train.batch(1000).prefetch(2), **bert_vocab_args
        )

    print("Build Portuguese vocab")
    start = time()
    pt_vocab = bert_vocab.bert_vocab_from_dataset(
        train_pt.batch(1000).prefetch(2), **bert_vocab_args
    )

    def write_vocab_file(filepath, vocab):
        with open(filepath, "w") as f:
            for token in vocab:
                print(token, file=f)

    write_vocab_file("pt_vocab.txt", pt_vocab)
    print(f"Time to build Portuguese vocab: {time() - start:.2f}s")

    print("Build English vocab")
    start = time()
    en_vocab = bert_vocab.bert_vocab_from_dataset(
        train_en.batch(1000).prefetch(2), **bert_vocab_args
    )
    write_vocab_file("en_vocab.txt", en_vocab)
    print(f"Time to build English vocab: {time() - start:.2f}s")


def _get_vocab(vocab_file):
    vocab = []
    with open(vocab_file) as f:
        for line in f:
            vocab.append(line.strip())
    return vocab


def _test(train_examples):
    # Build tokenizer
    pt_tokenizer = text.BertTokenizer("pt_vocab.txt", **bert_tokenizer_params)
    en_tokenizer = text.BertTokenizer("en_vocab.txt", **bert_tokenizer_params)
    for pt_examples, en_examples in train_examples.batch(3).take(1):
        for ex in en_examples:
            print(ex.numpy())

    # Tokenize the examples -> (batch, word, word-piece)
    token_batch = en_tokenizer.tokenize(en_examples)

    # Print token_batch before merge
    print("Token_match before merge")
    for ex in token_batch.to_list():
        print(ex)

    print("Token after merge")
    # Merge the word and word-piece axes -> (batch, tokens)
    token_batch = token_batch.merge_dims(-2, -1)

    for ex in token_batch.to_list():
        print(ex)

    # Lookup each token id in the vocabulary.
    en_vocab = _get_vocab("en_vocab.txt")
    txt_tokens = tf.gather(en_vocab, token_batch)
    # Join with spaces.
    print("Result when just join use vocab")
    print(tf.strings.reduce_join(txt_tokens, separator=" ", axis=-1))

    words = en_tokenizer.detokenize(token_batch)
    print("Result when use detokenize - better")
    print(tf.strings.reduce_join(words, separator=" ", axis=-1))

    words = en_tokenizer.detokenize(add_start_end(token_batch))
    print("Result when add start and end token")
    print(tf.strings.reduce_join(words, separator=" ", axis=-1))

    token_batch = en_tokenizer.tokenize(en_examples).merge_dims(-2, -1)
    words = en_tokenizer.detokenize(token_batch)
    print(cleanup_text(reserved_tokens, words).numpy())


def _export():
    export_tokenizer(
        model_name="ted_hrlr_translate_pt_en_converter",
        vocab_map={"pt": "pt_vocab.txt", "en": "en_vocab.txt"},
    )


def _test_export():
    model_name = "ted_hrlr_translate_pt_en_converter"
    reloaded_tokenizers = tf.saved_model.load(model_name)
    pt = reloaded_tokenizers.pt
    en = reloaded_tokenizers.en

    # Tokenize
    token_batch = pt.tokenize(["O Brasil é um país bonito"])
    words = pt.detokenize(token_batch)
    print(words.numpy())
    # [b'o brasil e um pais bonito']

    # Lookup
    token_ids = en.tokenize(["The Brazil is a beautiful country"])
    words = en.lookup(token_ids)
    print(words)
    # <tf.RaggedTensor [[b'[START]', b'the', b'brazil', b'is', b'a', b'beautiful', b'country',
    #   b'[END]']]>

    # Detokenize
    words = en.detokenize(token_ids)
    print(words.numpy())
    # [b'the brazil is a beautiful country']


if __name__ == "__main__":
    _train()
    _export()
