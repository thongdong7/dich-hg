import os
from posixpath import join
from typing import List
import tensorflow_text as text
import tensorflow as tf
import pathlib
from .config import Config, TranslateConfig


def get_tokenizer_from_config(config: Config):
    with open(config.model_file, "rb") as f:
        model = f.read()

    return text.SentencepieceTokenizer(model=model, add_bos=True, add_eos=True)


class CustomTokenizer(tf.Module):
    def __init__(self, model_file_path: str):
        with open(model_file_path, "rb") as f:
            model = f.read()

        self.tokenizer = text.SentencepieceTokenizer(
            model=model, add_bos=True, add_eos=True
        )

        ## Create the signatures for export:

        # Include a tokenize signature for a batch of strings.
        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string)
        )

        # Include `detokenize` and `lookup` signatures for:
        #   * `Tensors` with shapes [tokens] and [batch, tokens]
        #   * `RaggedTensors` with shape [batch, tokens]
        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int32)
        )
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32)
        )

        # These `get_*` methods take no arguments
        self.get_vocab_size.get_concrete_function()

    @tf.function
    def tokenize(self, strings):
        return self.tokenizer.tokenize(strings)

    @tf.function
    def detokenize(self, tokenized):
        return self.tokenizer.detokenize(tokenized)

    @tf.function
    def get_vocab_size(self):
        return self.tokenizer.vocab_size()
        # return tf.shape(self.vocab)[0]


def export_translate_tokenizer(config: TranslateConfig):
    tokenizers = tf.Module()
    tokenizers.__setattr__("src", CustomTokenizer(config.source.model_file))
    tokenizers.__setattr__("target", CustomTokenizer(config.target.model_file))
    tf.saved_model.save(tokenizers, config.model_name)


def load_translate_tokenizer(config: TranslateConfig):
    tokenizer_root = os.environ.get("TOKENIZER_ROOT", ".")
    return tf.saved_model.load(join(tokenizer_root, config.model_name))
