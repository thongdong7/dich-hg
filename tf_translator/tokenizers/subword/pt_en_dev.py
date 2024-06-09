from posixpath import abspath, dirname, join
import tensorflow as tf
from .utils import export_tokenizer

model_name = "ted_hrlr_translate_pt_en_converter"
file_dir = dirname(abspath(__file__))


def _export():
    export_tokenizer(
        model_name=model_name,
        vocab_map={
            "pt": join(file_dir, "pt_vocab.txt"),
            "en": join(file_dir, "en_vocab.txt"),
        },
    )


def _test_export():
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
    # print(words)
    # <tf.RaggedTensor [[b'[START]', b'the', b'brazil', b'is', b'a', b'beautiful', b'country',
    #   b'[END]']]>

    # Detokenize
    words = en.detokenize(token_ids)
    print(words.numpy()[0].decode("utf-8"))
    # [b'the brazil is a beautiful country']

    print("Vocab size: ", en.get_vocab_size().numpy())


if __name__ == "__main__":
    # _export()
    _test_export()
