from functools import partial
import tensorflow as tf
import tensorflow_text as text


def _test(translator, src: str, target: str):
    print("Input: ", src)
    translated = translator(src).numpy().decode("utf-8")
    print("Output: ", translated)
    print("Expected: ", target)
    print("-" * 30)


if __name__ == "__main__":
    translator = tf.saved_model.load("zh_vi_transformer")
    _t = partial(_test, translator)
    _t(
        "不幸中的万幸，这些南疆大汉一路上并未再多刁难他们。",
        "Vạn hạnh trong bất hạnh, đám đại hán Nam Cương kia trên đường đi cũng không làm khó dễ gì thêm bọn họ.",
    )
