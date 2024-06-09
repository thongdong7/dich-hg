from functools import partial
import tensorflow_text as text
from .config import (
    Config,
    vi_small_config,
    zh_small_config,
    zh_vi_small_config,
)
from .spm_utils import (
    export_translate_tokenizer,
    get_tokenizer_from_config,
    CustomTokenizer,
    load_translate_tokenizer,
)
import tensorflow as tf


def _test(tokenizer: text.SentencepieceTokenizer, text: str):
    print("Input: ", text)
    ids = tokenizer.tokenize(text)
    print("IDs: ", ids)
    output = tokenizer.detokenize(ids).numpy().decode("utf-8")
    print("Tokens: ", output)
    text = text.replace("，", ",").replace("：", ":")
    if output != text:
        print("\033[91mOutput not equal to input!!!!\033[0m")
        # show diff
        for i in range(len(text)):
            if text[i] != output[i]:
                print(f"\033[91mDiff at {i}: '{text[i]}' vs '{output[i]}'\033[0m")
    print("-" * 30)


if __name__ == "__main__":
    vi_tokenizer = get_tokenizer_from_config(vi_small_config)
    zh_tokenizer = get_tokenizer_from_config(zh_small_config)

    t_vi = partial(_test, vi_tokenizer)
    t_zh = partial(_test, zh_tokenizer)

    t_vi(
        "Loại dã thú như gấu đen này có ý thức lãnh thổ rất mạnh, bình thường rất ít khi rời khỏi khu vực sinh sống. Viên Minh muốn quay lại nơi này bởi mục tiêu săn giết của hắn chính là con gấu kia."
    )
    t_zh(
        "直到那年轻人走到近前，老者才出声招呼：“袁铭，你来的太是时候了，我这一锅鱼刚炖上，还没来得及下筷子，你就到了。”"
    )

    print("vi vocab size: ", vi_tokenizer.vocab_size().numpy())
    print("zh vocab size: ", zh_tokenizer.vocab_size().numpy())

    export_translate_tokenizer(zh_vi_small_config)

    loaded_tokenizer = load_translate_tokenizer(zh_vi_small_config)

    ids = loaded_tokenizer.target.tokenize(["Viên Minh"])
    print("ids: ", ids)
    output = loaded_tokenizer.target.detokenize(ids).numpy()[0].decode("utf-8")
    print("output: ", output)
    print("vocab size: ", loaded_tokenizer.target.get_vocab_size().numpy())
