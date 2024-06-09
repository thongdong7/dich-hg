from functools import partial
import sentencepiece as spm
from .config import small


def _get_processor(model_prefix: str):
    return spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")


def _get_processor_for_config(config: dict):
    model_prefix = f"spm_vi_zh_{config['ds_size']}_{config['vocab_size']}"
    return _get_processor(model_prefix)


def _test(processor: spm.SentencePieceProcessor, text: str):
    print("Input  : ", text)
    tokens = processor.encode_as_pieces(text, add_bos=True, add_eos=True)
    ids = processor.encode_as_ids(text, add_bos=True, add_eos=True)
    print(f"Tokens ({len(tokens)}): ", tokens)
    print(f"IDs ({len(ids)}): ", ids)
    decode = processor.decode(ids)
    print(f"Decode: {decode}")
    text_ = text.replace("，", ",").replace("：", ":")
    if decode != text_:
        print("\033[91m Decode not equal to text!!!! \033[0m")
        # show diff
        for i in range(len(text_)):
            if text_[i] != decode[i]:
                print(f"\033[91m Diff at {i}: '{text_[i]}' vs '{decode[i]}' \033[0m")

    print("-" * 30)


if __name__ == "__main__":
    processor = _get_processor_for_config(small)
    print(
        "bos:",
        processor.bos_id(),
        "eos:",
        processor.eos_id(),
        "pad:",
        processor.pad_id(),
    )
    print("unk:", processor.unk_id())
    print()
    t = partial(_test, processor)
    t("Viên Minh")
    # t(
    #     "Viên Minh nhìn một cái còn tại nhắm mắt điều tức Cáp Cống, không có lên tiếng quấy rầy, ngẩng đầu nhìn về phía hẻm núi phía trên.",
    # )

    # t("一道瘦弱身影跃入河面，溅起一片水花。")
    # t(
    #     "一名须发皆白的老者，坐在茅屋檐下，身前的火炉上架着一口陶锅，里面炖煮着肥美的鱼肉，咕嘟嘟翻滚着肉汤，散发着诱人的香气。",
    # )
    # t("老者远远看到一道人影出现在山谷入口方向，脸上顿时多了几分笑意。")
    # t(
    #     """Thẳng đến người tuổi trẻ kia đi đến phụ cận, lão giả mới lên tiếng chiêu hô: “Viên Minh, ngươi tới quá là thời điểm, ta này một nồi cá vừa ninh thượng, còn chưa kịp hạ đũa, ngươi liền đến rồi.”"""
    # )
    # t(
    #     "直到那年轻人走到近前，老者才出声招呼：“袁铭，你来的太是时候了，我这一锅鱼刚炖上，还没来得及下筷子，你就到了。”"
    # )
