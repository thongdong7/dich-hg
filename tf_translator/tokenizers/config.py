from dataclasses import dataclass


small = dict(
    ds_size=10000,
    vocab_size=16000,
)


@dataclass
class Config:
    lang: str
    ds_size: int
    vocab_size: int

    def __post_init__(self):
        self.ds_size_str = f"{int(self.ds_size/1000)}k"
        self.vocab_size_str = f"{int(self.vocab_size/1000)}k"

        self.source_file = f"{self.lang}_{self.ds_size_str}.txt"
        self.model_prefix = f"spm_{self.lang}_{self.ds_size_str}_{self.vocab_size_str}"
        self.model_file = f"{self.model_prefix}.model"


@dataclass
class TranslateConfig:
    source: Config
    target: Config

    def __post_init__(self):
        self.model_name = f"{self.source.lang}_{self.target.lang}_{self.source.vocab_size_str}_{self.target.vocab_size_str}"


vi_small_config = Config(lang="vi", ds_size=10000, vocab_size=8000)
zh_small_config = Config(lang="zh", ds_size=10000, vocab_size=10000)

zh_vi_small_config = TranslateConfig(source=zh_small_config, target=vi_small_config)
