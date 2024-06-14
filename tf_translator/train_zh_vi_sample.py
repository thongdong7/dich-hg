import os

from tf_translator.train_zh_vi import _train
from .train_config import sample_config

if __name__ == "__main__":
    os.environ["SAMPLE"] = "1"
    _train(save_as="zh_vi_transformer_sample", config=sample_config, epochs=1)
