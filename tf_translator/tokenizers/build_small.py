from .config import small
from .download_data import download_data_for_config
from .train_tokenizer_spm import _train_for_config

if __name__ == "__main__":
    # download_data_for_config(small)

    _train_for_config(small)
