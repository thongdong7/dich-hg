from dataclasses import dataclass


MAX_TOKENS = 256


@dataclass
class TrainConfig:
    """
    Configuration class for training a transformer model.

    Attributes:
        num_layers (int): The number of layers in the transformer model.
        d_model (int): The dimensionality of the model's hidden states.
        dff (int): The number of units in the feed-forward layer.
        num_heads (int): The number of attention heads in the multi-head attention layer.
        dropout_rate (float): The dropout rate to apply to the model's layers.
        batch_size (int): The batch size for training.
    """

    # The number of layers in the transformer model.
    num_layers: int
    # The dimensionality of the model's hidden states.
    d_model: int
    # The number of units in the feed-forward layer.
    dff: int
    # The number of attention heads in the multi-head attention layer.
    num_heads: int
    # The dropout rate to apply to the model's layers.
    dropout_rate: float
    # The batch size for training.
    batch_size: int
    # epochs: int
    # learning_rate: float
    # max_tokens: int


# For testing purpose
sample_config = TrainConfig(
    num_layers=4,
    d_model=128,
    dff=512,
    num_heads=8,
    dropout_rate=0.1,
    batch_size=128,
)

base_config = TrainConfig(
    num_layers=6,
    d_model=512,
    dff=2048,
    num_heads=8,
    dropout_rate=0.1,
    batch_size=64,
)
