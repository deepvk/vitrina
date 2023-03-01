from transformers import BertConfig, BertForSequenceClassification
from torch import nn
from loguru import logger


def get_model(
    vtr: bool,
    vocab_size: int = 30_000,
    max_position_embeddings: int = 512,
    hidden_size: int = 768,
    num_attention_heads: int = 12,
    num_hidden_layers: int = 1,
    type_vocab_size: int = 1,
    dropout: float = 0.0,
    num_classes: int = 2,
):
    if vtr:
        logger.info(f"Initializing VTR classifier | hidden size: {hidden_size}, # layers: {num_hidden_layers}")
        return nn.Linear(hidden_size, num_classes)

    else:
        logger.info(f"Initializing vanilla BERT classifier | hidden size: {hidden_size}, # layers: {num_hidden_layers}")

        model_config = BertConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            type_vocab_size=type_vocab_size,
            num_labels=num_classes,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            return_dict=False,
        )
        return BertForSequenceClassification(model_config)
