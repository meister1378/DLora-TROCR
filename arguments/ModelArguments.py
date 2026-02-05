from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    encoder_model_name_or_path: str = field(default=None)
    decoder_model_name_or_path: str = field(default=None)
    model_name_or_path: str = field(default=None)
    num_beams: int = field(default=10)
    max_length: int = field(default=32)
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "attention implementation, e.g. `flash_attention_2`"},
    )