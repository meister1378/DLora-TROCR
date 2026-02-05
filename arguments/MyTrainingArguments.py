from dataclasses import dataclass, field
from typing import Optional

from transformers import Seq2SeqTrainingArguments
from transformers.training_args import IntervalStrategy


@dataclass
class MyTrainingArguments(Seq2SeqTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # custom args
    # train_csv_path: str = field(default=None)
    # valid_csv_path: str = field(default=None)

    predict_with_generate: bool = field(
        default=True,
        metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )

    # 사용자 편의 별칭: --eval_strategy (Hf 표준은 --evaluation_strategy)
    eval_strategy: Optional[str] = field(
        default=None,
        metadata={
            "help": "Alias of evaluation_strategy. One of: 'no' | 'steps' | 'epoch'"
        },
    )

    def __post_init__(self):  # type: ignore[override]
        # 1) 별칭 사전 정규화: base __post_init__에서 self.eval_strategy를 Enum으로 캐스팅하므로 None 방지
        if self.eval_strategy is None:
            current = getattr(self, "evaluation_strategy", None)
            if isinstance(current, IntervalStrategy):
                self.eval_strategy = current.value
            elif current is None:
                self.eval_strategy = "no"
            else:
                self.eval_strategy = str(current)
        else:
            val = str(self.eval_strategy).strip().lower()
            if val in ("none", "off"):
                val = "no"
            if val == "step":
                val = "steps"
            if val == "epochs":
                val = "epoch"
            self.eval_strategy = val

        # 2) 부모 초기화 수행 (여기서 eval_strategy가 Enum으로 확정됨)
        super().__post_init__()

        # 3) 별칭을 공식 필드에 반영(안전하게 유지)
        try:
            if isinstance(self.eval_strategy, IntervalStrategy):
                self.evaluation_strategy = self.eval_strategy
            else:
                self.evaluation_strategy = IntervalStrategy(self.eval_strategy)  # type: ignore[arg-type]
        except Exception:
            pass
