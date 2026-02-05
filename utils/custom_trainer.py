from transformers import Seq2SeqTrainer


class OptimizerWrapperWithSetToNone:
    def __init__(self, optimizer):
        self._optimizer = optimizer

    def zero_grad(self):
        self._optimizer.zero_grad(set_to_none=True)

    def __getattr__(self, name):
        return getattr(self._optimizer, name)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def create_optimizer(self):
        optimizer = super().create_optimizer()
        return OptimizerWrapperWithSetToNone(optimizer) 