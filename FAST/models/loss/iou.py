import torch

EPS = 1e-6


def iou(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, n_class: int = 2, reduce: bool = True) -> torch.Tensor:
    """
    Vectorized IoU without Python loops to be compile‑friendly.
    a, b, mask: (N, H, W) or (N, L)
    Returns per-batch mIoU (mean over classes). If reduce, returns scalar mean over batch.
    """
    n, *rest = a.shape
    a = a.view(n, -1)
    b = b.view(n, -1)
    valid = (mask.view(n, -1) == 1)

    # Restrict to valid pixels
    a = torch.where(valid, a, torch.full_like(a, fill_value=-1))
    b = torch.where(valid, b, torch.full_like(b, fill_value=-1))

    # Classes 0..n_class-1
    classes = torch.arange(n_class, device=a.device, dtype=a.dtype)
    # (N, L, C)
    a_onehot = (a.unsqueeze(-1) == classes)
    b_onehot = (b.unsqueeze(-1) == classes)

    inter = (a_onehot & b_onehot).sum(dim=1).to(torch.float32)  # (N, C)
    union = (a_onehot | b_onehot).sum(dim=1).to(torch.float32)  # (N, C)
    iou_per_class = inter / (union + EPS)
    miou_per_batch = iou_per_class.mean(dim=1)  # (N,)
    if reduce:
        return miou_per_batch.mean()
    return miou_per_batch