import torch


def ohem_single(score: torch.Tensor, gt_text: torch.Tensor, training_mask: torch.Tensor) -> torch.Tensor:
    """
    Compile-friendly OHEM without Python int()/item() graph breaks.
    - score, gt_text, training_mask: (H, W)
    Returns: (1, H, W) float mask
    """
    device = score.device
    dtype = score.dtype

    base_mask = (training_mask > 0.5)
    pos_mask = (gt_text > 0.5) & base_mask
    pos_num = pos_mask.sum()  # tensor scalar

    neg_mask = (gt_text <= 0.5)
    neg_num_all = neg_mask.sum()  # tensor scalar
    neg_num = torch.minimum(pos_num * 3, neg_num_all)  # tensor scalar

    # Build threshold for negatives using sort+gather with tensor index (no Python int)
    neg_score = score[neg_mask]
    # If there are no negatives, avoid sort overhead by short-circuit via where()
    # idx = clamp(neg_num-1, min=0); threshold is inf when neg_num==0 to select none
    k_idx = torch.clamp(neg_num.to(torch.long) - 1, min=0)
    if neg_score.numel() > 0:
        sorted_vals, _ = torch.sort(neg_score, descending=True)
        # gather expects index tensor on same device
        gather_idx = k_idx.to(device)
        kth_val = sorted_vals.gather(0, gather_idx)
        threshold = torch.where(neg_num > 0, kth_val, torch.tensor(float('inf'), device=device, dtype=dtype))
    else:
        threshold = torch.tensor(float('inf'), device=device, dtype=dtype)

    selected_neg = (score >= threshold) & neg_mask & (neg_num > 0)
    selected_logic = torch.where(pos_num == 0, base_mask, (pos_mask | selected_neg) & base_mask)
    selected_mask = selected_logic.reshape(1, gt_text.shape[0], gt_text.shape[1]).to(dtype)
    return selected_mask


def ohem_batch(scores: torch.Tensor, gt_texts: torch.Tensor, training_masks: torch.Tensor) -> torch.Tensor:
    """
    scores, gt_texts, training_masks: (N, H, W)
    Returns: (N, 1, H, W)
    """
    batch = scores.shape[0]
    masks = [
        ohem_single(scores[i], gt_texts[i], training_masks[i])
        for i in range(batch)
    ]
    return torch.cat(masks, dim=0).float()
