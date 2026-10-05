import torch

def segmentation_metrics(logits, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = (targets >= 0.5).float()

    dims = (1, 2, 3)

    tp = (preds * targets).sum(dims)
    pred_sum = preds.sum(dims)
    target_sum = targets.sum(dims)

    fp = pred_sum - tp
    fn = target_sum - tp
    union = tp + fp + fn

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (union + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    pixel_acc = (preds == targets).float().mean(dims)

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "pixel_accuracy": pixel_acc.mean().item(),
        "pred_foreground_fraction": preds.mean(dims).mean().item(),
        "gt_foreground_fraction": targets.mean(dims).mean().item(),
    }
