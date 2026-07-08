import torch
import torch.nn.functional as F



def _as_float_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.float().reshape(-1)


def calculate_accuracy(y_true: torch.Tensor, y_pred_logits: torch.Tensor, threshold: float = 0.0) -> float:
    y_true = _as_float_tensor(y_true)
    y_pred_labels = (_as_float_tensor(y_pred_logits) >= threshold).float()
    correct = (y_pred_labels == y_true).sum().item()
    return correct / len(y_true) if len(y_true) > 0 else 0.0


def calculate_precision(y_true: torch.Tensor, y_pred_logits: torch.Tensor, threshold: float = 0.0) -> float:
    y_true = _as_float_tensor(y_true)
    y_pred_labels = (_as_float_tensor(y_pred_logits) >= threshold).float()
    true_positives = ((y_pred_labels == 1) & (y_true == 1)).sum().item()
    predicted_positives = (y_pred_labels == 1).sum().item()
    return true_positives / predicted_positives if predicted_positives > 0 else 0.0


def calculate_recall(y_true: torch.Tensor, y_pred_logits: torch.Tensor, threshold: float = 0.0) -> float:
    y_true = _as_float_tensor(y_true)
    y_pred_labels = (_as_float_tensor(y_pred_logits) >= threshold).float()
    true_positives = ((y_pred_labels == 1) & (y_true == 1)).sum().item()
    actual_positives = (y_true == 1).sum().item()
    return true_positives / actual_positives if actual_positives > 0 else 0.0


def calculate_f1(y_true: torch.Tensor, y_pred_logits: torch.Tensor, threshold: float = 0.0) -> float:
    precision = calculate_precision(y_true, y_pred_logits, threshold=threshold)
    recall = calculate_recall(y_true, y_pred_logits, threshold=threshold)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def calculate_fbeta(y_true: torch.Tensor, y_pred_logits: torch.Tensor, beta: float = 0.5, threshold: float = 0.0) -> float:
    precision = calculate_precision(y_true, y_pred_logits, threshold=threshold)
    recall = calculate_recall(y_true, y_pred_logits, threshold=threshold)
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)


def calculate_metrics(y_true: torch.Tensor, y_pred_logits: torch.Tensor, threshold: float = 0.0):
    accuracy = calculate_accuracy(y_true, y_pred_logits, threshold=threshold)
    precision = calculate_precision(y_true, y_pred_logits, threshold=threshold)
    recall = calculate_recall(y_true, y_pred_logits, threshold=threshold)
    f1 = calculate_f1(y_true, y_pred_logits, threshold=threshold)
    fbeta = calculate_fbeta(y_true, y_pred_logits, beta=0.5, threshold=threshold)
    return accuracy, precision, recall, f1, fbeta


def mean_bce_with_logits(y_true: torch.Tensor, y_pred_logits: torch.Tensor) -> float:
    y_true = _as_float_tensor(y_true)
    y_pred_logits = _as_float_tensor(y_pred_logits)
    if y_true.numel() == 0:
        return 0.0
    return float(F.binary_cross_entropy_with_logits(y_pred_logits, y_true).item())
def average_precision_at_k(ranked_ids, relevant_set, k=None):
    if k is not None:
        ranked_ids = ranked_ids[:k]

    if len(relevant_set) == 0:
        return None

    num_hits = 0
    precision_sum = 0.0

    for i, candidate_id in enumerate(ranked_ids, start=1):
        if candidate_id in relevant_set:
            num_hits += 1
            precision_sum += num_hits / i

    denom = min(len(relevant_set), len(ranked_ids)) if k is not None else len(relevant_set)
    if denom == 0:
        return None

    return precision_sum / denom