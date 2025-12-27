from typing import List, Set, Dict


def precision_recall_f1(
    retrieved_ids: List[str],
    relevant_ids: List[str]
) -> Dict[str, float]:

    retrieved: Set[str] = set(retrieved_ids)
    relevant: Set[str] = set(relevant_ids)

    tp = len(retrieved & relevant)
    fp = len(retrieved - relevant)
    fn = len(relevant - retrieved)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }

