"""Agreement-based per-instance weighting for paired auxiliary predictions."""

import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _stem(token: str) -> str:
    """Apply a small, dependency-free normalization for inflectional endings."""
    for suffix in ("ies", "ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[:-len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def tok(text: Optional[str]) -> List[str]:
    ascii_text = (text or "").encode("ascii", "ignore").decode("ascii")
    return [_stem(token) for token in re.findall(r"[a-z]+", ascii_text.lower())]


def target_match(aspect: Optional[str], target: Optional[str]) -> bool:
    """Diagnostic-only match: every target token occurs in generated aspect text."""
    target_tokens = set(tok(target))
    return bool(target_tokens) and target_tokens.issubset(set(tok(aspect)))


def token_f1(a: Optional[str], b: Optional[str]) -> float:
    a_tokens, b_tokens = tok(a), tok(b)
    if not a_tokens or not b_tokens:
        return float(a_tokens == b_tokens)
    common = sum((Counter(a_tokens) & Counter(b_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(a_tokens)
    recall = common / len(b_tokens)
    return 2 * precision * recall / (precision + recall)


def agreement(mistral: Dict, qwen: Dict) -> Tuple[float, bool]:
    """Return agreement and whether a polarity comparison was available."""
    polarity_available = (
        mistral.get("polarity") is not None and qwen.get("polarity") is not None
    )
    polarity_match = float(
        polarity_available
        and str(mistral.get("polarity", "")).strip().lower()
        == str(qwen.get("polarity", "")).strip().lower()
    )
    score = (
        0.35 * token_f1(mistral.get("aspect"), qwen.get("aspect"))
        + 0.35 * token_f1(mistral.get("opinion"), qwen.get("opinion"))
        + 0.30 * polarity_match
    )
    return score, polarity_available


def make_weights(conf: np.ndarray, agree: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Combine signals, stretch them, apply a floor, and normalize to mean one."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    conf = np.asarray(conf, dtype=np.float64)
    agree = np.asarray(agree, dtype=np.float64)
    if conf.shape != agree.shape or conf.ndim != 1 or conf.size == 0:
        raise ValueError("conf and agree must be non-empty one-dimensional arrays")
    score = alpha * conf + (1.0 - alpha) * agree
    score = (score - score.min()) / (score.max() - score.min() + 1e-8)
    weights = 0.2 + 0.8 * score
    return weights / weights.mean()


def merge_auxiliary_rows(
    mistral_rows: Iterable[Dict],
    qwen_rows: Iterable[Dict],
    alpha: float,
    rng: np.random.Generator,
) -> Tuple[List[Dict], Dict[str, float]]:
    """Join paired rows and attach confidence, agreement, and all weight modes."""
    mistral_by_id = {row["instance_id"]: row for row in mistral_rows}
    qwen_by_id = {row["instance_id"]: row for row in qwen_rows}
    ids = [instance_id for instance_id in mistral_by_id if instance_id in qwen_by_id]
    if not ids:
        raise ValueError("Mistral and Qwen auxiliary files have no shared instance_id values")

    confidences, agreements = [], []
    merged = []
    polarity_available = True
    for instance_id in ids:
        mistral = mistral_by_id[instance_id]
        qwen = qwen_by_id[instance_id]
        agreement_score, has_polarity = agreement(mistral, qwen)
        polarity_available = polarity_available and has_polarity
        model_conf = [
            float(mistral.get("aspect_confidence", 0.5)),
            float(mistral.get("opinion_confidence", 0.5)),
            float(qwen.get("aspect_confidence", 0.5)),
            float(qwen.get("opinion_confidence", 0.5)),
        ]
        confidences.append(float(np.clip(np.mean(model_conf), 0.5, 1.0)))
        agreements.append(agreement_score)
        row = dict(mistral)
        row["agreement"] = agreement_score
        row["ensemble_confidence"] = confidences[-1]
        merged.append(row)

    conf = np.asarray(confidences)
    agree = np.asarray(agreements)
    merged_weights = {
        "confidence": conf,
        "random": rng.uniform(0.2, 1.0, size=len(ids)),
        "agreement": make_weights(conf, agree, alpha=0.0),
        "combined": make_weights(conf, agree, alpha=alpha),
    }
    merged_weights["random"] /= merged_weights["random"].mean()
    for row_index, row in enumerate(merged):
        for name, values in merged_weights.items():
            row[f"weight_{name}"] = float(values[row_index])

    diagnostics = {
        "rows": float(len(merged)),
        "agreement_min": float(agree.min()),
        "agreement_max": float(agree.max()),
        "agreement_mean": float(agree.mean()),
        "confidence_min": float(conf.min()),
        "confidence_max": float(conf.max()),
        "confidence_mean": float(conf.mean()),
        "polarity_available": float(polarity_available),
    }
    if polarity_available:
        gold = np.asarray([
            str(mistral_by_id[instance_id].get("gold_polarity", "")).lower()
            for instance_id in ids
        ])
        mistral_correct = np.asarray([
            str(mistral_by_id[instance_id]["polarity"]).lower() == gold[index]
            for index, instance_id in enumerate(ids)
        ], dtype=np.float64)
        qwen_correct = np.asarray([
            str(qwen_by_id[instance_id]["polarity"]).lower() == gold[index]
            for index, instance_id in enumerate(ids)
        ], dtype=np.float64)
        agreement_values = np.asarray(agreements)
        if np.std(agreement_values) > 0:
            diagnostics["agreement_correctness_corr"] = float(
                np.corrcoef(agreement_values, (mistral_correct + qwen_correct) / 2)[0, 1]
            )
        else:
            diagnostics["agreement_correctness_corr"] = float("nan")
    target_correct = np.asarray([
        (
            target_match(mistral_by_id[instance_id].get("aspect"), mistral_by_id[instance_id].get("target")),
            target_match(qwen_by_id[instance_id].get("aspect"), qwen_by_id[instance_id].get("target")),
        )
        for instance_id in ids
    ], dtype=np.float64)
    diagnostics["mistral_target_match_rate"] = float(target_correct[:, 0].mean())
    diagnostics["qwen_target_match_rate"] = float(target_correct[:, 1].mean())
    diagnostics["high_agreement_target_match_rate"] = float(
        target_correct[agree >= np.median(agree)].mean()
    )
    diagnostics["low_agreement_target_match_rate"] = float(
        target_correct[agree < np.median(agree)].mean()
    )
    if np.std(agree) > 0 and np.std(target_correct.mean(axis=1)) > 0:
        diagnostics["agreement_target_match_corr"] = float(
            np.corrcoef(agree, target_correct.mean(axis=1))[0, 1]
        )
    return merged, diagnostics