from collections import Counter

import sacrebleu
from rouge_score import rouge_scorer

from src.utils import normalize_answer


def exact_match_any(prediction, answers):
    """
    Exact-match accuracy against any accepted ground-truth answer.
    """
    pred = normalize_answer(prediction)
    golds = [normalize_answer(ans) for ans in answers]

    return int(pred in golds)


def token_f1_single(prediction, ground_truth):
    """
    Token-level F1 between prediction and one ground-truth answer.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def token_f1_any(prediction, answers):
    """
    Best token F1 over all accepted answers.
    """
    if not answers:
        return 0.0

    return max(token_f1_single(prediction, ans) for ans in answers)


def rouge_l_any(prediction, answers):
    """
    Best ROUGE-L F1 over all accepted answers.
    """
    if not answers:
        return 0.0

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    scores = []
    for ans in answers:
        score = scorer.score(
            normalize_answer(ans),
            normalize_answer(prediction),
        )
        scores.append(score["rougeL"].fmeasure)

    return max(scores)


def bleu_any(prediction, answers):
    """
    Best sentence BLEU over all accepted answers.
    """
    if not answers:
        return 0.0

    pred = normalize_answer(prediction)

    scores = []
    for ans in answers:
        ref = normalize_answer(ans)

        try:
            score = sacrebleu.sentence_bleu(pred, [ref]).score / 100.0
        except Exception:
            score = 0.0

        scores.append(score)

    return max(scores)


def compute_metrics(records):
    """
    Compute aggregate metrics.

    Each record should have:
        prediction: str
        answers: list[str]
    """
    n = len(records)

    if n == 0:
        return {
            "accuracy": 0.0,
            "token_f1": 0.0,
            "rouge_l": 0.0,
            "bleu": 0.0,
            "n": 0,
        }

    total_accuracy = 0.0
    total_f1 = 0.0
    total_rouge_l = 0.0
    total_bleu = 0.0

    for record in records:
        prediction = record["prediction"]
        answers = record["answers"]

        total_accuracy += exact_match_any(prediction, answers)
        total_f1 += token_f1_any(prediction, answers)
        total_rouge_l += rouge_l_any(prediction, answers)
        total_bleu += bleu_any(prediction, answers)

    return {
        "accuracy": total_accuracy / n,
        "token_f1": total_f1 / n,
        "rouge_l": total_rouge_l / n,
        "bleu": total_bleu / n,
        "n": n,
    }