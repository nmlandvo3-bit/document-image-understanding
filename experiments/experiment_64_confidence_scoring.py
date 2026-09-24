from __future__ import annotations

import argparse
from dataclasses import dataclass

from experiments.experiment_62_answer_type_inference import infer_answer_type
from experiments.experiment_63_candidate_type_validation import validate_candidate


RELATION_BASE = {
    "same_row_right": 0.96,
    "same_row_right_continuation": 0.90,
    "table_column_below": 0.92,
    "below": 0.84,
    "unknown": 0.70,
}


@dataclass(frozen=True)
class ConfidenceResult:
    score: float
    level: str
    expected_type: str
    type_valid: bool
    relation: str
    ocr_confidence: float | None
    ambiguous: bool
    candidate_count: int
    reason: str


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def confidence_level(score: float) -> str:
    if score >= 0.90:
        return "high"
    if score >= 0.70:
        return "medium"
    return "low"


def score_candidate(
    *,
    question: str,
    candidate: str,
    relation: str = "unknown",
    ocr_confidence: float | None = None,
    ambiguous: bool = False,
    candidate_count: int = 1,
) -> ConfidenceResult:
    """
    Deterministic confidence score for one retrieved OCR candidate.

    Signals:
    - expected answer type inferred from the question,
    - whether the candidate matches that type,
    - spatial relation strength,
    - OCR confidence when available,
    - ambiguity / competing candidate count.

    This is not a calibrated probability. It is a transparent heuristic score
    for ranking and UI confidence display.
    """

    expected_type = infer_answer_type(question)
    validation = validate_candidate(expected_type, candidate)

    relation_score = RELATION_BASE.get(relation, RELATION_BASE["unknown"])

    score = relation_score
    reasons: list[str] = [f"relation={relation}:{relation_score:.2f}"]

    if validation.valid:
        score += 0.03
        reasons.append("type_match=+0.03")
    else:
        score -= 0.35
        reasons.append("type_mismatch=-0.35")

    if ocr_confidence is not None:
        ocr = clamp(float(ocr_confidence))

        # Pull the score slightly toward the OCR confidence rather than letting
        # OCR dominate the layout/type signals.
        ocr_adjustment = (ocr - 0.85) * 0.20
        score += ocr_adjustment
        reasons.append(f"ocr_adjustment={ocr_adjustment:+.3f}")

    if candidate_count > 1:
        penalty = min(0.20, 0.07 * (candidate_count - 1))
        score -= penalty
        reasons.append(f"candidate_count_penalty=-{penalty:.2f}")

    if ambiguous:
        # Ambiguity is an explicit safety override. The score remains useful
        # for inspection, but it must not be interpreted as answer confidence.
        score = min(score, 0.49)
        reasons.append("ambiguity_cap=0.49")

    score = round(clamp(score), 3)

    return ConfidenceResult(
        score=score,
        level=confidence_level(score),
        expected_type=expected_type,
        type_valid=validation.valid,
        relation=relation,
        ocr_confidence=ocr_confidence,
        ambiguous=ambiguous,
        candidate_count=candidate_count,
        reason="; ".join(reasons),
    )


def run_validation() -> int:
    cases = [
        {
            "name": "Strong payment date",
            "question": "What is the payment date?",
            "candidate": "115/09/21",
            "relation": "table_column_below",
            "ocr_confidence": 0.98,
            "ambiguous": False,
            "candidate_count": 1,
            "expect_level": "high",
        },
        {
            "name": "Strong customer name",
            "question": "What is the customer name?",
            "candidate": "Alex Carter",
            "relation": "same_row_right",
            "ocr_confidence": 0.99,
            "ambiguous": False,
            "candidate_count": 1,
            "expect_level": "high",
        },
        {
            "name": "Strong receipt total",
            "question": "What is the total?",
            "candidate": "310",
            "relation": "same_row_right",
            "ocr_confidence": 0.97,
            "ambiguous": False,
            "candidate_count": 1,
            "expect_level": "high",
        },
        {
            "name": "Weak spatial date",
            "question": "What is the payment date?",
            "candidate": "115/09/21",
            "relation": "unknown",
            "ocr_confidence": 0.80,
            "ambiguous": False,
            "candidate_count": 1,
            "expect_level": "medium",
        },
        {
            "name": "Type mismatch",
            "question": "What is the customer name?",
            "candidate": "03/14/1992",
            "relation": "same_row_right",
            "ocr_confidence": 0.99,
            "ambiguous": False,
            "candidate_count": 1,
            "expect_level": "low",
        },
        {
            "name": "Ambiguous total",
            "question": "What is the total?",
            "candidate": "310",
            "relation": "same_row_right",
            "ocr_confidence": 0.99,
            "ambiguous": True,
            "candidate_count": 2,
            "expect_level": "low",
        },
        {
            "name": "Ambiguous identifier",
            "question": "What is the reference number?",
            "candidate": "0055",
            "relation": "same_row_right",
            "ocr_confidence": 0.98,
            "ambiguous": True,
            "candidate_count": 2,
            "expect_level": "low",
        },
        {
            "name": "Continuation time",
            "question": "What time was the payment?",
            "candidate": "15:01:23",
            "relation": "same_row_right_continuation",
            "ocr_confidence": 0.95,
            "ambiguous": False,
            "candidate_count": 1,
            "expect_level": "high",
        },
    ]

    print("=" * 100)
    print("EXPERIMENT 64: DETERMINISTIC ANSWER CONFIDENCE SCORING")
    print("=" * 100)

    passed = 0

    for i, case in enumerate(cases, start=1):
        result = score_candidate(
            question=case["question"],
            candidate=case["candidate"],
            relation=case["relation"],
            ocr_confidence=case["ocr_confidence"],
            ambiguous=case["ambiguous"],
            candidate_count=case["candidate_count"],
        )

        ok = result.level == case["expect_level"]
        status = "PASS" if ok else "FAIL"

        print(
            f"[{status}] {i:02d}. {case['name']}\n"
            f"  question={case['question']!r}\n"
            f"  candidate={case['candidate']!r}\n"
            f"  expected_type={result.expected_type!r}\n"
            f"  type_valid={result.type_valid}\n"
            f"  relation={result.relation}\n"
            f"  ocr_confidence={result.ocr_confidence}\n"
            f"  ambiguous={result.ambiguous}\n"
            f"  candidate_count={result.candidate_count}\n"
            f"  score={result.score}\n"
            f"  level={result.level!r} (expected {case['expect_level']!r})\n"
            f"  reason={result.reason}"
        )

        if ok:
            passed += 1

    print("=" * 100)
    print(f"RESULT: {passed}/{len(cases)} tests passed")
    print("=" * 100)

    return 0 if passed == len(cases) else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 64: deterministic answer-confidence scoring."
    )

    parser.add_argument("question", nargs="?")
    parser.add_argument("candidate", nargs="?")
    parser.add_argument(
        "--relation",
        default="unknown",
        choices=sorted(RELATION_BASE.keys()),
    )
    parser.add_argument(
        "--ocr-confidence",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--ambiguous",
        action="store_true",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    if args.question is None and args.candidate is None:
        return run_validation()

    if not args.question or args.candidate is None:
        parser.error("Provide both question and candidate, or neither.")

    result = score_candidate(
        question=args.question,
        candidate=args.candidate,
        relation=args.relation,
        ocr_confidence=args.ocr_confidence,
        ambiguous=args.ambiguous,
        candidate_count=max(1, args.candidate_count),
    )

    print(f"QUESTION: {args.question}")
    print(f"CANDIDATE: {args.candidate}")
    print(f"EXPECTED ANSWER TYPE: {result.expected_type}")
    print(f"TYPE VALID: {result.type_valid}")
    print(f"RELATION: {result.relation}")
    print(f"OCR CONFIDENCE: {result.ocr_confidence}")
    print(f"AMBIGUOUS: {result.ambiguous}")
    print(f"CANDIDATE COUNT: {result.candidate_count}")
    print(f"CONFIDENCE SCORE: {result.score}")
    print(f"CONFIDENCE LEVEL: {result.level}")
    print(f"REASON: {result.reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
