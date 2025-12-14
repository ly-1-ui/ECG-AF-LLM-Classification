"""
Utility script to oversample the positive samples in the LLM instruction dataset.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, MutableMapping, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Duplicate positive samples in a JSONL dataset to mitigate class imbalance. "
            "Each positive sample is written `factor` times total."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input JSONL file (e.g., data/llm_cv0/mm_instructions_train_cv0.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Where to write the balanced JSONL file. "
            "If omitted, a *_posx{factor}.jsonl file is created alongside the input."
        ),
    )
    parser.add_argument(
        "--positive-answer",
        type=str,
        default="有房颤。",
        help="Answer string that should be considered a positive sample.",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=5,
        help="Number of total copies per positive sample (must be >= 1).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Optionally shuffle the oversampled dataset before saving.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when --shuffle is supplied.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterator[MutableMapping[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def oversample_positive(
    records: Iterable[MutableMapping[str, object]],
    positive_answer: str,
    factor: int,
) -> List[MutableMapping[str, object]]:
    augmented: List[MutableMapping[str, object]] = []
    for record in records:
        augmented.append(record)
        if record.get("answer") == positive_answer:
            for _ in range(factor - 1):
                augmented.append(deepcopy(record))
    return augmented


def main() -> None:
    args = parse_args()

    if args.factor < 1:
        raise ValueError("--factor must be >= 1.")

    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} does not exist.")

    output_path = (
        args.output
        if args.output is not None
        else input_path.with_name(f"{input_path.stem}_posx{args.factor}{input_path.suffix}")
    )

    base_records = list(read_jsonl(input_path))
    counts = Counter("positive" if r.get("answer") == args.positive_answer else "negative" for r in base_records)

    oversampled = (
        base_records
        if args.factor == 1
        else oversample_positive(base_records, args.positive_answer, args.factor)
    )

    if args.shuffle:
        import random

        random.seed(args.seed)
        random.shuffle(oversampled)

    write_jsonl(output_path, oversampled)

    aug_counts = Counter(
        "positive" if r.get("answer") == args.positive_answer else "negative" for r in oversampled
    )

    print(
        f"Wrote {len(oversampled)} samples to {output_path}. "
        f"Positives: {aug_counts['positive']} (was {counts['positive']}), "
        f"Negatives: {aug_counts['negative']} (was {counts['negative']})."
    )


if __name__ == "__main__":
    main()
