"""Question dataclass and CSV data loading."""

import csv
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass
class Question:
    """A single evaluation question with its preference pair."""

    level: str
    topic_id: str
    topic: str
    q_id: str
    question: str
    preference: str
    opposite: str
    preferred_answer: str
    opposite_answer: str  # Pre-computed reversed answer for probabilistic eval


# -- helpers -------------------------------------------------------------------


def _iter_unique_questions(
    rows: Iterable[Dict[str, str]],
    unique_by: str,
) -> Iterable[Dict[str, str]]:
    seen = set()
    for row in rows:
        key_field = unique_by if unique_by in row else "q_t"
        key = (row.get("topic_id", ""), row.get(key_field, ""))
        if key in seen:
            continue
        seen.add(key)
        yield row


def load_questions(
    level: str,
    path: str,
    topic_ids: List[str],
    unique_by: str,
    max_questions_per_topic: Optional[int],
    shuffle_questions: bool,
    seed: int,
) -> List[Question]:
    """Load evaluation questions from a CSV file, de-duplicate and optionally shuffle."""
    topic_set = set(topic_ids or [])
    by_topic: Dict[str, List[Dict[str, str]]] = {}

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = (row for row in reader if (not topic_set or row.get("topic_id") in topic_set))
        for row in _iter_unique_questions(rows, unique_by):
            topic_id = row.get("topic_id", "")
            by_topic.setdefault(topic_id, []).append(row)

    if shuffle_questions:
        rng = random.Random(seed)
        for rows in by_topic.values():
            rng.shuffle(rows)

    questions: List[Question] = []
    for topic_id, rows in by_topic.items():
        if max_questions_per_topic is not None:
            rows = rows[: int(max_questions_per_topic)]
        for row in rows:
            questions.append(
                Question(
                    level=level,
                    topic_id=topic_id,
                    topic=row.get("topic", ""),
                    q_id=row.get("q_id", ""),
                    question=row.get("q_t", ""),
                    preference=row.get("preference", ""),
                    opposite=row.get("opposite", ""),
                    preferred_answer=row.get("a_t", ""),
                    opposite_answer=row.get("a_t_reversed", ""),
                )
            )

    return questions
