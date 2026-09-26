from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from .common import (fingerprint, load_records, mean, require_text, require_texts,
                     require_unique_ids)

PROTOCOL = "graphpi-synchronous-hidden-profile/1"


@dataclass(frozen=True)
class HiddenBenchTask:
    id: str
    name: str
    description: str
    shared_information: tuple[str, ...]
    hidden_information: tuple[str, ...]
    possible_answers: tuple[str, ...]
    correct_answer: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "HiddenBenchTask":
        required = {"id", "name", "description", "shared_information", "hidden_information",
                    "possible_answers", "correct_answer"}
        missing = required - row.keys()
        if missing:
            raise ValueError(f"HiddenBench task is missing {sorted(missing)}")
        choices = require_texts(row["possible_answers"], "possible_answers")
        if len(set(choices)) != len(choices):
            raise ValueError("possible_answers must be unique")
        answer = require_text(row["correct_answer"], "correct_answer")
        if answer not in choices:
            raise ValueError("correct_answer must be one of possible_answers")
        return cls(
            str(row["id"]), require_text(row["name"], "name"),
            require_text(row["description"], "description"),
            require_texts(row["shared_information"], "shared_information", allow_empty=True),
            require_texts(row["hidden_information"], "hidden_information"),
            choices, answer, {key: value for key, value in row.items() if key not in required},
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            **self.metadata, "id": self.id, "name": self.name, "description": self.description,
            "shared_information": list(self.shared_information),
            "hidden_information": list(self.hidden_information),
            "possible_answers": list(self.possible_answers), "correct_answer": self.correct_answer,
        }

    def allocate(self, seed: int) -> dict[str, list[str]]:
        rng = random.Random(seed)
        private = list(self.hidden_information)
        rng.shuffle(private)
        observations = {}
        for index, fact in enumerate(private, start=1):
            facts = [*self.shared_information, fact]
            rng.shuffle(facts)
            observations[f"agent_{index}"] = facts
        return observations

    @property
    def actor_task(self) -> str:
        return self.description + "\nAllowed answers: " + json.dumps(list(self.possible_answers))

    @property
    def final_instruction(self) -> str:
        return (
            'Select one allowed answer. Return only a JSON object with keys "vote" '
            '(the exact allowed answer string) and "rationale" (a brief justification).'
        )


def load_hiddenbench(path: str | Path) -> list[HiddenBenchTask]:
    tasks = [HiddenBenchTask.from_mapping(row) for row in load_records(path)]
    require_unique_ids(tasks)
    return tasks


def parse_vote(raw: Any, choices: Sequence[str]) -> str | None:
    if isinstance(raw, Mapping):
        value = raw.get("vote", raw.get("answer"))
    elif isinstance(raw, str):
        value = raw.strip()
        if value in choices:
            return value
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(parsed, dict):
            return None
        value = parsed.get("vote", parsed.get("answer"))
    else:
        return None
    return value if isinstance(value, str) and value in choices else None


def score_task(task: HiddenBenchTask, final_answers: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(final_answers, Mapping) or not final_answers:
        raise ValueError("A HiddenBench result requires final answers for every agent")
    if len(final_answers) != len(task.hidden_information):
        raise ValueError(f"Task {task.id} expects {len(task.hidden_information)} agents")
    votes = {agent: parse_vote(raw, task.possible_answers) for agent, raw in final_answers.items()}
    n = len(votes)
    correct = sum(vote == task.correct_answer for vote in votes.values())
    counts = Counter(votes.values())
    erroneous_consensus = (len(counts) == 1 and None not in counts
                           and correct == 0)
    return {
        "task_id": task.id, "num_agents": n, "votes": votes,
        "average_accuracy": correct / n,
        "majority_accuracy": float(correct > n / 2),
        "unanimous_error": float(erroneous_consensus),
        "invalid_answers": sum(vote is None for vote in votes.values()),
    }


def _answers_from_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    if "final_answers" in row:
        return row["final_answers"]
    votes = row.get("final_votes")
    if isinstance(votes, list):
        pairs = [(str(item["agent"]), item.get("vote")) for item in votes]
        if len(dict(pairs)) != len(pairs):
            raise ValueError("Duplicate agent in final_votes")
        return dict(pairs)
    raise ValueError("Prediction requires final_answers or final_votes")


def score_hiddenbench(tasks: Sequence[HiddenBenchTask],
                      results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {task.id: task for task in tasks}
    scored = []
    seen = set()
    for result in results:
        task_id = str(result.get("task_id", result.get("id")))
        if task_id not in by_id:
            raise ValueError(f"Unknown result task: {task_id}")
        identity = (task_id, result.get("seed", 0))
        if identity in seen:
            raise ValueError(f"Duplicate task/seed result: {identity}")
        seen.add(identity)
        score = score_task(by_id[task_id], _answers_from_row(result))
        score["seed"] = result.get("seed", 0)
        scored.append(score)
    if not scored:
        raise ValueError("No HiddenBench results to score")
    by_task = []
    for task_id in by_id:
        rows = [row for row in scored if row["task_id"] == task_id]
        if rows:
            by_task.append({
                "task_id": task_id, "num_runs": len(rows),
                **{name: mean([row[name] for row in rows]) for name in
                   ("average_accuracy", "majority_accuracy", "unanimous_error")},
            })
    return {
        "metric_scale": "0-1", "num_runs": len(scored), "num_tasks": len(by_task),
        "expected_tasks": len(tasks), "complete": len(by_task) == len(tasks),
        "missing_task_ids": sorted(set(by_id) - {row["task_id"] for row in scored}),
        "average_accuracy": mean([row["average_accuracy"] for row in by_task]),
        "majority_accuracy": mean([row["majority_accuracy"] for row in by_task]),
        "unanimous_error": mean([row["unanimous_error"] for row in by_task]),
        "invalid_answers": sum(row["invalid_answers"] for row in scored),
        "by_task": by_task, "runs": scored,
    }


def run_hiddenbench(task: HiddenBenchTask, engine: Any, *, seed: int = 0,
                    rounds: int = 6) -> dict[str, Any]:
    observations = task.allocate(seed)
    result = engine.run(task.actor_task, observations, rounds=rounds,
                        final_instruction=task.final_instruction)
    if not isinstance(result, Mapping) or "final_answers" not in result:
        raise ValueError("GraphPI did not return final_answers")
    scores = score_task(task, result["final_answers"])
    return {
        **dict(result), "task_id": task.id, "seed": seed, "protocol": PROTOCOL, "rounds": rounds,
        "fact_assignments": observations, "input_fingerprint": fingerprint(task.to_mapping()),
        "metrics": scores,
    }
