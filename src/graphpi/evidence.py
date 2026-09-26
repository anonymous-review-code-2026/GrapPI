from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Iterable, Mapping, Sequence

from .graph import StateGraph


@dataclass(frozen=True)
class EvidenceTarget:

    node_id: str
    kind: str
    path: tuple[str, ...]
    related_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"conflict", "open_question", "coverage"}:
            raise ValueError(f"Unsupported evidence target kind: {self.kind!r}")
        if not self.path or self.path[0] != self.node_id:
            raise ValueError("A target path must start at its target node")


@dataclass(frozen=True)
class EvidenceCandidate:

    peer_id: str
    graph: StateGraph
    target_ids: frozenset[str]


@dataclass
class SelectionResult:
    candidates: list[EvidenceCandidate]
    cost: int
    utility: float


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("Cosine similarity requires nonempty equal-length embeddings")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not isfinite(value) for value in (*a, *b)):
        raise ValueError("Cosine similarity requires finite numeric components")
    scale_a, scale_b = max(map(abs, a)), max(map(abs, b))
    if scale_a == 0 or scale_b == 0:
        raise ValueError("Cosine similarity is undefined for zero embeddings")
    normalized_a = [value / scale_a for value in a]
    normalized_b = [value / scale_b for value in b]
    score = sum(x * y for x, y in zip(normalized_a, normalized_b)) / (
        sqrt(sum(value * value for value in normalized_a))
        * sqrt(sum(value * value for value in normalized_b))
    )
    return min(1.0, max(-1.0, score))


def locate_targets(receiver: StateGraph, *, allow_coverage: bool = True) -> list[EvidenceTarget]:
    receiver.validate()
    predecessors: dict[str, set[str]] = defaultdict(set)
    for source, target in receiver.premise_edges | receiver.support_edges:
        predecessors[target].add(source)
    paths = {identifier: (identifier,) for identifier in sorted(receiver.latest_self)}
    queue = deque(paths)
    while queue:
        current = queue.popleft()
        for predecessor in sorted(predecessors[current]):
            if predecessor not in paths:
                paths[predecessor] = (predecessor,) + paths[current]
                queue.append(predecessor)

    conflicts: dict[str, set[str]] = defaultdict(set)
    for left, right in sorted(receiver.contradict_edges):
        if left in paths:
            conflicts[left].add(right)
        if right in paths:
            conflicts[right].add(left)
    targets = [
        EvidenceTarget(identifier, "conflict", paths[identifier], tuple(sorted(witnesses)))
        for identifier, witnesses in conflicts.items()
    ]
    for identifier in paths:
        if receiver.nodes[identifier].kind == "open_question":
            related = {
                right if left == identifier else left
                for left, right in receiver.relate_edges
                if identifier in (left, right)
            }
            targets.append(EvidenceTarget(
                identifier, "open_question", paths[identifier], tuple(sorted(related))
            ))
    if not targets and allow_coverage:
        has_premises = {target for _, target in receiver.premise_edges}
        targets = [
            EvidenceTarget(identifier, "coverage", path)
            for identifier, path in paths.items()
            if receiver.nodes[identifier].kind == "claim"
            and (identifier in receiver.latest_self or identifier not in has_premises)
        ]
    order = {"conflict": 0, "open_question": 1, "coverage": 2}
    return sorted(targets, key=lambda item: (order[item.kind], len(item.path), item.node_id))


def _expand_seed(
    graph: StateGraph,
    seed: str,
    span_nodes: Mapping[str, set[str]],
    premise_predecessors: Mapping[str, set[str]],
) -> frozenset[str]:
    included = {seed}
    queue = deque([seed])
    while queue:
        current = queue.popleft()
        required = span_nodes[graph.nodes[current].span.id] | premise_predecessors.get(current, set())
        for identifier in sorted(required - included):
            included.add(identifier)
            queue.append(identifier)
    return frozenset(included)


def retrieve_candidates(
    receiver: StateGraph,
    peers: Mapping[str, StateGraph],
    targets: Iterable[EvidenceTarget],
    *,
    top_k: int = 24,
) -> list[EvidenceCandidate]:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
        raise ValueError("top_k must be a nonnegative integer")
    receiver.validate()
    target_list = list(targets)
    if top_k == 0 or not target_list or not peers:
        return []
    for target in target_list:
        if target.node_id not in receiver.nodes:
            raise ValueError(f"Unknown evidence target: {target.node_id!r}")
    span_indices: dict[str, dict[str, set[str]]] = {}
    premise_indices: dict[str, dict[str, set[str]]] = {}
    for peer_id, graph in peers.items():
        graph.validate()
        spans: dict[str, set[str]] = defaultdict(set)
        predecessors: dict[str, set[str]] = defaultdict(set)
        for node in graph.nodes.values():
            spans[node.span.id].add(node.id)
        for source, target in graph.premise_edges:
            predecessors[target].add(source)
        span_indices[peer_id] = spans
        premise_indices[peer_id] = predecessors

    closures: dict[tuple[str, str], frozenset[str]] = {}

    def closure(peer_id: str, node_id: str) -> frozenset[str]:
        key = (peer_id, node_id)
        if key not in closures:
            closures[key] = _expand_seed(
                peers[peer_id], node_id, span_indices[peer_id], premise_indices[peer_id]
            )
        return closures[key]

    pairs: list[tuple[float, str, str, str]] = []
    for target in target_list:
        query = receiver.nodes[target.node_id]
        for peer_id in sorted(peers):
            graph = peers[peer_id]
            for node_id in sorted(graph.nodes):
                node = graph.nodes[node_id]
                if node.kind != "claim":
                    continue
                if target.kind == "coverage":
                    if node.span.origin != "task_observation":
                        continue
                    if any(graph.nodes[item].span.origin != "task_observation"
                           for item in closure(peer_id, node_id)):
                        continue
                pairs.append((-cosine(query.embedding, node.embedding),
                              target.node_id, peer_id, node_id))
    pairs.sort()
    seed_targets: dict[tuple[str, str], set[str]] = defaultdict(set)
    for _, target_id, peer_id, node_id in pairs[:top_k]:
        seed_targets[(peer_id, node_id)].add(target_id)

    grouped: dict[tuple[str, tuple[str, ...]], set[str]] = defaultdict(set)
    for (peer_id, node_id), target_ids in seed_targets.items():
        key = (peer_id, tuple(sorted(closure(peer_id, node_id))))
        grouped[key].update(target_ids)
    return [
        EvidenceCandidate(peer_id, peers[peer_id].subgraph(identifiers), frozenset(target_ids))
        for (peer_id, identifiers), target_ids in sorted(grouped.items())
    ]


def select_evidence(
    receiver: StateGraph,
    candidates: Iterable[EvidenceCandidate],
    *,
    budget: int,
) -> SelectionResult:
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
        raise ValueError("The evidence budget must be a nonnegative integer")
    receiver.validate()
    known_spans = receiver.span_ids()
    remaining = list(candidates)
    for candidate in remaining:
        candidate.graph.validate()
    combined = receiver.copy()
    for candidate in remaining:
        combined.merge(candidate.graph)
    remaining.sort(key=lambda candidate: (
        candidate.peer_id, tuple(sorted(candidate.graph.nodes)), tuple(sorted(candidate.target_ids))
    ))
    selected: list[EvidenceCandidate] = []
    introduced: set[str] = set()
    coverage: dict[str, set[str]] = defaultdict(set)
    while remaining:
        best_index: int | None = None
        best_ratio = -1.0
        best_cost = 0
        for index, candidate in enumerate(remaining):
            spans = candidate.graph.span_ids() - known_spans
            cost = len(spans - introduced)
            if cost == 0 or cost > budget - len(introduced):
                continue
            gain = sum(
                sqrt(len(coverage[target] | spans)) - sqrt(len(coverage[target]))
                for target in sorted(candidate.target_ids)
            )
            if gain > 0 and gain / cost > best_ratio:
                best_index, best_ratio, best_cost = index, gain / cost, cost
        if best_index is None:
            break
        chosen = remaining.pop(best_index)
        spans = chosen.graph.span_ids() - known_spans
        for target in chosen.target_ids:
            coverage[target].update(spans)
        introduced.update(spans)
        selected.append(chosen)
        assert len(introduced) <= budget and best_cost > 0
    utility = sum(sqrt(len(coverage[target])) for target in sorted(coverage))
    return SelectionResult(selected, len(introduced), utility)
