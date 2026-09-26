from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class SourceSpan:

    id: str
    source_id: str
    owner: str
    start: int
    end: int
    text: str
    origin: str = "message"

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in
                   (self.id, self.source_id, self.owner, self.origin)):
            raise ValueError("Source span identifiers, owner, and origin must be nonempty strings")
        if (not isinstance(self.start, int) or isinstance(self.start, bool)
                or not isinstance(self.end, int) or isinstance(self.end, bool)
                or self.start < 0 or self.end <= self.start):
            raise ValueError("Source span offsets must describe a nonempty half-open interval")
        if not isinstance(self.text, str) or len(self.text) != self.end - self.start:
            raise ValueError("Source span text must match its character interval")


@dataclass(frozen=True)
class Node:

    id: str
    kind: str
    span: SourceSpan
    summary: str
    embedding: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("A node must have a nonempty identifier")
        if self.kind not in {"claim", "open_question"}:
            raise ValueError(f"Unsupported node kind: {self.kind!r}")
        if not isinstance(self.span, SourceSpan):
            raise ValueError("A node must reference a SourceSpan")
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("A node summary must be nonempty")
        if not isinstance(self.embedding, tuple):
            raise ValueError("Embeddings must be immutable tuples")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool)
               or not isfinite(value) for value in self.embedding):
            raise ValueError("Embedding components must be finite numbers")


@dataclass
class StateGraph:

    nodes: dict[str, Node] = field(default_factory=dict)
    premise_edges: set[tuple[str, str]] = field(default_factory=set)
    support_edges: set[tuple[str, str]] = field(default_factory=set)
    contradict_edges: set[tuple[str, str]] = field(default_factory=set)
    relate_edges: set[tuple[str, str]] = field(default_factory=set)
    joint_premises: set[tuple[tuple[str, ...], str]] = field(default_factory=set)
    latest_self: set[str] = field(default_factory=set)
    standalone_premises: set[tuple[str, str]] = field(default_factory=set)

    def _require_nodes(self, identifiers: Iterable[str]) -> None:
        missing = set(identifiers) - self.nodes.keys()
        if missing:
            raise ValueError(f"Unknown graph nodes: {sorted(missing)!r}")

    def validate(self) -> None:
        spans: dict[str, SourceSpan] = {}
        addresses: dict[tuple[str, str, int, int], SourceSpan] = {}
        for identifier, node in self.nodes.items():
            if not isinstance(node, Node) or identifier != node.id:
                raise ValueError("Graph node keys must match immutable node identifiers")
            previous = spans.setdefault(node.span.id, node.span)
            if previous != node.span:
                raise ValueError(f"Conflicting source span: {node.span.id!r}")
            address = (node.span.source_id, node.span.owner, node.span.start, node.span.end)
            previous_address = addresses.setdefault(address, node.span)
            if previous_address != node.span:
                raise ValueError(f"Conflicting source span address: {address!r}")
        self._require_nodes(self.latest_self)
        for source, target in self.premise_edges:
            self._require_nodes((source, target))
            if source == target:
                raise ValueError("A node cannot be its own premise")
        for source, target in self.standalone_premises:
            self._require_nodes((source, target))
            if (source, target) not in self.premise_edges:
                raise ValueError("Standalone premises must retain their premise edges")
        for relation, edges in (
            ("support", self.support_edges),
            ("contradict", self.contradict_edges),
            ("relate", self.relate_edges),
        ):
            for source, target in edges:
                self._validate_relation(source, target, relation)
                if relation != "support" and (source, target) != tuple(sorted((source, target))):
                    raise ValueError(f"{relation} edges must use normalized endpoint order")
        for premises, conclusion in self.joint_premises:
            if len(premises) < 2 or tuple(sorted(set(premises))) != premises:
                raise ValueError("Joint premise groups require at least two sorted unique nodes")
            self._require_nodes((*premises, conclusion))
            if any((premise, conclusion) not in self.premise_edges for premise in premises):
                raise ValueError("Joint premise groups must retain all constituent premise edges")

    def copy(self) -> StateGraph:
        return StateGraph(
            nodes=self.nodes.copy(),
            premise_edges=self.premise_edges.copy(),
            support_edges=self.support_edges.copy(),
            contradict_edges=self.contradict_edges.copy(),
            relate_edges=self.relate_edges.copy(),
            joint_premises=self.joint_premises.copy(),
            latest_self=self.latest_self.copy(),
            standalone_premises=self.standalone_premises.copy(),
        )

    def standalone_premise_edges(self) -> set[tuple[str, str]]:
        joint_edges = {(premise, conclusion) for premises, conclusion in self.joint_premises
                       for premise in premises}
        return self.standalone_premises | (self.premise_edges - joint_edges)

    def merge(self, other: StateGraph) -> set[str]:
        self.validate()
        other.validate()
        for identifier in self.nodes.keys() & other.nodes.keys():
            if self.nodes[identifier] != other.nodes[identifier]:
                raise ValueError(f"Conflicting immutable node: {identifier!r}")
        new_ids = other.nodes.keys() - self.nodes.keys()
        merged = self.copy()
        merged.standalone_premises = (
            self.standalone_premise_edges() | other.standalone_premise_edges())
        merged.nodes.update(other.nodes)
        for name in ("premise_edges", "support_edges", "contradict_edges",
                     "relate_edges", "joint_premises"):
            getattr(merged, name).update(getattr(other, name))
        merged.validate()
        self.nodes = merged.nodes
        self.premise_edges = merged.premise_edges
        self.support_edges = merged.support_edges
        self.contradict_edges = merged.contradict_edges
        self.relate_edges = merged.relate_edges
        self.joint_premises = merged.joint_premises
        self.standalone_premises = merged.standalone_premises
        return set(new_ids)

    def add_premise(self, premises: Iterable[str], conclusion: str) -> None:
        group = tuple(sorted(set(premises)))
        if not group:
            raise ValueError("A premise dependency requires at least one premise")
        self._require_nodes((*group, conclusion))
        if conclusion in group:
            raise ValueError("A node cannot be its own premise")
        self.standalone_premises.update(self.standalone_premise_edges())
        self.premise_edges.update((premise, conclusion) for premise in group)
        if len(group) > 1:
            self.joint_premises.add((group, conclusion))
        else:
            self.standalone_premises.add((group[0], conclusion))

    def _validate_relation(self, source: str, target: str, relation: str) -> None:
        self._require_nodes((source, target))
        if source == target:
            raise ValueError("Relations require two distinct nodes")
        kinds = (self.nodes[source].kind, self.nodes[target].kind)
        if relation in {"support", "contradict"}:
            if kinds != ("claim", "claim"):
                raise ValueError(f"{relation} requires two claim nodes")
        elif relation == "relate":
            if set(kinds) != {"claim", "open_question"}:
                raise ValueError("relate requires exactly one claim and one open question")
        else:
            raise ValueError(f"Unsupported relation: {relation!r}")

    def add_relation(self, source: str, target: str, relation: str) -> None:
        self._validate_relation(source, target, relation)
        pair = (source, target) if relation == "support" else tuple(sorted((source, target)))
        getattr(self, f"{relation}_edges").add(pair)

    def subgraph(self, ids: Iterable[str]) -> StateGraph:
        self.validate()
        identifiers = set(ids)
        self._require_nodes(identifiers)
        standalone = self.standalone_premise_edges()
        groups = {(premises, conclusion) for premises, conclusion in self.joint_premises
                  if conclusion in identifiers and set(premises) <= identifiers}
        complete_joint_edges = {(premise, conclusion) for premises, conclusion in groups
                                for premise in premises}
        graph = StateGraph(
            nodes={identifier: self.nodes[identifier] for identifier in sorted(identifiers)},
            premise_edges={(source, target) for source, target in self.premise_edges
                           if source in identifiers and target in identifiers
                           and ((source, target) in standalone
                                or (source, target) in complete_joint_edges)},
            support_edges={(source, target) for source, target in self.support_edges
                           if source in identifiers and target in identifiers},
            contradict_edges={pair for pair in self.contradict_edges if set(pair) <= identifiers},
            relate_edges={pair for pair in self.relate_edges if set(pair) <= identifiers},
            joint_premises=groups,
            latest_self=self.latest_self & identifiers,
            standalone_premises={pair for pair in standalone if set(pair) <= identifiers},
        )
        graph.validate()
        return graph

    def span_ids(self) -> set[str]:
        return {node.span.id for node in self.nodes.values()}

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        nodes = []
        for identifier in sorted(self.nodes):
            node = asdict(self.nodes[identifier])
            node["embedding"] = list(node["embedding"])
            nodes.append(node)
        return {
            "nodes": nodes,
            "premise_edges": [list(pair) for pair in sorted(self.premise_edges)],
            "standalone_premises": [list(pair) for pair in sorted(self.standalone_premises)],
            "support_edges": [list(pair) for pair in sorted(self.support_edges)],
            "contradict_edges": [list(pair) for pair in sorted(self.contradict_edges)],
            "relate_edges": [list(pair) for pair in sorted(self.relate_edges)],
            "joint_premises": [
                {"premises": list(premises), "conclusion": conclusion}
                for premises, conclusion in sorted(self.joint_premises)
            ],
            "latest_self": sorted(self.latest_self),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StateGraph:
        graph = cls()
        for row in value.get("nodes", []):
            node = Node(
                id=row["id"], kind=row["kind"], span=SourceSpan(**row["span"]),
                summary=row["summary"], embedding=tuple(row.get("embedding", ())),
            )
            if node.id in graph.nodes:
                raise ValueError(f"Duplicate serialized node: {node.id!r}")
            graph.nodes[node.id] = node
        for name in ("premise_edges", "standalone_premises", "support_edges",
                     "contradict_edges", "relate_edges"):
            setattr(graph, name, {tuple(pair) for pair in value.get(name, [])})
        graph.joint_premises = {
            (tuple(row["premises"]), row["conclusion"])
            for row in value.get("joint_premises", [])
        }
        graph.latest_self = set(value.get("latest_self", []))
        graph.validate()
        return graph
