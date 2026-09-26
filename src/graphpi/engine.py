from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Mapping, Sequence

from .clients import ChatClient, EmbeddingClient, UsageTracker
from .config import GraphPIConfig
from .construction import GraphBuilder
from .evidence import locate_targets, retrieve_candidates, select_evidence
from .graph import StateGraph
from .integration import PeerIntegrator
from .rendering import ACTOR_SYSTEM, DISCUSSION_INSTRUCTION, INITIAL_SYSTEM, render_graph


@dataclass
class AgentState:
    graph: StateGraph = field(default_factory=StateGraph)
    message_graph: StateGraph = field(default_factory=StateGraph)
    message: str = ""
    message_index: int = 0
    evidence_spent: int = 0
    acquisition_closed: bool = False
    pending_nodes: set[str] = field(default_factory=set)


class GraphPI:

    def __init__(self, config: GraphPIConfig | None = None, *, actor=None,
                 builder=None, integrator=None):
        self.config = config or GraphPIConfig()
        self.usage = UsageTracker()
        self.actor = actor if actor is not None else ChatClient(
            self.config.actor, component="actor", tracker=self.usage)
        if builder is None:
            embedding = EmbeddingClient(
                self.config.embedding, tracker=self.usage,
                batch_size=self.config.algorithm.embedding_batch_size)
            builder = GraphBuilder(
                ChatClient(self.config.builder, component="builder", tracker=self.usage),
                embedding, max_nodes=self.config.algorithm.max_nodes_per_source,
                max_batch_chars=self.config.algorithm.builder_batch_chars,
                max_batch_units=self.config.algorithm.builder_batch_units)
        self.builder = builder
        self.integrator = integrator if integrator is not None else PeerIntegrator(
            ChatClient(self.config.linker, component="linker", tracker=self.usage),
            neighbors=self.config.algorithm.neighbors, pair_limit=self.config.algorithm.pair_limit)
        self.task = ""
        self.states: dict[str, AgentState] = {}
        self.trace: list[dict] = []

    @classmethod
    def from_config(cls, config: GraphPIConfig) -> GraphPI:
        return cls(config)

    @property
    def graphs(self) -> dict[str, StateGraph]:
        return {owner: state.graph for owner, state in self.states.items()}

    def snapshot(self) -> dict[str, StateGraph]:
        return {owner: state.graph.copy() for owner, state in self.states.items()}

    def initialize_agent(self, task: str, agent_id: str, observations: Sequence[str],
                         *, initial_message: str | None = None) -> str:
        if (not isinstance(task, str) or not task.strip()
                or not isinstance(agent_id, str) or not agent_id.strip()):
            raise ValueError("Task and agent ID must be nonempty")
        if agent_id in self.states:
            raise ValueError(f"Agent already initialized: {agent_id}")
        if self.states and task != self.task:
            raise ValueError("A GraphPI instance represents one task episode")
        if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)) or any(
                not isinstance(text, str) for text in observations):
            raise ValueError("Observations must be a sequence of strings")
        if initial_message is not None and not isinstance(initial_message, str):
            raise ValueError("initial_message must be a string or None")
        self.task = task
        state = AgentState()
        if initial_message is None:
            initial_message = self.actor.complete([
                {"role": "system", "content": INITIAL_SYSTEM},
                {"role": "user", "content":
                 "Task:\n" + task + "\n\nYour assigned observations:\n"
                 + "\n\n".join(observations)
                 + "\n\n" + DISCUSSION_INSTRUCTION}])
        sources = [{"source_id": "task", "owner": "task", "text": task,
                    "origin": "task_observation"}]
        sources.extend(
            {"source_id": f"observation:{agent_id}:{index}", "owner": agent_id,
             "text": text, "origin": "task_observation"}
            for index, text in enumerate(observations))
        message_id = f"message:{agent_id}:0"
        if initial_message:
            sources.append({"source_id": message_id, "owner": agent_id,
                            "text": initial_message, "origin": "message"})
        if hasattr(self.builder, "build_many"):
            graphs = self.builder.build_many(task=task, sources=sources)
        else:
            graphs = {source["source_id"]: self.builder.build(task=task, **source)
                      for source in sources}
        for source in sources:
            if source["origin"] == "task_observation":
                state.graph.merge(graphs[source["source_id"]])
        state.pending_nodes.update(state.graph.nodes)
        self.states[agent_id] = state
        try:
            if initial_message:
                self._commit_message(agent_id, initial_message, graphs[message_id])
        except Exception:
            del self.states[agent_id]
            raise
        return initial_message

    def add_observation(self, agent_id: str, text: str, *, source_id: str,
                        origin: str = "task_observation") -> StateGraph:
        state = self.states[agent_id]
        graph = self.builder.build(task=self.task, source_id=source_id, owner=agent_id,
                                   text=text, origin=origin)
        event = self.integrator.integrate(state.graph, graph)
        state.pending_nodes.update(event["new_nodes"])
        self.trace.append({"stage": "observation", "receiver": agent_id, **event})
        return graph

    def compile_message(self, owner: str, text: str, *, source_id: str) -> StateGraph:
        return self.builder.build(task=self.task, source_id=source_id, owner=owner,
                                  text=text, origin="message")

    def record_message(self, agent_id: str, text: str) -> StateGraph:
        return self.record_messages({agent_id: text})[agent_id]

    def record_messages(self, messages: Mapping[str, str]) -> dict[str, StateGraph]:
        if not isinstance(messages, Mapping):
            raise ValueError("Actor messages must be a mapping")
        sources = []
        for agent_id, text in messages.items():
            if agent_id not in self.states:
                raise ValueError(f"Unknown agent: {agent_id}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Actor messages must be nonempty strings")
            sources.append({
                "source_id": f"message:{agent_id}:{self.states[agent_id].message_index}",
                "owner": agent_id, "text": text, "origin": "message"})
        if not sources:
            return {}
        if hasattr(self.builder, "build_many"):
            graphs = self.builder.build_many(task=self.task, sources=sources)
        else:
            graphs = {source["source_id"]: self.builder.build(task=self.task, **source)
                      for source in sources}
        prepared = []
        for source in sources:
            owner = source["owner"]
            graph = graphs[source["source_id"]]
            state, event = self._prepare_message(owner, source["text"], graph)
            prepared.append((owner, state, event, graph))
        for owner, state, event, _ in prepared:
            self.states[owner] = state
            self.trace.append(event)
        return {owner: graph.copy() for owner, _, _, graph in prepared}

    def _prepare_message(self, agent_id: str, text: str,
                         graph: StateGraph) -> tuple[AgentState, dict]:
        state = self.states[agent_id]
        updated = graph.copy()
        event = self.integrator.integrate(updated, state.graph)
        combined = state.graph.copy()
        combined.merge(updated)
        combined.latest_self = set(graph.nodes)
        event["new_nodes"] = sorted(graph.nodes)
        event["support_direction"] = "existing_to_current"
        prepared = replace(state, graph=combined, message_graph=graph.copy(),
                           message=text, message_index=state.message_index + 1,
                           pending_nodes=set())
        return prepared, {"stage": "self_message", "receiver": agent_id,
                          "message_index": state.message_index, **event}

    def _commit_message(self, agent_id: str, text: str, graph: StateGraph) -> StateGraph:
        state, event = self._prepare_message(agent_id, text, graph)
        self.states[agent_id] = state
        self.trace.append(event)
        return graph.copy()

    def _integrate_stage(self, receiver: StateGraph,
                         incoming: Mapping[str, StateGraph]) -> list[dict]:
        peers = sorted(incoming)
        graphs = [incoming[peer] for peer in peers]
        if hasattr(self.integrator, "integrate_many"):
            events = self.integrator.integrate_many(receiver, graphs)
        else:
            events = [self.integrator.integrate(receiver, graph) for graph in graphs]
        return [{"peer": peer, **event} for peer, event in zip(peers, events)]

    def close_acquisition(self, agent_id: str) -> None:
        self.states[agent_id].acquisition_closed = True

    def receive(self, agent_id: str, peer_messages: Mapping[str, StateGraph],
                peer_graphs: Mapping[str, StateGraph] | None = None, *,
                acquire: bool = True) -> dict:
        state = self.states[agent_id]
        updated = state.graph.copy()
        integration = self._integrate_stage(
            updated, {peer: graph for peer, graph in peer_messages.items()
                      if peer != agent_id})
        algorithm = self.config.algorithm
        budget = algorithm.evidence_budget
        if algorithm.budget_scope == "episode":
            budget = max(0, budget - state.evidence_spent)
        if not acquire or state.acquisition_closed:
            budget = 0
        peers = {peer_id: graph for peer_id, graph in (peer_graphs or {}).items()
                 if peer_id != agent_id}
        targets = locate_targets(updated) if budget and peers else []
        candidates = retrieve_candidates(
            updated, peers, targets, top_k=algorithm.retrieval_top_k) if targets else []
        selected = select_evidence(updated, candidates, budget=budget)
        merged: dict[str, StateGraph] = {}
        for candidate in selected.candidates:
            merged.setdefault(candidate.peer_id, StateGraph()).merge(candidate.graph)
        evidence_integration = self._integrate_stage(updated, merged)
        new_nodes = updated.nodes.keys() - state.graph.nodes.keys()
        state.graph = updated
        state.pending_nodes.update(new_nodes)
        state.evidence_spent += selected.cost
        event = {
            "stage": "receive", "receiver": agent_id,
            "message_integration": integration, "targets": [asdict(q) for q in targets],
            "candidate_count": len(candidates), "evidence_cost": selected.cost,
            "evidence_utility": selected.utility,
            "budget_before": budget, "budget_scope": algorithm.budget_scope,
            "total_evidence_spent": state.evidence_spent,
            "selected_evidence": [
                {"peer": candidate.peer_id, "nodes": sorted(candidate.graph.nodes),
                 "source_spans": sorted(candidate.graph.span_ids()),
                 "targets": sorted(candidate.target_ids)} for candidate in selected.candidates],
            "evidence_integration": evidence_integration,
        }
        self.trace.append(event)
        return {"view": self.view(agent_id), **event}

    def view(self, agent_id: str, *, full: bool = False,
             task_in_prompt: bool = False) -> str:
        state = self.states[agent_id]
        return render_graph(state.graph, focus_nodes=state.pending_nodes, full=full,
                            task_text=self.task if task_in_prompt and not full else None)

    def operation_counts(self) -> dict:
        linker = dict(candidate_pairs=0, cache_hits=0, classified_pairs=0, linker_calls=0)
        for event in self.trace:
            rows = ([*event["message_integration"], *event["evidence_integration"]]
                    if event["stage"] == "receive" else [event])
            for row in rows:
                for key in linker:
                    linker[key] += row.get(key, 0)
        embedding = getattr(self.builder, "embedding", None)
        return {
            "builder": dict(getattr(self.builder, "stats", {})),
            "linker": linker,
            "embedding": dict(getattr(embedding, "stats", {})),
        }

    def reason(self, agent_id: str, *, instruction: str = "") -> str:
        if not isinstance(instruction, str):
            raise ValueError("Reasoning instructions must be strings")
        return self.actor.complete([
            {"role": "system", "content": ACTOR_SYSTEM},
            {"role": "user", "content":
             "Task:\n" + self.task + "\n\nYour current reasoning state:\n"
             + self.view(agent_id, task_in_prompt=True) + "\n\n"
             + (instruction or DISCUSSION_INSTRUCTION)}])

    def finalize(self, agent_id: str, *, instruction: str = "") -> str:
        if not isinstance(instruction, str):
            raise ValueError("Final instructions must be strings")
        self.close_acquisition(agent_id)
        return self.reason(agent_id, instruction=instruction or
                           "Give your final answer to the task, supported by the available evidence.")

    def run(self, task: str, observations: Mapping[str, Sequence[str]], *,
            rounds: int | None = None, final_instruction: str = "",
            incoming_transform: Callable[[int, str, str, str], str] | None = None) -> dict:
        if self.states:
            raise ValueError("Create a fresh GraphPI instance for each task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Task must be a nonempty string")
        if not isinstance(final_instruction, str):
            raise ValueError("Final instructions must be strings")
        if not isinstance(observations, Mapping) or not observations:
            raise ValueError("Observations must be a nonempty mapping")
        for owner, texts in observations.items():
            if (not isinstance(owner, str) or not owner.strip()
                    or not isinstance(texts, Sequence) or isinstance(texts, (str, bytes))
                    or any(not isinstance(text, str) for text in texts)):
                raise ValueError("Each agent needs a string ID and a sequence of observation strings")
        if incoming_transform is not None and not callable(incoming_transform):
            raise ValueError("incoming_transform must be callable")
        rounds = self.config.algorithm.rounds if rounds is None else rounds
        if type(rounds) is not int or rounds < 0 or not observations:
            raise ValueError("A run needs agents and a nonnegative round count")
        for owner, texts in observations.items():
            self.initialize_agent(task, owner, texts)
        history = [{"round": 0, "messages": {
            owner: state.message for owner, state in self.states.items()}}]
        for round_index in range(1, rounds + 1):
            snapshots = self.snapshot()
            messages = {owner: state.message_graph.copy() for owner, state in self.states.items()}
            texts = {owner: state.message for owner, state in self.states.items()}
            answers = {}
            for owner in self.states:
                incoming = self._incoming(owner, round_index, texts, messages, incoming_transform)
                self.receive(owner, incoming, snapshots)
                answers[owner] = self.reason(owner)
            self.record_messages(answers)
            history.append({"round": round_index, "messages": {
                owner: state.message for owner, state in self.states.items()}})
        final_answers = {}
        if rounds:
            snapshots = self.snapshot()
            messages = {owner: state.message_graph.copy() for owner, state in self.states.items()}
            texts = {owner: state.message for owner, state in self.states.items()}
            for owner in self.states:
                incoming = self._incoming(owner, rounds + 1, texts, messages, incoming_transform)
                self.receive(owner, incoming, snapshots, acquire=False)
        for owner in self.states:
            final_answers[owner] = self.finalize(owner, instruction=final_instruction)
        return {
            "protocol": "graphpi-synchronous", "rounds": rounds,
            "final_answers": final_answers, "messages": history,
            "graphs": {owner: state.graph.to_dict() for owner, state in self.states.items()},
            "trace": list(self.trace), "usage": self.usage.to_dict(),
            "operations": self.operation_counts(),
        }

    def _incoming(self, receiver, round_index, texts, graphs, transform):
        incoming = {}
        for peer, text in texts.items():
            if peer == receiver:
                continue
            delivered = transform(round_index, receiver, peer, text) if transform else text
            if delivered == text:
                incoming[peer] = graphs[peer]
            else:
                incoming[peer] = self.compile_message(
                    peer, delivered, source_id=f"delivered:{round_index}:{receiver}:{peer}")
        return incoming
