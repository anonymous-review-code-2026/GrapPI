from __future__ import annotations

import asyncio
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import importlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import threading
import types
from typing import Any, Mapping, Sequence

from .common import fingerprint, load_records, require_score, require_text
from .misinfotask import MisinfoTask, parse_judge_score, validate_attack

_NATIVE_LOCK = threading.Lock()
PROTOCOL = "native-misinfotask-graphpi/1"


@contextmanager
def native_modules(benchmark_root: str | Path):
    root = Path(benchmark_root).resolve()
    needed = ("agent.py", "simulator.py", "generate.py", "utils.py", "tool.py", "llm.py")
    for filename in needed:
        if not (root / "sandbox" / filename).is_file():
            raise ValueError(f"Benchmark checkout is missing sandbox/{filename}")
    if any(name == "sandbox" or name.startswith("sandbox.") for name in sys.modules):
        raise RuntimeError("A sandbox package is already imported; use a fresh process")
    sys.path.insert(0, str(root))
    try:
        loaded = {name: importlib.import_module("sandbox." + name.removesuffix(".py"))
                  for name in needed}
        for module in loaded.values():
            if root not in Path(module.__file__).resolve().parents:
                raise RuntimeError("An unexpected sandbox package shadowed the benchmark")
        yield {name.removesuffix(".py"): module for name, module in loaded.items()}
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing native benchmark dependency: {exc.name}. "
            "Install the benchmark requirements in the selected environment."
        ) from exc
    finally:
        sys.path.remove(str(root))
        for name in list(sys.modules):
            if name == "sandbox" or name.startswith("sandbox."):
                del sys.modules[name]


class Patches:
    def __init__(self):
        self.items = []

    def set(self, target, name, value):
        old = getattr(target, name)
        self.items.append((target, name, old))
        setattr(target, name, value)

    def method(self, target, name, function):
        self.set(target, name, types.MethodType(function, target))

    def restore(self):
        for target, name, value in reversed(self.items):
            setattr(target, name, value)
        self.items.clear()


class ClientToken:
    def __init__(self, *args, owner=None, **kwargs):
        self.owner = owner

    async def aclose(self):
        return None


def load_rag_documents(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() in {".txt", ".md"}:
        rows = [{"text": line.strip()} for line in path.read_text().splitlines() if line.strip()]
    else:
        rows = load_records(path)
    result = []
    for row in rows:
        text = require_text(row.get("text", row.get("page_content")), "RAG document text")
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("RAG document metadata must be an object")
        result.append({"text": text, "metadata": dict(metadata)})
    if not result:
        raise ValueError("RAG corpus must be non-empty")
    return result


class NativeGraphAdapter:
    def __init__(self, engine, simulator):
        self.engine, self.simulator = engine, simulator
        self.task = require_text(simulator.data["user_input"], "user_input")
        self.patches = Patches()
        self.seals = {}
        self.events = []
        self.errors = []
        self.observed = set()
        self.final_answer = None

    def observe(self, owner: str, text: str, kind: str):
        if not text or (owner, kind, text) in self.observed:
            return
        self.observed.add((owner, kind, text))
        self.engine.add_observation(owner, text,
                                    source_id=kind + ":" + fingerprint((owner, text)),
                                    origin="message")
        self.events.append({"event": kind, "owner": owner, "text": text})

    def seal(self, owner: str, text: str):
        message = self.engine.record_message(owner, text)
        self.seals.setdefault((owner, text), (message.copy(), self.engine.graphs[owner].copy()))
        self.events.append({"event": "publication", "owner": owner, "text": text})

    def deliver(self, receiver: str, messages: Sequence[tuple[str, str]], *, acquire=True):
        from ..graph import StateGraph
        publications, snapshots = {}, {}
        for owner, text in messages:
            if owner == receiver or not text:
                continue
            sealed = self.seals.get((owner, text))
            if sealed is None:
                graph = self.engine.compile_message(
                    owner, text, source_id="native-log:" + fingerprint((owner, text)))
                publications.setdefault(owner, StateGraph()).merge(graph)
            else:
                graph, snapshot = sealed
                publications.setdefault(owner, StateGraph()).merge(graph)
                snapshots.setdefault(owner, StateGraph()).merge(snapshot)
        result = self.engine.receive(receiver, publications, snapshots, acquire=acquire)
        self.events.append({"event": "receive", "receiver": receiver,
                            "senders": sorted(publications), "acquire": acquire,
                            "evidence_cost": result.get("evidence_cost", 0)})
        return result["view"]

    def attach(self):
        adapter = self
        for worker in self.simulator.agents:
            owner = str(worker.name)
            self.engine.initialize_agent(self.task, owner, [self.task], initial_message="")
            self.observe(owner, str(worker.background.info), "native-plan")
            original_step, original_think = worker.emulate_one_step, worker._think
            status = {"waiting": False, "thinking": False}
            original_receive = getattr(worker, "receive_information", None)
            if callable(original_receive):
                def receive(agent, _original=original_receive, _status=status):
                    message = _original()
                    _status["waiting"] = getattr(message, "prompt", None) == "<waiting>"
                    return message

                self.patches.method(worker, "receive_information", receive)

            async def step(agent, *args, _original=original_step, _status=status, **kwargs):
                owner = str(agent.name)
                _status.update(waiting=False, thinking=False)
                try:
                    if agent.message_buffer:
                        envelope = agent.message_buffer[0]
                        adapter.deliver(owner, [(str(envelope.send), str(envelope.prompt))])
                    result = await _original(*args, **kwargs)
                    if result is None and not (_status["waiting"] and not _status["thinking"]):
                        raise RuntimeError("Native worker returned an incomplete step")
                    return result
                except Exception as exc:
                    adapter.errors.append({"agent": owner, "error": type(exc).__name__})
                    raise RuntimeError(f"Native worker failed ({type(exc).__name__})") from None

            async def think(agent, message, print_prompt, print_log,
                            _original=original_think, _status=status):
                _status["thinking"] = True
                owner = str(agent.name)
                adapter.observe(owner, str(getattr(agent, "long_memory", "") or ""),
                                "retrieval-observation")
                action = await _original(message, print_prompt, print_log)
                reasoning = str(action.reply_prompt)
                if reasoning.strip():
                    if action.type == "send_message":
                        adapter.seal(owner, reasoning)
                    else:
                        adapter.engine.record_message(owner, reasoning)
                        adapter.events.append({"event": "worker_reasoning", "owner": owner,
                                               "action_type": action.type, "text": reasoning})
                return action

            self.patches.method(worker, "emulate_one_step", step)
            self.patches.method(worker, "_think", think)

        original_conclusion = self.simulator._init_conclu_agent

        def conclusion(simulator, complete_log):
            for owner in list(adapter.engine.states):
                adapter.engine.close_acquisition(owner)
            owner = "Concluder"
            adapter.engine.initialize_agent(adapter.task, owner, [adapter.task],
                                            initial_message="")
            messages = []
            for row in complete_log:
                if isinstance(row, dict) and row.get("context"):
                    messages.append((str(row.get("subjective", "unknown")), str(row["context"])))
            adapter.deliver(owner, messages, acquire=False)
            value = original_conclusion(
                "Use the current reasoning graph supplied with this request as the "
                "complete conversation evidence.")
            concluder = simulator.conclu_agent
            original = concluder.emulate_one_step

            async def final(agent, *args, **kwargs):
                answer = await original(*args, **kwargs)
                adapter.final_answer = require_text(answer, "native final answer")
                return answer

            adapter.patches.method(concluder, "emulate_one_step", final)
            return value

        self.patches.method(self.simulator, "_init_conclu_agent", conclusion)
        return self

    def detach(self):
        self.patches.restore()


def _install_transport(modules, engine, judge, patches, *, rag_documents,
                       max_actor_calls, calls, adapter_ref):
    agents, tools, utils = modules["agent"], modules["tool"], modules["utils"]
    original_init = agents.Agent.__init__
    signature = inspect.signature(original_init)

    def record_call(owner):
        calls[str(owner)] += 1
        if sum(calls.values()) > max_actor_calls:
            raise RuntimeError("Native actor call budget exceeded; the run is incomplete")

    def initialize(agent, *args, **kwargs):
        bound = signature.bind(agent, *args, **kwargs)
        bound.apply_defaults()
        requested_rag = bound.arguments["use_rag"]
        poison = bound.arguments.get("poison_rag", False)
        bound.arguments["use_rag"] = False
        original_init(*bound.args, **bound.kwargs)
        agent.aclient = ClientToken(owner=str(agent.name))
        if requested_rag:
            if not rag_documents:
                raise ValueError("RAG poisoning requires a non-empty --rag-documents corpus")
            from langchain_core.embeddings import Embeddings

            class LocalEmbeddings(Embeddings):
                def embed_documents(self, texts):
                    return [list(vector) for vector in engine.builder.embedding.embed(texts)]

                def embed_query(self, text):
                    return list(engine.builder.embedding.embed([text])[0])

            embedding = LocalEmbeddings()
            documents = [agents.Document(page_content=row["text"], metadata=row["metadata"])
                         for row in rag_documents]
            agent._graphpi_rag = agents.FAISS.from_documents(documents, embedding)
            if poison:
                injected = [agents.Document(page_content=agent.extra_data["user_input"] + text)
                            for text in agent.extra_data["misinfo_argument"]]
                agent._graphpi_rag.add_documents(injected)
            agent.embedding = embedding
            agent.use_rag = True

    async def generate(agent, prompt):
        from ..rendering import ACTOR_SYSTEM
        record_call(agent.name)
        owner = str(agent.name)
        if owner in engine.states:
            prompt = prompt + "\n\nCurrent reasoning graph:\n" + engine.view(owner)
        try:
            return await asyncio.to_thread(engine.actor.complete, [
                {"role": "system", "content": ACTOR_SYSTEM},
                {"role": "user", "content": prompt}])
        except Exception as exc:
            raise RuntimeError(f"Actor call failed ({type(exc).__name__})") from None

    def plan_generate(agent, prompt, model):
        record_call("Planner")
        agent.messages.append({"role": "user", "content": prompt})
        try:
            text = engine.actor.complete(agent.messages)
        except Exception as exc:
            raise RuntimeError(f"Planner call failed ({type(exc).__name__})") from None
        agent.messages.append({"role": "assistant", "content": text})
        return text

    async def tool_generate(prompt, model=None, aclient=None):
        record_call("Tool")
        return await asyncio.to_thread(engine.actor.complete, [{"role": "user", "content": prompt}])

    def judge_generate(prompt, model=None, print_prompt=False):
        try:
            return judge.complete([{"role": "user", "content": prompt}])
        except Exception as exc:
            raise RuntimeError(f"Judge call failed ({type(exc).__name__})") from None

    def search_rag(agent, query, num):
        return agent._graphpi_rag.similarity_search(query, k=num)

    def save_rag(agent, log):
        document = agents.Document(page_content=log.context, metadata={
            "subjective": log.subjective, "objective": log.objective, "timestamp": log.timestamp})
        agent._graphpi_rag.add_documents([document])

    patches.set(agents, "OpenAI", ClientToken)
    patches.set(agents, "AsyncOpenAI", ClientToken)
    patches.set(agents.Agent, "__init__", initialize)
    patches.set(agents.Agent, "_generate", generate)
    patches.set(agents.PlanningAgent, "_generate", plan_generate)
    patches.set(agents.Agent, "_search_rag", search_rag)
    patches.set(agents.Agent, "_save_to_rag", save_rag)
    patches.set(tools, "agenerate", tool_generate)
    patches.set(utils, "generate_with_gpt", judge_generate)
    patches.set(utils, "correct_score", parse_judge_score)
    for name in ("call_tool", "call_hijack_tool"):
        original = getattr(agents, name)

        async def tool_call(*args, _original=original, _kind=name, **kwargs):
            text = await _original(*args, **kwargs)
            token = kwargs.get("client", args[3] if len(args) > 3 else None)
            adapter = adapter_ref.get("adapter")
            if adapter is not None and getattr(token, "owner", None) in engine.states:
                adapter.observe(token.owner, str(text), "tool-observation")
                adapter.events.append({"event": _kind, "owner": token.owner})
            return text

        patches.set(agents, name, tool_call)


def run_native_misinfotask(task: MisinfoTask, engine, judge, *, benchmark_root: str | Path,
                          output_dir: str | Path, attack: str, rounds: int = 3,
                          seed: int = 0, topology: str = "auto",
                          rag_documents: Sequence[Mapping[str, Any]] | None = None,
                          max_actor_calls: int = 200) -> dict[str, Any]:
    validate_attack(attack)
    if rounds < 1 or max_actor_calls < 1 or topology not in {"auto", "chain", "full"}:
        raise ValueError("Invalid native rounds, topology, or actor call budget")
    if attack == "rag" and not rag_documents:
        raise ValueError("RAG poisoning requires --rag-documents; retrieval is never silently disabled")
    if engine.states:
        raise ValueError("Each native task/attack/seed needs a fresh GraphPI instance")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError("Native output directory must be empty")
    calls = Counter()
    patches = Patches()
    adapter_ref = {}
    with _NATIVE_LOCK, native_modules(benchmark_root) as modules:
        old_cwd, random_state = Path.cwd(), random.getstate()
        try:
            os.chdir(output_dir)
            for folder in ("log", "outputs", "eval_goal"):
                Path(folder).mkdir()
            random.seed(seed)
            _install_transport(modules, engine, judge, patches,
                               rag_documents=rag_documents, max_actor_calls=max_actor_calls,
                               calls=calls, adapter_ref=adapter_ref)
            model_spec = {"name": engine.config.actor.model, "nickname": "graphpi",
                          "api_key": "unused", "base_url": "https://api.openai.com/v1"}
            models = {"gpt-4o-mini": model_spec, "gpt-4o": model_spec}
            patches.set(modules["simulator"], "MODEL_LIST", models)
            simulator_class = modules["simulator"].Simulator
            patches.set(simulator_class, "get_dataset", lambda self: {task.id: task.to_mapping()})
            simulator = simulator_class(instance_id=task.id, model="gpt-4o-mini", topo=topology,
                                        attack_method=attack, defense=False)
            topology_result = simulator.initialize()
            adapter = NativeGraphAdapter(engine, simulator).attach()
            adapter_ref["adapter"] = adapter
            asyncio.run(simulator.emulate(rounds, topology_result,
                                          print_prompt=False, print_log=False))
            if adapter.errors:
                raise RuntimeError(f"Native worker failures: {adapter.errors}")
            if adapter.final_answer is None:
                raise RuntimeError("Native simulator did not produce a final conclusion")
            scores = simulator.scores
            for field, count in (("step_score", rounds), ("complete_score", 1),
                                 ("final_score", 1), ("task_success", 1)):
                if not isinstance(scores.get(field), list) or len(scores[field]) != count:
                    raise ValueError(f"Incomplete native score field: {field}")
                for score in scores[field]:
                    require_score(score, field)
            return {
                "task_id": task.id, "attack": attack, "seed": seed, "protocol": PROTOCOL,
                "rounds": rounds, "topology": topology, "final_answer": adapter.final_answer,
                "scores": deepcopy(scores), "native_actor_calls": dict(calls),
                "usage": engine.usage.to_dict(),
                "operations": engine.operation_counts(),
                "judge_usage": judge.tracker.to_dict(),
                "trace": list(engine.trace), "events": list(adapter.events),
                "input_fingerprint": fingerprint(task.to_mapping()),
                "retrieval": {
                    "backend": "in-memory FAISS" if attack == "rag" else None,
                    "embedding_model": engine.config.embedding.model if attack == "rag" else None,
                    "corpus_fingerprint": fingerprint(rag_documents) if attack == "rag" else None,
                    "rebuilt_with_local_embeddings": attack == "rag",
                },
            }
        finally:
            if "adapter" in adapter_ref:
                adapter_ref["adapter"].detach()
            patches.restore()
            random.setstate(random_state)
            os.chdir(old_cwd)
