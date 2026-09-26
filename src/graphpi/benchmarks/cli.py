from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from .common import (fingerprint, load_records, select_ids, write_json, write_jsonl)
from .hiddenbench import load_hiddenbench, run_hiddenbench, score_hiddenbench, PROTOCOL as HB_PROTOCOL
from .misinfotask import ATTACKS, load_misinfotask, prepare_cases, score_misinfotask


def _data_arguments(parser):
    parser.add_argument("--data", type=Path, required=True, help="Benchmark JSON or JSONL")
    parser.add_argument("--task-id", action="append", help="Select IDs; repeat as needed")
    parser.add_argument("--limit", type=int, help="Explicitly limit the selected tasks")


def _evaluation_arguments(parser):
    _data_arguments(parser)
    parser.add_argument("--config", type=Path, help="GraphPI JSON configuration")
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--predictions", type=Path, help="Score existing results without model calls")
    parser.add_argument("--seed", type=int, action="append", help="Random seed; repeat as needed")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without model calls")
    parser.add_argument("--allow-partial", action="store_true", help="Permit an incomplete saved result set")


def _empty_output(path):
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError("Output directory must be new or empty")


def _selected(args, loader):
    return select_ids(loader(args.data), args.task_id, args.limit)


def _seeds(args):
    seeds = args.seed if args.seed is not None else [0]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct")
    return seeds


def hiddenbench_main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate HiddenBench with GraphPI or score saved votes")
    _evaluation_arguments(parser)
    parser.add_argument("--rounds", type=int, default=6,
                        help="Synchronous discussion rounds before the final decision (default: 6)")
    args = parser.parse_args(argv)
    tasks = _selected(args, load_hiddenbench)
    seeds = _seeds(args)
    if args.rounds < 0:
        parser.error("--rounds must be nonnegative")
    from ..config import load_config
    config = load_config(args.config)
    plan = {
        "benchmark": "HiddenBench", "protocol": HB_PROTOCOL,
        "num_tasks": len(tasks), "num_runs": len(tasks) * len(seeds),
        "rounds": args.rounds, "seeds": seeds,
        "dataset_fingerprint": fingerprint([task.to_mapping() for task in tasks]),
        "task_ids": [task.id for task in tasks], "config": config.to_dict(),
    }
    if args.dry_run:
        if args.predictions:
            result = score_hiddenbench(tasks, load_records(args.predictions))
            plan["prediction_validation"] = {"complete": result["complete"], "num_runs": result["num_runs"]}
        print(json.dumps(plan, indent=2))
        return 0
    _empty_output(args.output)
    write_json(args.output / "manifest.json", plan)
    if args.predictions:
        runs = load_records(args.predictions)
    else:
        from ..engine import GraphPI
        runs = []
        for task in tasks:
            for seed in seeds:
                result = run_hiddenbench(task, GraphPI(config), seed=seed, rounds=args.rounds)
                runs.append(result)
                write_json(args.output / "cases" / (fingerprint((task.id, seed))[:20] + ".json"), result)
                print(json.dumps({"task_id": task.id, "seed": seed, "status": "complete"}), flush=True)
    summary = score_hiddenbench(tasks, runs)
    write_jsonl(args.output / "runs.jsonl", runs)
    write_json(args.output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("num_tasks", "average_accuracy", "majority_accuracy", "complete")}))
    return int(not summary["complete"] and not args.allow_partial)


def misinfotask_main(argv=None):
    parser = argparse.ArgumentParser(description="Run native MisinfoTask attacks with GraphPI or score saved results")
    _evaluation_arguments(parser)
    parser.add_argument("--benchmark-root", type=Path, help="Independent native benchmark checkout")
    parser.add_argument("--attack", action="append", choices=ATTACKS,
                        help="Attack settings; default: all three")
    parser.add_argument("--rounds", type=int, default=3, help="Native simulator steps (default: 3)")
    parser.add_argument("--topology", choices=("auto", "chain", "full"), default="auto")
    parser.add_argument("--rag-documents", type=Path, help="Clean JSONL/text retrieval corpus for RAG poisoning")
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="Standard OpenAI judge model")
    parser.add_argument("--max-actor-calls", type=int, default=200,
                        help="Per native episode call cap; exceeding it fails the run")
    args = parser.parse_args(argv)
    tasks = _selected(args, load_misinfotask)
    seeds, attacks = _seeds(args), args.attack or list(ATTACKS)
    cases = prepare_cases(tasks, attacks, seeds)
    if args.rounds < 1 or args.max_actor_calls < 1:
        parser.error("--rounds and --max-actor-calls must be positive")
    from ..config import load_config
    config = load_config(args.config)
    if args.config is None:
        config = replace(config, algorithm=replace(config.algorithm, budget_scope="receive"))
    plan = {
        "benchmark": "MisinfoTask", "protocol": "native-misinfotask-graphpi/1",
        "num_tasks": len(tasks), "num_runs": len(cases), "attacks": attacks,
        "rounds": args.rounds, "topology": args.topology, "seeds": seeds,
        "dataset_fingerprint": fingerprint([task.to_mapping() for task in tasks]),
        "config": config.to_dict(), "judge_model": args.judge_model,
    }
    if args.predictions:
        runs = load_records(args.predictions)
        summary = score_misinfotask(tasks, runs, attacks=attacks)
    else:
        root = args.benchmark_root.resolve() if args.benchmark_root is not None else None
        prerequisites = []
        if root is None:
            prerequisites.append("Supply --benchmark-root for native execution.")
        else:
            for name in ("agent.py", "simulator.py", "utils.py", "tool.py", "generate.py", "llm.py"):
                if not (root / "sandbox" / name).is_file():
                    prerequisites.append(f"Benchmark checkout is missing sandbox/{name}.")
        from .native_misinfotask import load_rag_documents
        corpus = load_rag_documents(args.rag_documents) if args.rag_documents else None
        if "rag" in attacks and not corpus:
            prerequisites.append("Supply --rag-documents for RAG poisoning.")
        plan["execution_prerequisites"] = prerequisites
        plan["native_dependencies_verified"] = False
        plan["retrieval_corpus_fingerprint"] = fingerprint(corpus) if corpus else None
        if prerequisites and not args.dry_run:
            parser.error(" ".join(prerequisites))
    if args.dry_run:
        if args.predictions:
            plan["prediction_validation"] = {"complete": summary["complete"], "num_runs": summary["num_runs"]}
        print(json.dumps(plan, indent=2))
        return 0
    _empty_output(args.output)
    write_json(args.output / "manifest.json", plan)
    if not args.predictions:
        from ..engine import GraphPI
        from ..clients import ChatClient
        from .native_misinfotask import run_native_misinfotask
        runs = []
        tasks_by_id = {task.id: task for task in tasks}
        for case in cases:
            engine = GraphPI(config)
            judge = ChatClient(replace(config.actor, model=args.judge_model), component="judge")
            case_id = fingerprint((case["task_id"], case["attack"], case["seed"]))[:20]
            result = run_native_misinfotask(
                tasks_by_id[case["task_id"]], engine, judge,
                benchmark_root=root, output_dir=args.output / "native" / case_id,
                attack=case["attack"], rounds=args.rounds, seed=case["seed"],
                topology=args.topology, rag_documents=corpus, max_actor_calls=args.max_actor_calls)
            runs.append(result)
            write_json(args.output / "cases" / (case_id + ".json"), result)
            print(json.dumps({"case_id": case["case_id"], "status": "complete"}), flush=True)
        summary = score_misinfotask(tasks, runs, attacks=attacks)
    write_jsonl(args.output / "runs.jsonl", runs)
    write_json(args.output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in
                      ("num_runs", "average_misinformation_toxicity", "average_task_success_rate", "complete")}))
    return int(not summary["complete"] and not args.allow_partial)


def prepare_main(argv=None):
    parser = argparse.ArgumentParser(description="Normalize benchmark schemas and save a deterministic case plan")
    _data_arguments(parser)
    parser.add_argument("--benchmark", choices=("hiddenbench", "misinfotask"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="New normalized JSONL dataset")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--attack", action="append", choices=ATTACKS)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        parser.error("Preparation output already exists")
    loader = load_hiddenbench if args.benchmark == "hiddenbench" else load_misinfotask
    tasks, seeds = _selected(args, loader), _seeds(args)
    if args.benchmark == "hiddenbench":
        if args.attack:
            parser.error("--attack is only valid for MisinfoTask")
        cases = [{"task_id": task.id, "seed": seed, "fact_assignments": task.allocate(seed)}
                 for task in tasks for seed in seeds]
    else:
        cases = [{key: value for key, value in row.items() if key != "task"}
                 for row in prepare_cases(tasks, args.attack or ATTACKS, seeds)]
    records = [task.to_mapping() for task in tasks]
    manifest = {
        "benchmark": args.benchmark, "num_tasks": len(tasks), "num_cases": len(cases),
        "dataset_fingerprint": fingerprint(records), "seeds": seeds, "cases": cases,
    }
    write_jsonl(args.output, records)
    write_json(args.output.with_suffix(".manifest.json"), manifest)
    print(json.dumps({"num_tasks": len(tasks), "num_cases": len(cases), "output": str(args.output)}))
    return 0


def safe_main(function):
    try:
        return function()
    except Exception as exc:
        print(f"Evaluation failed ({type(exc).__name__}); check inputs and local configuration.", file=sys.stderr)
        return 1
