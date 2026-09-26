from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run graph-based multi-agent collaboration")
    parser.add_argument("--config", type=Path, help="JSON configuration")
    parser.add_argument("--input", type=Path, required=True,
                        help="JSON object with task and observations by agent")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate input and configuration without calling models")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    item = json.loads(args.input.read_text(encoding="utf-8"))
    if (not isinstance(item, dict) or not isinstance(item.get("task"), str)
            or not item["task"].strip() or not isinstance(item.get("observations"), dict)
            or not item["observations"]):
        parser.error("Input needs a nonempty task and observations mapping")
    for owner, texts in item["observations"].items():
        if (not owner or not isinstance(texts, list)
                or any(not isinstance(text, str) for text in texts)):
            parser.error("Each agent must have a list of observation strings")
    if not isinstance(item.get("final_instruction", ""), str):
        parser.error("final_instruction must be a string")
    if args.rounds is not None and args.rounds < 0:
        parser.error("--rounds must be nonnegative")
    if args.validate_only:
        print(json.dumps({"valid": True, "agents": len(item["observations"]),
                          "actor": config.actor.model}))
        return
    from .engine import GraphPI
    if args.output.exists():
        parser.error("Output exists; choose a new output path")
    result = GraphPI(config).run(item["task"], item["observations"],
                                rounds=args.rounds, final_instruction=item.get("final_instruction", ""))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=True)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "final_answers": result["final_answers"]}))


if __name__ == "__main__":
    main()
