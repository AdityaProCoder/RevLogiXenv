from __future__ import annotations

import argparse
import json
import os
from typing import Iterable

from dotenv import load_dotenv

load_dotenv()

from RevLogiXenv_v0 import (
    BaselineAgent,
    BaselineRunConfig,
    LocalBaselineRunner,
)


def _parse_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_int_csv(value: str) -> list[int]:
    out: list[int] = []
    for token in _parse_csv(value):
        out.append(int(token))
    return out


def _auto_detect_provider() -> str:
    """Detect LLM provider from available API keys."""
    if os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN"):
        return "openai"
    raise ValueError(
        "No LLM API key found. Set OPENAI_API_KEY or HF_TOKEN in .env"
    )


def run_local(tasks: Iterable[str], seeds: Iterable[int]) -> dict:
    runner = LocalBaselineRunner()
    return runner.run(tasks=tasks, seeds=seeds)


def run_llm(
    provider: str,
    model: str | None,
    temperature: float | None,
    tasks: Iterable[str],
    seeds: Iterable[int],
    verbose: bool,
) -> dict:
    config = BaselineRunConfig(provider=provider, model=model, temperature=temperature)
    agent = BaselineAgent(config)
    return agent.run_benchmark(tasks=tasks, seeds=seeds, verbose=verbose)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "AutonomousReturns-v0 baseline CLI. "
            "Supports deterministic local baseline and LLM baseline (OpenAI)."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("local", "llm"),
        default="local",
        help="Execution mode: local (deterministic) or llm (auto/OpenAI/Google).",
    )
    parser.add_argument(
        "--tasks",
        default="easy,medium,hard",
        help="Comma-separated task list (default: easy,medium,hard).",
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated integer seeds (default: 42).",
    )
    parser.add_argument(
        "--provider",
        choices=("auto", "openai"),
        default="auto",
        help="LLM provider (used only in --mode llm).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override for LLM mode.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM temperature (0.0-1.0). Default: 0.15. Lower = more deterministic.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose rollout logging (mostly useful for LLM mode).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    tasks = _parse_csv(args.tasks)
    seeds = _parse_int_csv(args.seeds)

    if not tasks:
        raise ValueError("No tasks specified.")
    if not seeds:
        raise ValueError("No seeds specified.")

    if args.mode == "local":
        report = run_local(tasks=tasks, seeds=seeds)
    else:
        provider = args.provider if args.provider != "auto" else _auto_detect_provider()
        report = run_llm(
            provider=provider,
            model=args.model,
            temperature=args.temperature,
            tasks=tasks,
            seeds=seeds,
            verbose=args.verbose,
        )

    if args.pretty:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(json.dumps(report))


if __name__ == "__main__":
    main()
