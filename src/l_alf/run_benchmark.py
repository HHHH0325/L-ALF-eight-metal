from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import BenchmarkConfig, SUITES, run_suite
from l_alf.providers import create_provider

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the saved L-ALF B2 ablation implementation")
    parser.add_argument("--suite", choices=["B2"], default="B2")
    parser.add_argument("--config", default=str(ROOT / "configs" / "module_30_rounds.example.json"))
    parser.add_argument("--methods", nargs="*", help="Override the suite method list")
    parser.add_argument("--output-dir", help="Override the output directory")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--candidate-pool-size", type=int)
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip API initialization; LLM-backed methods cannot run without a provider",
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    config_root = config_path.parent
    for key in ("initial_data_path", "oracle_model_path", "output_dir"):
        value = Path(raw[key]); raw[key] = value if value.is_absolute() else (config_root / value).resolve()
    if args.output_dir: raw["output_dir"] = Path(args.output_dir).resolve()
    if args.iterations is not None: raw["iterations"] = args.iterations
    if args.repeats is not None: raw["repeats"] = args.repeats
    if args.candidate_pool_size is not None: raw["candidate_pool_size"] = args.candidate_pool_size
    cfg = BenchmarkConfig(**raw)
    methods = args.methods or SUITES.get(args.suite, [])
    provider = None
    if not args.no_llm and any(m in {"A0", "A1", "A2", "A3", "A4"} or "llm" in m or m == "lalf" for m in methods):
        provider = create_provider()
    result = run_suite(cfg, args.suite, methods, provider)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
