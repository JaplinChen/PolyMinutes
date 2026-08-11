"""Self-checks for the ASR and diarization logic. Run: python -m server.test_pipeline

Model-free: the parts worth testing are the decision rules — when a collapsed decode is detected,
when a speaker's language is allowed to change, how offline clustering merges — and those are
plain functions over data.

The checks live in the `test_pipeline_*` modules; this file is the runner. Unlike the e2e suite
these share no state, so the order is for a readable log rather than for correctness — it stays
one global alphabetical sequence, which is what a single namespace gave.
"""

from __future__ import annotations

from typing import Callable

from . import test_pipeline_asr, test_pipeline_correct, test_pipeline_diarize
from . import test_pipeline_llm, test_pipeline_noise, test_pipeline_refine, test_pipeline_segment

MODULES = (test_pipeline_asr, test_pipeline_correct, test_pipeline_diarize,
           test_pipeline_llm, test_pipeline_noise, test_pipeline_refine, test_pipeline_segment)


def collect() -> list[tuple[str, Callable]]:
    found: dict[str, Callable] = {}
    for module in MODULES:
        for name, fn in vars(module).items():
            if not name.startswith("test_"):
                continue
            if name in found:
                raise AssertionError(f"two checks named {name}; one would shadow the other")
            found[name] = fn
    return sorted(found.items())


def main() -> None:
    checks = collect()
    for name, fn in checks:
        fn()
        print(f"ok  {name}")
    print(f"\n{len(checks)} passed")


if __name__ == "__main__":
    main()
