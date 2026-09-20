"""Terminal loop: read questions, print the answer with citations, steps and timings."""
from __future__ import annotations

import logging
import time

from backend.agent import TurnResult, answer
from backend.citations import cited

EXIT_WORDS = {"exit", "quit", "q"}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")   # trace lines land here
    print("recite.net — ask a question (or 'quit')")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if not question:
            continue
        if question.lower() in EXIT_WORDS:
            print("Bye.")
            return
        started = time.monotonic()
        try:
            result = answer(question)
        except Exception as e:  # noqa: BLE001 — keep the loop alive and show the error
            print(f"\n[error] {e}")
            continue
        elapsed = time.monotonic() - started
        print()
        print(result.text)
        print()
        print(f"cited:     {cited(result.text)}")
        print(f"retrieved: {citations(result)}")
        print(f"grounding: {result.grounding} (retried: {result.retried})")
        print(f"steps: {result.steps} | passages: {len(result.passages)} | total: {elapsed:.1f}s")


def citations(result: TurnResult) -> str:
    """Pages of the passages that were retrieved for the model."""
    seen: list[tuple[str, int]] = []
    for p in result.passages:
        key = (p.doc_name, p.page)
        if key not in seen:
            seen.append(key)
    return ", ".join(f"{name}, p. {page}" for name, page in seen) or "(none)"


if __name__ == "__main__":
    main()
