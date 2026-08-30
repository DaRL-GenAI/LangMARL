#!/usr/bin/env python3
"""Fetch and build the language benchmarks under ``env/lang_benchmark/``.

Only the small evaluation splits are kept in git. The large files -- the
HotPotQA dev sets and the 12.5k-problem MATH training split -- are rebuilt here
from their upstream sources. Both ``tasks.jsonl`` files come out byte-identical
to the ones the paper's experiments used.

HotPotQA comes from the Hugging Face mirror rather than the canonical
curtis.ml.cmu.edu URLs, which have been unreachable; the mirror carries the same
records in the same order.

Usage:
    python scripts/prepare_data.py            # everything that is missing
    python scripts/prepare_data.py qa math    # just these
    python scripts/prepare_data.py --force    # rebuild even if present
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "env" / "lang_benchmark"

HF = "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main"
HOTPOT_PARQUETS = {
    "distractor": f"{HF}/distractor/validation-00000-of-00001.parquet",
    "fullwiki": f"{HF}/fullwiki/validation-00000-of-00001.parquet",
}
MATH_PARQUET = BENCH / "MATH" / "competition_math" / "data" / "train-00000-of-00001-7320a6f3aba8ebd2.parquet"
MATH_PARQUET_URL = (
    "https://huggingface.co/datasets/qwedsacf/competition_math/resolve/main/"
    "data/train-00000-of-00001-7320a6f3aba8ebd2.parquet"
)
#: sha256 of the parquet the paper's experiments used
MATH_PARQUET_SHA256 = "2325458edc03d786939ee9e1e5795efb9e2480247b6e1ed2c51f41bea7369c6a"


def _verify_sha256(path: Path, expected: str) -> None:
    """Fail loudly if a downloaded file is not the one the paper used."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise SystemExit(
            f"{path} has sha256 {actual}, expected {expected}. "
            "The upstream file changed; delete it and re-run, or update "
            "MATH_PARQUET_SHA256 after checking what changed."
        )


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    print(f"          -> {dest}")

    def progress(count, block_size, total):
        if total <= 0:
            return
        done = min(count * block_size, total)
        pct = 100 * done / total
        sys.stdout.write(f"\r          {pct:5.1f}%  ({done >> 20} / {total >> 20} MiB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    sys.stdout.write("\n")


def _hotpot_records(parquet: Path) -> list[dict]:
    """Read a HotPotQA parquet back into the upstream JSON record shape."""
    import pandas as pd

    df = pd.read_parquet(parquet)
    records = []
    for row in df.itertuples(index=False):
        context = row.context
        supporting = row.supporting_facts
        records.append({
            "_id": row.id,
            "answer": row.answer,
            "question": row.question,
            "supporting_facts": [
                [title, int(sent_id)]
                for title, sent_id in zip(supporting["title"], supporting["sent_id"])
            ],
            "context": [
                [title, list(sentences)]
                for title, sentences in zip(context["title"], context["sentences"])
            ],
            "type": row.type,
            "level": row.level,
        })
    return records


def build_qa(force: bool = False) -> None:
    """HotPotQA: fetch the dev splits, write the raw JSON and flatten to tasks.jsonl."""
    out_dir = BENCH / "HotPotQA"
    tasks = out_dir / "tasks.jsonl"
    if tasks.exists() and not force:
        print(f"qa:   {tasks.relative_to(REPO)} already present, skipping")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / ".cache"
    cache.mkdir(exist_ok=True)

    for split, url in HOTPOT_PARQUETS.items():
        parquet = cache / f"{split}.parquet"
        raw_json = out_dir / f"hotpot_dev_{split}_v1.json"
        if raw_json.exists() and not force:
            print(f"  {raw_json.name} already present")
            continue
        if not parquet.exists() or force:
            _download(url, parquet)
        print(f"  writing {raw_json.name}")
        with open(raw_json, "w") as f:
            json.dump(_hotpot_records(parquet), f, ensure_ascii=False)

    print("  building tasks.jsonl")
    with open(out_dir / "hotpot_dev_distractor_v1.json") as f:
        raw = json.load(f)

    with open(tasks, "w") as out:
        for i, record in enumerate(raw):
            # Each context entry is [title, [sentence, ...]]; the sentences
            # already carry their own trailing spaces, so they join with "".
            context = "\n\n".join(
                f"{title}: {''.join(sentences)}" for title, sentences in record["context"]
            )
            out.write(json.dumps({
                "task_id": f"qa_{i:05d}",
                "type": "qa",
                "question": record["question"],
                "context": context,
                "answer": record["answer"],
                "metadata": {
                    "source": "hotpotqa_distractor",
                    "original_id": record["_id"],
                    "type": record["type"],
                    "level": record["level"],
                },
            }, ensure_ascii=False) + "\n")
    print(f"  wrote {len(raw)} tasks to {tasks.relative_to(REPO)}")


def _extract_boxed(solution: str) -> str | None:
    r"""Return the contents of the last ``\boxed{...}``, matching braces."""
    idx = solution.rfind("\\boxed")
    if idx < 0:
        return None
    start = idx + len("\\boxed")
    if start >= len(solution) or solution[start] != "{":
        return None
    depth = 0
    for j in range(start, len(solution)):
        if solution[j] == "{":
            depth += 1
        elif solution[j] == "}":
            depth -= 1
            if depth == 0:
                return solution[start + 1:j]
    return None


def build_math(force: bool = False) -> None:
    """MATH: flatten the competition_math training parquet into tasks.jsonl."""
    out_dir = BENCH / "MATH"
    tasks = out_dir / "tasks.jsonl"
    if tasks.exists() and not force:
        print(f"math: {tasks.relative_to(REPO)} already present, skipping")
        return

    if not MATH_PARQUET.exists():
        _download(MATH_PARQUET_URL, MATH_PARQUET)
    _verify_sha256(MATH_PARQUET, MATH_PARQUET_SHA256)

    try:
        import pandas as pd
    except ImportError:
        print("math: needs pandas + pyarrow (pip install pandas pyarrow)", file=sys.stderr)
        raise SystemExit(1)

    print("  building tasks.jsonl")
    df = pd.read_parquet(MATH_PARQUET)
    with open(tasks, "w") as out:
        for i, row in enumerate(df.itertuples(index=False)):
            # No \boxed at all -> keep the whole solution so the verifier still
            # has something to compare against. An empty \boxed{} stays empty.
            boxed = _extract_boxed(row.solution)
            answer = row.solution if boxed is None else boxed
            out.write(json.dumps({
                "task_id": f"math_{i:05d}",
                "type": "math",
                "problem": row.problem,
                "solution": row.solution,
                "answer": answer,
                "metadata": {
                    "source": "competition_math",
                    "level": row.level,
                    "subject": row.type,
                },
            }, ensure_ascii=False) + "\n")
    print(f"  wrote {len(df)} tasks to {tasks.relative_to(REPO)}")


def check_tracked(name: str, path: Path) -> None:
    if path.exists():
        print(f"{name}: {path.relative_to(REPO)} ships with the repo, nothing to do")
    else:
        print(f"{name}: expected {path.relative_to(REPO)} to be in the checkout", file=sys.stderr)


BUILDERS = {
    "qa": build_qa,
    "math": build_math,
    "coding": lambda force=False: check_tracked("coding", BENCH / "coding" / "test_tasks.jsonl"),
    "writing": lambda force=False: check_tracked("writing", BENCH / "creative_writing" / "tasks.jsonl"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tasks", nargs="*", choices=list(BUILDERS),
                        help="which benchmarks to prepare (default: all)")
    parser.add_argument("--force", action="store_true", help="rebuild even if already present")
    args = parser.parse_args()

    for name in args.tasks or list(BUILDERS):
        BUILDERS[name](force=args.force)
    print("\nDone.")


if __name__ == "__main__":
    main()
