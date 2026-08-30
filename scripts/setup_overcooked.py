#!/usr/bin/env python3
"""Build and install the overcooked_ai the Overcooked environment needs.

LangMARL's ProAgent planner is written against overcooked_ai 0.0.1 -- the
version the paper's experiments ran on -- with a set of local modifications.
Neither half is installable on its own:

* 0.0.1 was never published to PyPI (which starts at 1.0.0), so the base has to
  come from the ``neurips2019`` tag of the upstream repository.
* The modifications live in ``env/overcooked_ai/`` in this repository. They add
  the search helpers ProAgent imports (``find_path``, ``get_visitable_positions``,
  ``get_intersect_counter``, ``query_counter_states``) and extra medium-level
  actions, and they drop the four high-level planner classes -- which
  ``agents/agent.py`` still imports, so those are merged back in here.

The PyPI release cannot be used instead: 1.1.0 renamed ``MediumLevelPlanner`` to
``MediumLevelActionManager``, dropped the ``order_list`` state API the 0.0.1
planner relies on, and renamed every layout file.

Usage:
    python scripts/setup_overcooked.py            # build and install
    python scripts/setup_overcooked.py --build-only
    python scripts/setup_overcooked.py --check    # verify an existing install
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FORK = REPO / "env" / "overcooked_ai" / "overcooked_ai_py"
BUILD = REPO / "env" / "overcooked_ai" / ".build"
TARBALL_URL = (
    "https://codeload.github.com/HumanCompatibleAI/overcooked_ai/tar.gz/refs/tags/neurips2019"
)

#: Overlaid wholesale -- the fork's versions are supersets of 0.0.1's.
OVERLAY = ["planning/search.py", "planning/__init__.py", "utils.py", "static.py"]

#: Present in 0.0.1's planners.py, dropped by the fork, still imported by
#: agents/agent.py. Merged back onto the end of the fork's file.
MERGE_BACK_FROM = "class HighLevelAction"


def _download(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {TARBALL_URL}")
    urllib.request.urlretrieve(TARBALL_URL, dest)
    print(f"  -> {dest} ({dest.stat().st_size >> 20} MiB)")


def build(force: bool = False) -> Path:
    """Assemble the patched source tree and return its path."""
    if BUILD.exists() and not force:
        print(f"  {BUILD.relative_to(REPO)} already built (use --force to rebuild)")
        return BUILD

    if not FORK.is_dir():
        raise SystemExit(f"missing {FORK.relative_to(REPO)}; is this a full checkout?")

    tarball = BUILD.parent / ".neurips2019.tar.gz"
    if not tarball.exists() or force:
        _download(tarball)

    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)

    print("  extracting")
    with tarfile.open(tarball) as tf:
        prefix = tf.getnames()[0].split("/")[0] + "/"
        for member in tf.getmembers():
            if not member.name.startswith(prefix):
                continue
            member.name = member.name[len(prefix):]
            if member.name:
                tf.extract(member, BUILD, filter="data")

    pkg = BUILD / "overcooked_ai_py"
    if not pkg.is_dir():
        raise SystemExit(f"unexpected tarball layout: no overcooked_ai_py in {BUILD}")

    print("  applying the local modifications")
    for rel in OVERLAY:
        src, dst = FORK / rel, pkg / rel
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    # planners.py needs merging rather than replacing: the fork's copy adds the
    # ProAgent medium-level actions but removes the high-level classes that
    # agents/agent.py imports.
    base_planners = (pkg / "planning" / "planners.py").read_text()
    fork_planners = (FORK / "planning" / "planners.py").read_text()
    if MERGE_BACK_FROM not in base_planners:
        raise SystemExit(
            f"cannot find {MERGE_BACK_FROM!r} in the upstream planners.py; "
            "the tag's contents changed and this script needs updating."
        )
    tail = base_planners[base_planners.index(MERGE_BACK_FROM):]
    merged = fork_planners.rstrip() + "\n\n\n" + tail
    (pkg / "planning" / "planners.py").write_text(merged)
    classes = re.findall(r"^class (\w+)", merged, re.M)
    print(f"    planners.py now defines: {', '.join(classes)}")

    print(f"  built {BUILD.relative_to(REPO)}")
    return BUILD


def install(build_dir: Path) -> None:
    """Editable-install the tree, keeping its layout and planner data in place."""
    # 0.0.1's setup.py uses find_packages(), which misses data/ (it has no
    # __init__.py). An editable install keeps the data files reachable.
    cmd = [sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(build_dir)]
    if shutil.which("uv"):
        cmd = ["uv", "pip", "install", "--no-deps", "-e", str(build_dir)]
    print(f"  {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def check() -> bool:
    """Verify the pieces LangMARL's Overcooked environment actually uses."""
    ok = True
    try:
        from overcooked_ai_py.planning.planners import Heuristic, MediumLevelPlanner  # noqa: F401
        print("  MediumLevelPlanner, Heuristic     ok")
    except Exception as exc:
        print(f"  planners                          FAILED: {exc}")
        ok = False
    try:
        from overcooked_ai_py.planning.search import (  # noqa: F401
            find_path,
            get_intersect_counter,
            get_visitable_positions,
            query_counter_states,
        )
        print("  ProAgent search helpers           ok")
    except Exception as exc:
        print(f"  search helpers                    FAILED: {exc}")
        ok = False
    try:
        from overcooked_ai_py.static import LAYOUTS_DIR

        n = len(list(Path(LAYOUTS_DIR).glob("*.layout")))
        print(f"  layouts                           ok ({n} found)")
    except Exception as exc:
        print(f"  layouts                           FAILED: {exc}")
        ok = False
    try:
        import langmarl

        if "overcooked" in langmarl.list_envs():
            print("  langmarl registers 'overcooked'   ok")
        else:
            print("  langmarl registers 'overcooked'   FAILED: not registered")
            ok = False
    except Exception as exc:
        print(f"  langmarl                          FAILED: {exc}")
        ok = False
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--build-only", action="store_true", help="assemble without installing")
    parser.add_argument("--check", action="store_true", help="only verify an existing install")
    parser.add_argument("--force", action="store_true", help="re-download and rebuild")
    args = parser.parse_args()

    if args.check:
        raise SystemExit(0 if check() else 1)

    build_dir = build(force=args.force)
    if args.build_only:
        print("\nBuilt but not installed.")
        return

    install(build_dir)

    print("\nVerifying:")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--check"],
        cwd=REPO,
    )
    if result.returncode != 0:
        raise SystemExit(1)
    print("\nDone. Overcooked is ready.")


if __name__ == "__main__":
    main()
