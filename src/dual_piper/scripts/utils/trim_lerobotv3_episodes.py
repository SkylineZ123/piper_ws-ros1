#!/home/user/anaconda3/envs/lerobot/bin/python
"""Trim leading episodes from a LeRobot v3.0 dataset.

In LeRobot v3.0 multiple episodes are packed into shared parquet/mp4 files, so
you cannot simply delete per-episode files. This script relies on
``lerobot.datasets.dataset_tools.delete_episodes`` which correctly:
  - filters and re-indexes the data parquet rows (episode_index / index / ...)
  - cuts and re-concatenates the shared video files per kept episode
  - rebuilds meta/episodes/*, recomputes stats.json and info.json

Typical use case: the first N episodes of a teleoperation collection are low
quality and hurt VLA training accuracy, so you keep only the last K episodes.

Examples:
    # Keep only the last 80 episodes (drop the first total-80)
    python trim_lerobotv3_episodes.py \
        --dataset /home/user/hdd_4t/data/data_lerobot3/dualpiper/BloodGasAnalysis_20260618 \
        --keep-last 80

    # Equivalent: explicitly drop the first 70 episodes
    python trim_lerobotv3_episodes.py \
        --dataset /home/user/hdd_4t/data/data_lerobot3/dualpiper/BloodGasAnalysis_20260618 \
        --drop-first 70

By default the trimmed dataset is written to a sibling directory
``<name>_trimmed`` and the original is left untouched. Pass ``--in-place`` to
replace the original directory once the trimmed copy is built successfully.

NOTE: this must run inside an environment that has lerobot >= 0.5.x with v3.0
support (the shebang points at the `lerobot` conda env on this machine).
"""

import argparse
import shutil
import sys
from pathlib import Path

try:
    from lerobot.datasets.dataset_tools import delete_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as e:  # pragma: no cover - environment guard
    sys.exit(
        f"Failed to import lerobot ({e}).\n"
        "Run this script with an env that has lerobot v3.0, e.g.:\n"
        "  /home/user/anaconda3/envs/lerobot/bin/python trim_lerobotv3_episodes.py ..."
    )


def derive_repo_id(dataset_dir: Path) -> str:
    """Build a `<parent>/<name>` repo id from the dataset directory path.

    LeRobotDataset requires a repo_id of the form ``namespace/name``. We reuse
    the dataset folder name and its parent so the id is stable and human
    readable, but the value only matters for local bookkeeping here.
    """
    return f"{dataset_dir.parent.name}/{dataset_dir.name}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drop leading episodes from a LeRobot v3.0 dataset (keep the tail).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to the LeRobot v3.0 dataset root (the dir containing meta/, data/, videos/).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--keep-last",
        type=int,
        help="Number of trailing episodes to keep; all earlier episodes are deleted.",
    )
    group.add_argument(
        "--drop-first",
        type=int,
        help="Number of leading episodes to delete from the front.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the trimmed dataset. Default: '<dataset>_trimmed' sibling dir.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace the original dataset dir with the trimmed result once it is built.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset.expanduser().resolve()
    if not (dataset_dir / "meta" / "info.json").exists():
        sys.exit(f"'{dataset_dir}' does not look like a LeRobot dataset (no meta/info.json).")

    repo_id = derive_repo_id(dataset_dir)
    print(f"Loading dataset: {dataset_dir}\n  repo_id = {repo_id}")
    src = LeRobotDataset(repo_id=repo_id, root=dataset_dir)
    total = src.meta.total_episodes
    print(f"  total_episodes = {total}, total_frames = {src.meta.total_frames}")

    # Resolve how many leading episodes to drop.
    if args.keep_last is not None:
        if not (0 < args.keep_last <= total):
            sys.exit(f"--keep-last must be in 1..{total}, got {args.keep_last}.")
        drop_first = total - args.keep_last
    else:
        drop_first = args.drop_first
        if not (0 <= drop_first < total):
            sys.exit(f"--drop-first must be in 0..{total - 1}, got {drop_first}.")

    if drop_first == 0:
        print("Nothing to drop (every episode would be kept). Exiting.")
        return

    episodes_to_delete = list(range(drop_first))  # indices 0 .. drop_first-1
    kept = total - drop_first
    print(
        f"\nWill DELETE the first {drop_first} episodes (indices 0..{drop_first - 1}) "
        f"and KEEP the last {kept} episodes (indices {drop_first}..{total - 1})."
    )

    # Determine output location. When operating in place we still build into a
    # temporary sibling dir first, then swap, so a crash never corrupts the source.
    default_out = dataset_dir.parent / f"{dataset_dir.name}_trimmed"
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else default_out)
    if output_dir == dataset_dir:
        sys.exit("Output dir must differ from the source dataset dir.")
    if output_dir.exists():
        sys.exit(f"Output dir already exists: {output_dir}\nRemove it or pass a different --output-dir.")

    print(f"Trimmed dataset will be written to: {output_dir}")
    if args.in_place:
        print(f"After success the ORIGINAL '{dataset_dir}' will be REPLACED by the trimmed copy.")

    if not args.yes:
        resp = input("\nProceed? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("Aborted.")
            return

    new_repo_id = f"{repo_id}_trimmed"
    print("\nTrimming... (re-encoding videos and rebuilding metadata, this may take a while)")
    delete_episodes(
        dataset=src,
        episode_indices=episodes_to_delete,
        output_dir=output_dir,
        repo_id=new_repo_id,
    )
    print(f"Done. Trimmed dataset written to: {output_dir}")

    if args.in_place:
        backup = dataset_dir.parent / f"{dataset_dir.name}_orig_backup"
        if backup.exists():
            sys.exit(
                f"Cannot swap in place: backup path already exists: {backup}\n"
                f"Trimmed dataset is safe at {output_dir}; move it manually."
            )
        print(f"\nSwapping in place:\n  {dataset_dir} -> {backup}\n  {output_dir} -> {dataset_dir}")
        shutil.move(str(dataset_dir), str(backup))
        shutil.move(str(output_dir), str(dataset_dir))
        print(
            f"In-place trim complete. Original backed up at: {backup}\n"
            "Verify the trimmed dataset, then delete the backup when satisfied."
        )


if __name__ == "__main__":
    main()