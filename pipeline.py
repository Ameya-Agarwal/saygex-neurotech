"""Run the analysis end to end.

    python pipeline.py              # stages 1-4 (fast)
    python pipeline.py --validate   # stages 1-5, including the null distributions
    python pipeline.py --only 3 4   # re-run particular stages
"""

import argparse
import os
import subprocess
import sys
import time

from saygex import config

ROOT = os.path.dirname(os.path.abspath(__file__))

STAGES = [
    (1, "stage1_data_cleanup.py", "load cohorts, label samples, write matrices"),
    (2, "stage2_tumor_selection.py", "pair separability, differential expression"),
    (3, "stage3_feature_panels.py", "build the anchor panel and gene lists"),
    (4, "stage4_predictive_models.py", "leak-free benchmark of every panel"),
    (5, "stage5_validation.py", "null distributions and the anchor ablation"),
]


def run_stage(number, filename, description):
    print("\n" + "=" * 72)
    print(f"STAGE {number}: {description}")
    print("=" * 72, flush=True)
    started = time.time()
    # sys.executable, not "python": the interpreter running this file is the one
    # with the dependencies, and a bare "python" on Windows can resolve to the
    # Microsoft Store stub instead.
    module = f"stages.{filename[:-3]}"
    subprocess.run([sys.executable, "-m", module], check=True, cwd=ROOT)
    print(f"[stage {number} finished in {time.time() - started:.1f}s]", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validate", action="store_true",
                        help="also run stage 5 (slow: null distributions)")
    parser.add_argument("--only", nargs="+", type=int, metavar="N",
                        help="run only these stage numbers")
    args = parser.parse_args()

    config.ensure_dirs()

    if args.only:
        selected = [s for s in STAGES if s[0] in args.only]
    else:
        limit = 5 if args.validate else 4
        selected = [s for s in STAGES if s[0] <= limit]

    started = time.time()
    for stage in selected:
        run_stage(*stage)

    print(f"\nAll requested stages complete in {time.time() - started:.1f}s.")
    print(f"Results in {os.path.relpath(config.RESULTS_DIR, ROOT)}/")


if __name__ == "__main__":
    main()
