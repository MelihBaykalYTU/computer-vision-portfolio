"""New 2026 AI-assisted educational demo; not recovered 2024 source/results."""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
import platform
import shutil
import struct
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
BASE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
GRIDS = {"gd": [0.001, 0.01, 0.1],
         "sgd": [0.00001, 0.0001, 0.001],
         "adam": [0.00001, 0.0001, 0.001]}
PROVENANCE = (
    "New AI-assisted educational companion implemented on 2026-10-08, inspired by "
    "the 2024 joint coursework of Ceyda Tolunay and Melih Baykal. Not recovered "
    "original source code, not a historical result reproduction, and not a claim "
    "of Melih Baykal's individual historical performance."
)

def digest(path, algorithm="sha256"):
    return hashlib.new(algorithm, Path(path).read_bytes()).hexdigest()

def fetch_one(item, directory):
    name, expected = item
    dest = directory / name
    if not dest.exists() or digest(dest, "md5") != expected:
        with urllib.request.urlopen(BASE + name, timeout=60) as response:
            dest.write_bytes(response.read())
    if digest(dest, "md5") != expected:
        raise ValueError("Official dataset checksum mismatch: " + name)
    return {"file": name, "url": BASE + name, "md5": expected, "sha256": digest(dest)}

def idx(path):
    raw = gzip.decompress(path.read_bytes())
    magic, n = struct.unpack(">II", raw[:8])
    if magic == 2049:
        x = np.frombuffer(raw, dtype=np.uint8, offset=8)
        assert x.size == n
        return x
    assert magic == 2051
    height, width = struct.unpack(">II", raw[8:16])
    assert (height, width) == (28, 28)
    x = np.frombuffer(raw, dtype=np.uint8, offset=16)
    assert x.size == n * height * width
    return x.reshape(n, height * width)

def prepare(directory, seed):
    directory.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        sources = list(pool.map(lambda item: fetch_one(item, directory), FILES.items()))
    train_x = idx(directory / "train-images-idx3-ubyte.gz")
    train_y = idx(directory / "train-labels-idx1-ubyte.gz")
    test_x = idx(directory / "t10k-images-idx3-ubyte.gz")
    test_y = idx(directory / "t10k-labels-idx1-ubyte.gz")
    rng = np.random.default_rng(seed)
    splits = {name: [] for name in ("train", "validation", "test")}
    for cls in (0, 6):
        ids = rng.permutation(np.flatnonzero(train_y == cls))
        splits["train"].extend(ids[:500].tolist())
        splits["validation"].extend(ids[500:600].tolist())
        ids_test = rng.permutation(np.flatnonzero(test_y == cls))
        splits["test"].extend(ids_test[:500].tolist())
    assert not set(splits["train"]) & set(splits["validation"])
    paths = {}
    counts = {}
    for name, ids in splits.items():
        ids = np.asarray(ids, dtype=np.int64)
        x, y = (test_x, test_y) if name == "test" else (train_x, train_y)
        yy = np.where(y[ids] == 0, -1, 1)
        matrix = np.column_stack((yy, x[ids].astype(np.float64) / 255.0))
        path = directory / (name + ".csv")
        np.savetxt(path, matrix, fmt="%.9g", delimiter=",",
                   header=f"{len(ids)},784", comments="")
        paths[name] = path
        counts[name] = {"total": len(ids),
                        "class_0": int(np.count_nonzero(y[ids] == 0)),
                        "class_6": int(np.count_nonzero(y[ids] == 6)),
                        "official_source": "test" if name == "test" else "train",
                        "index_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                        "prepared_csv_sha256": digest(path)}
    (directory / "split_indices.json").write_text(
        json.dumps({"seed": seed, "indices": splits}, indent=2), encoding="utf-8")
    return paths, counts, sources

def execute(exe, method, lr, epochs, seed, paths, directory, tag, evaluate_test):
    trace = directory / (tag + ".csv")
    model = directory / (tag + ".bin")
    args = [str(exe), "--train", method, str(lr), str(epochs), str(seed),
            str(paths["train"]), str(paths["validation"]),
            str(paths["test"]) if evaluate_test else "-", str(trace), str(model)]
    start = time.perf_counter()
    run = subprocess.run(args, check=True, capture_output=True, text=True)
    result = json.loads(run.stdout)
    result["wall_seconds"] = time.perf_counter() - start
    result["model_sha256"] = digest(model)
    frame = np.genfromtxt(trace, delimiter=",", names=True)
    for col in ("train_loss", "validation_loss", "train_accuracy", "validation_accuracy"):
        assert np.isfinite(frame[col]).all(), (method, col)
    assert np.isfinite(result["train_loss"]) and np.isfinite(result["validation_loss"])
    if evaluate_test:
        assert np.isfinite(result["test_loss"])
    return result, trace, frame

def main():
    parser = argparse.ArgumentParser(description=PROVENANCE)
    parser.add_argument("--work-dir", type=Path,
                        default=Path.home() / ".cache" / "melih-optimizer-demo")
    parser.add_argument("--gcc", default="gcc")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 10000 or not 1 <= args.seed < 2**32:
        parser.error("epochs must be 1..10000 and seed must be nonzero uint32")
    directory = args.work_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    assets = here / "assets"
    assets.mkdir(exist_ok=True)
    exe = directory / "optimizer.exe"
    command = [args.gcc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
               str(here / "optimizer.c"), "-o", str(exe), "-lm"]
    build = subprocess.run(command, check=True, capture_output=True, text=True)
    verification = json.loads(subprocess.check_output([str(exe), "--selftest"], text=True))
    assert verification["passed"]
    paths, counts, sources = prepare(directory / "data", args.seed)
    trials, selected, histories = [], [], {}
    for method, grid in GRIDS.items():
        candidates = []
        for lr in grid:
            result, _, _ = execute(exe, method, lr, args.epochs, args.seed, paths,
                                   directory, f"tune_{method}_{lr}", False)
            assert not result["test_evaluated"]
            candidates.append(result)
            trials.append(result)
        winner = min(candidates, key=lambda r: r["validation_loss"])
        result, trace, history = execute(
            exe, method, winner["lr"], args.epochs, args.seed, paths,
            directory, f"selected_{method}", True)
        repeat, _, repeat_history = execute(
            exe, method, winner["lr"], args.epochs, args.seed, paths,
            directory, f"repeat_{method}", True)
        exact = result["model_sha256"] == repeat["model_sha256"]
        for key in history.dtype.names:
            if key != "c_clock_seconds":
                exact = exact and np.array_equal(history[key], repeat_history[key])
        assert exact, "fixed-seed reproducibility failed"
        assert result["model_sha256"] == winner["model_sha256"]
        result["fixed_seed_repeat_passed"] = bool(exact)
        selected.append(result)
        histories[method] = history
        shutil.copyfile(trace, assets / (method + "_trace.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    for method, history in histories.items():
        axes[0].plot(history["epoch"], history["validation_loss"], label=method.upper())
        axes[1].plot(history["updates"], history["validation_loss"], label=method.upper())
        axes[2].plot(history["c_clock_seconds"], history["validation_loss"], label=method.upper())
    for ax, xlabel in zip(axes, ("Epoch", "Parameter updates", "C clock seconds (platform dependent)")):
        ax.set_xlabel(xlabel); ax.set_ylabel("Validation half-MSE")
        ax.grid(alpha=0.25); ax.legend()
    axes[1].set_xscale("symlog", linthresh=10)
    fig.suptitle("2026 AI-assisted NEW educational demo — not 2024 results", fontsize=11)
    fig.tight_layout()
    fig.savefig(assets / "validation_comparison.png", dpi=150)
    plt.close(fig)
    summary = {
        "created_date": "2026-10-08", "provenance": PROVENANCE,
        "original_report_limitations": [
            "Data-preparation screenshot selects classes 1/6; training screenshot specifies 4/6.",
            "Report prose alternates between 40 test samples total and 40 per class; screenshot shows 20 per class.",
            "Original .c/.py files unavailable; historical plots/results not reproduced."
        ],
        "new_experiment": {
            "classes": {"0": "T-shirt/top (-1)", "6": "Shirt (+1)"},
            "seed": args.seed, "epochs": args.epochs, "counts": counts,
            "model": "tanh(w dot [1,pixels/255]), 785 parameters",
            "loss": "mean(0.5 * (prediction - target)^2)",
            "accuracy": "threshold tanh output at zero; labels -1/+1",
            "selection": "Separate optimizer LR grids; lowest final-epoch validation loss only.",
            "grids": GRIDS,
            "split_checks": {
                "train_validation_indices_disjoint": True,
                "test_uses_only_official_test_partition": True,
                "preprocessing": "fixed pixels/255; no fitted statistics",
                "test_not_read_by_C_during_tuning": True
            },
            "gd": "one mean full-batch gradient update per epoch",
            "sgd": "one sample gradient update per training example; shuffled each epoch",
            "adam": "one sample gradient update; beta1=.9,beta2=.999,epsilon=1e-8; bias correction",
            "comparison_limits": "Same epoch budget, different update counts; independently tuned LRs. No claim of universally best optimizer or historical/personal achievement."
        },
        "verification": {
            **verification, "compile_command_template": ["gcc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "optimizer.c", "-o", "<work-dir>/optimizer.exe", "-lm"], "compiler_stdout": build.stdout,
            "compiler_stderr": build.stderr, "warnings_as_errors_passed": True,
            "fixed_seed_reproducibility_all_optimizers": True,
            "all_reported_losses_and_weights_finite": True,
            "same_model_before_and_after_test_evaluation": True
        },
        "dataset_sources": sources,
        "implementation": {
            "optimizer_c_sha256": digest(here / "optimizer.c"),
            "runner_python_sha256": digest(here / "run_demo.py"),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "matplotlib_version": matplotlib.__version__,
            "platform": platform.system() + " " + platform.machine(),
            "compiler_version": subprocess.check_output([args.gcc, "--version"], text=True).splitlines()[0]
        },
        "dataset_license": {
            "name": "MIT", "copyright": "2017 Zalando SE",
            "url": "https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE",
            "citation": "Han Xiao, Kashif Rasul, Roland Vollgraf (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. arXiv:1708.07747."
        },
        "adam_reference": "Diederik P. Kingma and Jimmy Ba (2014), Adam: A Method for Stochastic Optimization. https://arxiv.org/abs/1412.6980",
        "validation_trials": trials, "selected_final_test_results": selected,
        "timing_note": "C clock() is a platform-dependent clock measurement, not asserted to be CPU time. CSV clocks include training and per-epoch evaluation; stdout clocks also include final model serialization/test evaluation. Python perf_counter wall_seconds includes process startup, data load and final test. Machine-dependent timings are excluded from deterministic checks."
    }
    (assets / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"assets":str(assets), "selected":selected, "verification":summary["verification"]}, indent=2))

if __name__ == "__main__":
    main()
