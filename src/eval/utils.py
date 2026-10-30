import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from importlib import metadata
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

import config

RESULTS_DIR = config.ROOT / "data" / "results"
STRUCTURE_SOURCE = SRC_DIR / "pipeline" / "build_structure.py"
CHUNKING_SOURCE = SRC_DIR / "pipeline" / "chunk.py"
TRACKED_PACKAGES = ("pylate", "sentence-transformers", "torch", "google-genai")
FINGERPRINTED_CONFIG_PATHS = (
    "CHUNKS_PATH",
    "SQL_DB_PATH",
    "SQL_DICTIONARY_PATH",
    "GOLD_RAG_PATH",
    "GOLD_SQL_PATH",
    "GOLD_AGENT_PATH",
)
FINGERPRINTED_SOURCES = (
    SRC_DIR / "agent" / "react_agent.py",
    SRC_DIR / "pipeline" / "generate.py",
)
STAGES = (
    "retrieval",
    "reformulation",
    "sql_generation",
    "sql_execution",
    "verification",
    "final_answer",
)

_RUN = {"id": None, "dir": None}
_CONTEXT = {}


class LatencyTracker:
    def __init__(self):
        self.durations = defaultdict(float)
        self.counts = defaultdict(int)

    def add(self, stage, seconds):
        self.durations[stage] += seconds
        self.counts[stage] += 1

    @contextmanager
    def measure(self, stage):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(stage, time.perf_counter() - start)

    def total(self):
        return self.durations.get("total", 0.0)

    def line(self):
        parts = [f"total {self.total():.2f}s"]
        for stage in STAGES:
            count = self.counts.get(stage, 0)
            if count:
                parts.append(f"{stage} {self.durations[stage]:.2f}s (x{count})")
        return " | ".join(parts)

    def as_dict(self):
        stages = {
            stage: {
                "seconds": round(self.durations[stage], 4),
                "calls": self.counts[stage],
            }
            for stage in STAGES
            if self.counts.get(stage, 0)
        }
        return {"total_seconds": round(self.total(), 4), "stages": stages}


def latency_summary(trackers):
    n = len(trackers)
    if not n:
        return {}

    total_sum = sum(t.total() for t in trackers)
    stages = {}
    tracked_sum = 0.0
    for stage in STAGES:
        stage_sum = sum(t.durations.get(stage, 0.0) for t in trackers)
        calls = sum(t.counts.get(stage, 0) for t in trackers)
        tracked_sum += stage_sum
        stages[stage] = {
            "mean_seconds": round(stage_sum / n, 4),
            "share": round(stage_sum / total_sum, 4) if total_sum else 0.0,
            "calls_per_question": round(calls / n, 2),
        }

    other_sum = total_sum - tracked_sum
    totals = sorted(t.total() for t in trackers)
    p95_index = min(n - 1, round(0.95 * (n - 1)))
    return {
        "n_questions": n,
        "stages": stages,
        "other": {
            "mean_seconds": round(other_sum / n, 4),
            "share": round(other_sum / total_sum, 4) if total_sum else 0.0,
        },
        "total": {
            "mean_seconds": round(total_sum / n, 4),
            "median_seconds": round(totals[n // 2], 4),
            "p95_seconds": round(totals[p95_index], 4),
            "max_seconds": round(totals[-1], 4),
        },
    }


def report(trackers):
    summary = latency_summary(trackers)
    if not summary:
        print("No latency data collected.")
        return

    print("\n--- Latency (cumulated per question, averaged over questions) ---")
    print(f"{'stage':<16}{'mean s':>9}{'share':>8}{'calls/q':>9}")
    for stage, values in summary["stages"].items():
        print(
            f"{stage:<16}{values['mean_seconds']:>9.2f}"
            f"{100 * values['share']:>7.0f}%{values['calls_per_question']:>9.1f}"
        )
    other = summary["other"]
    total = summary["total"]
    print(f"{'other':<16}{other['mean_seconds']:>9.2f}{100 * other['share']:>7.0f}%")
    print(f"{'total':<16}{total['mean_seconds']:>9.2f}{100:>7}%")
    print(
        f"Total per question: median {total['median_seconds']:.2f}s | "
        f"p95 {total['p95_seconds']:.2f}s | max {total['max_seconds']:.2f}s"
    )


def to_jsonable(value):
    if isinstance(value, Path):
        try:
            return value.relative_to(config.ROOT).as_posix()
        except ValueError:
            return value.as_posix()
    return value


def config_values():
    return {
        name: to_jsonable(getattr(config, name))
        for name in dir(config)
        if name.isupper() and name != "ROOT"
    }


def literal_constants(path):
    if not path.exists():
        return {}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        is_simple_assign = (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        )
        if not is_simple_assign or not node.targets[0].id.isupper():
            continue
        try:
            values[node.targets[0].id] = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            continue
    return values


def describe_pipeline():
    cfg = config_values()
    return {
        "ocr": {"model": cfg.get("OCR_MODEL"), "pdf": cfg.get("PDF_PATH")},
        "structure": literal_constants(STRUCTURE_SOURCE),
        "chunking": literal_constants(CHUNKING_SOURCE),
        "indexing": {
            "embedding_model": cfg.get("EMBEDDING_MODEL"),
            "index_name": cfg.get("INDEX_NAME"),
            "index_dir": cfg.get("INDEX_DIR"),
        },
        "retrieval": {"top_k": cfg.get("RAG_TOP_K")},
        "agent": {
            "llm": cfg.get("GEMINI_MODEL"),
            "max_iterations": cfg.get("AGENT_MAX_ITERATIONS"),
            "sql_max_rows": cfg.get("SQL_MAX_ROWS"),
            "sql_database": cfg.get("SQL_DB_PATH"),
            "sql_dictionary": cfg.get("SQL_DICTIONARY_PATH"),
        },
    }


def run_git(*args):
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=config.ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def git_state():
    commit = run_git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None}
    changes = run_git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(changes)}


def environment():
    packages = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None

    device = "unknown"
    try:
        import torch

        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": packages,
    }


def sha256_of(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprints():
    paths = [getattr(config, name, None) for name in FINGERPRINTED_CONFIG_PATHS]
    paths.extend(FINGERPRINTED_SOURCES)
    result = {}
    for path in paths:
        if path is not None and Path(path).is_file():
            result[to_jsonable(Path(path))] = {
                "sha256": sha256_of(Path(path)),
                "bytes": Path(path).stat().st_size,
            }
    return result


def run_context():
    if not _CONTEXT:
        _CONTEXT.update(
            {
                "pipeline": describe_pipeline(),
                "config": config_values(),
                "git": git_state(),
                "environment": environment(),
                "fingerprints": fingerprints(),
            }
        )
    return _CONTEXT


def start_run():
    if _RUN["dir"] is None:
        context = run_context()
        commit = (context["git"]["commit"] or "nogit")[:7]
        suffix = "-dirty" if context["git"]["dirty"] else ""
        run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{commit}{suffix}"
        run_dir = RESULTS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        _RUN.update({"id": run_id, "dir": run_dir})
    return _RUN["dir"]


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def save_result(name, metrics, questions, trackers=None):
    run_dir = start_run()
    payload = {
        "eval": name,
        "run_id": _RUN["id"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "context": run_context(),
        "metrics": metrics,
        "questions": questions,
    }
    if trackers:
        payload["latency"] = latency_summary(trackers)
    path = run_dir / f"{name}.json"
    write_json(path, payload)
    print(f"Results saved to {path}")
    return path


def save_manifest(all_metrics):
    run_dir = start_run()
    payload = {
        "run_id": _RUN["id"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "context": run_context(),
        "metrics": all_metrics,
    }
    path = run_dir / "manifest.json"
    write_json(path, payload)
    print(f"Manifest saved to {path}")
    return path