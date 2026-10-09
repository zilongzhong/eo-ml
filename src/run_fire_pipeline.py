"""Run preprocessing, pytest, training, and held-out evaluation in order."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import torch
import yaml

from src.config import load_config


def now():
    return datetime.now(ZoneInfo("America/Toronto"))


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_stage(label, command, project_dir, environment, log):
    heading = f"\n[{label}] {' '.join(command)}\n"
    print(heading, flush=True)
    log.write(heading)
    log.flush()
    with subprocess.Popen(
        command, cwd=project_dir, env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fire_segmentation.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest after preprocessing")
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    os.chdir(project_dir)
    config = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Launch Docker with --gpus all.")

    run_name = args.run_name or now().strftime("fire_%Y%m%d_%H%M%S_%f")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("Run name may contain only letters, numbers, '_' and '-'")
    output_dir = Path(config["output"]["output_dir"])
    run_dir = output_dir / "runs" / run_name
    checkpoint = Path(config["output"]["checkpoint_dir"]) / run_name / "best.pt"
    if checkpoint.parent.exists():
        raise FileExistsError(f"Checkpoint run already exists: {checkpoint.parent}")
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshot = run_dir / "config.yaml"
    snapshot.write_text(yaml.safe_dump(config, sort_keys=False))
    evaluation_dir = output_dir / "evaluation" / run_name
    summary_path = run_dir / "summary.json"
    summary = {
        "run_name": run_name, "status": "running", "started_at": now().isoformat(),
        "config": str(snapshot), "checkpoint": str(checkpoint),
        "train_regions": config["data"]["train_regions"],
        "val_regions": config["data"]["val_regions"],
        "test_regions": config["data"]["test_regions"],
        "evaluation_dir": str(evaluation_dir), "log": str(run_dir / "pipeline.log"),
        "history_csv": str(output_dir / "tables" / f"{run_name}_training_history.csv"),
        "tensorboard_dir": str(Path(config["output"]["tensorboard_dir"]) / run_name),
    }
    environment = dict(os.environ, PYTHONUNBUFFERED="1", EO_FIRE_CONFIG=str(snapshot))
    module_command = [sys.executable, "-u", "-m"]
    config_arguments = ["--config", str(snapshot)]
    stages = [
        ("training statistics", module_command + ["src.data.compute_fire_stats"] + config_arguments),
        ("patch index", module_command + ["src.data.build_fire_patch_index"] + config_arguments),
    ]
    if not args.skip_tests:
        stages.append(("pytest", module_command + [
            "pytest", "-q", f"--junitxml={run_dir / 'pytest.xml'}",
        ]))
    stages.extend([
        ("dataset figure", module_command + ["src.data.check_fire_dataloader"] + config_arguments + [
            "--output", str(output_dir / "figures" / "dataset_samples.png"),
        ]),
        ("training", module_command + ["src.train_fire"] + config_arguments + ["--run-name", run_name]),
        ("held-out evaluation", module_command + ["src.evaluate_fire"] + config_arguments + [
            "--checkpoint", str(checkpoint),
        ]),
    ])
    save_json(summary_path, summary)
    with (run_dir / "pipeline.log").open("w") as log:
        try:
            for stage, command in stages:
                summary["stage"] = stage
                save_json(summary_path, summary)
                run_stage(stage, command, project_dir, environment, log)
            summary["held_out_metrics"] = json.loads((evaluation_dir / "metrics.json").read_text())
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            summary["best_epoch"] = saved["epoch"]
            summary["best_validation_metrics"] = saved["validation_metrics"]
            del saved
            summary.update(status="complete", finished_at=now().isoformat())
            save_json(summary_path, summary)
            save_json(output_dir / "latest_run.json", summary)
            report = (
                f"\nRun complete: {run_name}\nSummary: {summary_path}\n"
                f"Latest successful run: {output_dir / 'latest_run.json'}\n"
                f"Held-out results: {evaluation_dir}\n"
            )
            for row in summary["held_out_metrics"]:
                report += (
                    f"Region {row['region']}: IoU={row['iou']:.4f}, F1={row['f1']:.4f}, "
                    f"Precision={row['precision']:.4f}, Recall={row['recall']:.4f}\n"
                )
            print(report, flush=True)
            log.write(report)
        except Exception as error:
            summary.update(status="failed", finished_at=now().isoformat(), error=str(error))
            save_json(summary_path, summary)
            log.write(f"\nFAILED: {error}\n")
            raise


if __name__ == "__main__":
    main()
