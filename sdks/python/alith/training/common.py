import json
import os
from pathlib import Path
from typing import Any, Optional

from .types import TrainingStatus

RUNNING_LOG = "running_log.txt"
TRAINING_LOG = "trainer_log.jsonl"
TRAINING_ARGS = "training_args.yaml"
TRAINING_TYPES = {
    "Supervised Fine-Tuning": "sft",
    "Reward Modeling": "rm",
    "PPO": "ppo",
    "DPO": "dpo",
    "KTO": "kto",
    "Pre-Training": "pt",
}
DEFAULT_DATASET_DIR: Path = Path(__file__).parent.parent.joinpath("data").absolute()
DEFAULT_DATASET: str = "identity"
DATASET_INFO_FILE: str = "dataset_info.json"


def add_dataset(dataset: str, path: str):
    """Add dataset and data file into the dataset info json file."""
    file = DEFAULT_DATASET_DIR / DATASET_INFO_FILE
    with open(file) as fp:
        data = json.load(fp)
        data[dataset]["file_name"] = path
    file.write_text(json.dumps(data))


def get_output_dir(job_id: str) -> str:
    return f"saves/train_{job_id}"


def get_training_status(job_id: str) -> TrainingStatus:
    return get_training_status_from_dir(get_output_dir(job_id))


def get_running_log(job_id: str, max: int = 20) -> str:
    return get_running_log_from_dir(get_output_dir(job_id), max)


def get_training_status_from_dir(path: str) -> TrainingStatus:
    training_log_path = os.path.join(path, TRAINING_LOG)
    if os.path.isfile(training_log_path):
        trainer_log: list[dict[str, Any]] = []
        with open(training_log_path, encoding="utf-8") as f:
            for line in f:
                trainer_log.append(json.loads(line))

        steps, losses = [], []

        for log in trainer_log:
            if log.get("loss", None):
                steps.append(log["current_steps"])
                losses.append(log["loss"])

        if trainer_log:
            latest_log = trainer_log[-1]
            percentage = latest_log["percentage"]
            return TrainingStatus(
                percentage=percentage,
                current_step=steps[-1],
                loss=losses[-1],
                total_steps=latest_log["total_steps"],
                elapsed_time=latest_log["elapsed_time"],
                remaining_time=latest_log["remaining_time"],
                total_tokens=latest_log["total_tokens"],
                epoch=latest_log["epoch"],
                log=get_running_log_from_dir(path),
            )
    return TrainingStatus(
        log=get_running_log_from_dir(path),
    )


def get_running_log_from_dir(path: str, max: int = 20) -> Optional[str]:
    running_log_path = os.path.join(path, RUNNING_LOG)
    if os.path.isfile(running_log_path):
        with open(running_log_path, encoding="utf-8") as f:
            running_log = f.read()[-max:]  # avoid lengthy log
            return running_log
    return None


def generate_job_id() -> str:
    return f"{os.urandom(4).hex()}"
