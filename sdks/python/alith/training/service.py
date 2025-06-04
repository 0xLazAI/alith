from fastapi import APIRouter, BackgroundTasks, status
from .types import TrainingParams, TrainingResult, TrainingStatus
from .trainer import start_trainer
from .common import generate_job_id, get_training_status as _get_training_status
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "",
    response_model=TrainingResult,
    status_code=status.HTTP_202_ACCEPTED,
    name="training",
)
async def training(params: TrainingParams, tasks: BackgroundTasks) -> TrainingResult:
    job_id = generate_job_id()
    tasks.add_task(
        start_trainer,
        params=params,
        job_id=job_id,
    )
    return TrainingResult(
        job_id=job_id,
        message=f"Training job {job_id} started",
    )


@router.get("/{job_id}", response_model=TrainingStatus)
async def get_training_status(job_id: str) -> TrainingStatus:
    return _get_training_status(job_id)
