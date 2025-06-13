import logging
from asyncio import Lock

from fastapi import APIRouter, BackgroundTasks, Request, status

from ..lazai.request import TRAINING_TYPE, validate_request
from .common import generate_job_id
from .common import get_training_status as _get_training_status
from .trainer import start_trainer
from .types import TrainingParams, TrainingResult, TrainingStatus

logger = logging.getLogger(__name__)
router = APIRouter()
training_lock = Lock()


@router.post(
    "",
    response_model=TrainingResult,
    status_code=status.HTTP_202_ACCEPTED,
    name="training",
)
async def training(
    request: Request, params: TrainingParams, tasks: BackgroundTasks
) -> TrainingResult:
    validate_request(request, TRAINING_TYPE)
    job_id = generate_job_id()
    # We use asynchronous locks to add tasks here because training
    # tasks are resource consuming and cannot be added without restrictions,
    # and it is limited to `config.max_training_task_queue_size`
    async with training_lock:
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
