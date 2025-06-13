from typing import Optional

from pydantic import BaseModel
from pydantic_config import SettingsModel


class DataParams(BaseModel):
    data_url: str
    encryption_key: Optional[str] = None


class LoraParams(BaseModel):
    """Configuration class for LoRA (Low-Rank Adaptation) parameters.

    This class defines hyperparameters for LoRA fine-tuning, which is used to
    adapt pre-trained models efficiently with minimal parameter changes.

    Attributes:
        rank (int): The rank of the low-rank matrix decomposition. Controls the
                    number of trainable parameters. Defaults to 8.
        alpha (int): The scaling factor for LoRA weights. Typically proportional
                     to `rank` to maintain numerical stability. Defaults to 16.
        dropout (float): The dropout probability for regularization. Set to 0 by
                         default (no dropout).
        target (str): The target module(s) to apply LoRA to. Use "all" to apply
                      to all applicable modules. Defaults to "all".
    """

    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target: str = "all"


class TrainingParams(BaseModel):
    """Configuration class for training parameters.

    Manages hyperparameters for the entire training workflow, including model
    selection, optimization strategy, and training schedule.

    Attributes:
        model (str): The name or path of the pre-trained model. Defaults to
                          "Qwen/Qwen2-0.5B".
        training_type (str): The type of training task (e.g., "sft" for Supervised
                            Fine-Tuning, "ppo" for Proximal Policy Optimization,
                            "rm" for reward model). Defaults to "sft".
                            Optional for {"pt","sft","rm","ppo","dpo","kto"}
        finetuning_type (str): The type of fine-tuning method. Supports "lora"
                               and other strategies e.g., "full" or "freeze".
                               Defaults to "lora".
                               Optional for {"lora","freeze","full"}
        lr_scheduler_type (str): The learning rate scheduler type (e.g., "cosine",
                                 "linear"). Defaults to "cosine".
        learning_rate (float): The initial learning rate for optimizer. Controls
                               the step size during parameter updates. Defaults to
                               5e-5.
        num_epochs (int): The number of training epochs. Defaults to 3.
        max_samples (int): The maximum number of training samples to use.
                           Defaults to 100,000.
        bf16 (bool): Whether to enable BF16 mixed-precision training. Defaults to
                     True.
        optim (str): The optimizer type (e.g., "adamw_torch", "sgd"). Defaults to
                     "adamw_torch".
        cutoff_len (int): The maximum sequence length for input truncation.
                          Defaults to 2048.
        flash_attn (bool): Whether to enable FlashAttention optimization for
                           faster attention computation. Defaults to False.
        save_steps (int): The interval (in steps) to save model checkpoints.
                          Defaults to 100.
        template (str): Which template to use for constructing prompts in.
                        Defaults to "default".
        lora_params (LoraParams): An instance of LoraParams class with default
                                 configurations.
    """

    model: str = "Qwen/Qwen2-0.5B"
    training_type: str = "sft"
    finetuning_type: str = "lora"
    lr_scheduler_type: str = "cosine"
    learning_rate: float = 5e-5
    num_epochs: int = 3
    max_samples: int = 100000
    bf16: bool = True
    optim: str = "adamw_torch"
    cutoff_len: int = 2048
    flash_attn: str = "auto"
    save_steps: int = 100
    template: str = "qwen"
    lora_params: LoraParams = LoraParams()
    data_params: Optional[DataParams] = None


class TrainingResult(BaseModel):
    job_id: str
    message: str


class TrainingStatus(BaseModel):
    percentage: float = 0
    current_step: int = 0
    total_steps: int = 0
    epoch: float = 0.0
    loss: float = 0.0
    elapsed_time: str = "0"
    remaining_time: str = "0"
    total_tokens: int = 0
    log: Optional[str] = None


class TrainingTask(BaseModel):
    id: str
    created_at: str
    nonce: int
    user: str
    node: str
    status: TrainingStatus


class Config(SettingsModel):
    max_training_task_queue_size: int = 2
    price_per_token: int = 100
