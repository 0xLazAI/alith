from pathlib import Path

import rsa
from llamafactory.train.tuner import run_exp

from ..data import decrypt, download_file
from .common import DEFAULT_DATASET, DEFAULT_DATASET_DIR, add_dataset, get_output_dir
from .types import TrainingParams
from ..lazai.node.validator import rsa_private_key


def preprocess_data(params: TrainingParams) -> str:
    """Preprocess the data return the trained dataset name"""
    dataset = DEFAULT_DATASET
    # If file url is provided, download the file and decrpt it
    if params.data_params:
        file = download_file(params.data_params.data_url)
        file_path = Path(file)
        dataset = file_path.name
        # If the file is encrypted and provided the encryption key, decrypt it
        if params.data_params.encryption_key and rsa_private_key:
            encryption_key = params.data_params.encryption_key
            content = file_path.read_bytes()
            priv_key = rsa.PrivateKey.load_pkcs1(rsa_private_key.strip().encode())
            password = rsa.decrypt(
                bytes.fromhex(encryption_key.removeprefix("0x")), priv_key
            )
            decrypted_data_bytes = decrypt(content, password=password.decode())
            file_path.write_bytes(decrypted_data_bytes)
        add_dataset(dataset, file)
    return dataset


def start_trainer(params: TrainingParams, job_id: str):
    """Here we use the llamafactory to train the model"""
    run_exp(
        {
            "do_train": True,
            "model_name_or_path": params.model,
            "stage": params.training_type,
            "finetuning_type": params.finetuning_type,
            "lr_scheduler_type": params.lr_scheduler_type,
            "learning_rate": params.learning_rate,
            "num_train_epochs": params.num_epochs,
            "max_samples": params.max_samples,
            "bf16": params.bf16,
            "optim": params.optim,
            "cutoff_len": params.cutoff_len,
            "flash_attn": params.flash_attn,
            "save_steps": params.save_steps,
            "template": params.template,
            "lora_rank": params.lora_params.rank,
            "lora_alpha": params.lora_params.alpha,
            "lora_dropout": params.lora_params.dropout,
            "lora_target": params.lora_params.target,
            "preprocessing_num_workers": 16,
            "dataset": preprocess_data(params),
            "dataset_dir": str(DEFAULT_DATASET_DIR),
            "output_dir": get_output_dir(job_id),
            "include_num_input_tokens_seen": True,
            "ddp_timeout": 180000000,
            "trust_remote_code": True,
            "plot_loss": True,
            "report_to": "none",
            "packing": False,
            "warmup_steps": 0,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "max_grad_norm": 1.0,
            "logging_steps": 5,
        }
    )


if __name__ == "__main__":
    start_trainer(params=TrainingParams(), job_id="0xFFFF")
