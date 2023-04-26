# imports
import time

import jax

import wandb


def log_init_wandb(args):
    wandb.init(
        entity="charred",
        project="charred",
        job_type="train",
        config=args,
    )
    wandb.config.update(
        {
            "num_devices": jax.device_count(),
        }
    )
    wandb.define_metric("train/*", step_metric="train/global_step")
    wandb.define_metric("val/*", step_metric="val/global_step")
    wandb.define_metric("train/global_step", step_metric="walltime")
    wandb.define_metric("train/epoch", step_metric="train/global_step")
    wandb.define_metric("train/secs_per_epoch", step_metric="train/epoch")


def log_wandb_finish():
    wandb.finish()
    print("WandB closed...")


def log_train_step_metrics(
    global_walltime,
    epoch_steps,
    global_training_steps,
    epoch,
    unreplicated_train_metric,
    t0,
    batch_walltime,
):
    global_walltime = time.monotonic() - t0
    delta_time = time.monotonic() - batch_walltime
    wandb.log(
        data={
            "walltime": global_walltime,
            "train/step": epoch_steps,
            "train/global_step": global_training_steps,
            "train/steps_per_sec": 1 / delta_time,
            "train/epoch": epoch,
            **{f"train/{k}": v for k, v in unreplicated_train_metric.items()},
        },
        commit=True,
    )
    return global_walltime


def log_train_epoch_metrics(epoch_walltime, global_walltime, global_training_steps):
    epoch_walltime = global_walltime - epoch_walltime
    wandb.log(
        data={
            "train/secs_per_epoch": epoch_walltime,
            "train/global_step": global_training_steps,
        },
        commit=True,
    )
    return epoch_walltime


def log_val_metrics(global_val_steps, unreplicated_val_metric):
    wandb.log(
        data={
            "val/global_step": global_val_steps,
            **{f"val/{k}": v for k, v in unreplicated_val_metric.items()},
        },
        commit=True,
    )
