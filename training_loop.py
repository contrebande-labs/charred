import time

# misc. utils
import wandb

# jax/flax
import jax
from flax.training.common_utils import shard
from flax import jax_utils

from batch import setup_dataloader
from dataset import setup_dataset
from repository import save_to_repository
from training_step import get_training_step_lambda


def training_loop(
    text_encoder,
    text_encoder_params,
    vae,
    vae_params,
    unet,
    state,
    rng,
    max_train_steps,
    num_train_epochs,
    train_batch_size,
    output_dir,
    repo_id,
    log_wandb,
):

    # rng setup
    train_rngs = jax.random.split(rng, jax.local_device_count())

    # dataset setup
    train_dataset = setup_dataset(max_train_steps)
    print("dataset loaded...")

    # batch setup
    total_train_batch_size = train_batch_size * jax.local_device_count()
    train_dataloader = setup_dataloader(train_dataset, total_train_batch_size)
    print("dataloader setup...")

    # Create parallel version of the train step
    jax_pmap_train_step = jax.pmap(
        get_training_step_lambda(text_encoder, vae, unet), "batch", donate_argnums=(0,)
    )
    print("training step compiling...")

    # Epoch setup
    t0 = time.monotonic()
    global_training_steps = 0
    for epoch in range(num_train_epochs):

        walltime = 0

        unreplicated_train_metric = None

        epoch_steps = 0

        for batch in train_dataloader:

            batch = shard(batch)

            state, train_rngs, train_metric = jax_pmap_train_step(
                state, text_encoder_params, vae_params, batch, train_rngs
            )
            if global_training_steps == 0:
                print("training step compiled (process #%d)..." % jax.process_index())

            epoch_steps += 1
            global_training_steps += 1

            if log_wandb:
                unreplicated_train_metric = jax_utils.unreplicate(train_metric)
                walltime = time.monotonic() - t0
                wandb.log(
                    data={
                        "walltime": walltime,
                        "train/step": epoch_steps,
                        "train/global_step": global_training_steps,
                        "train/steps_per_sec": global_training_steps / walltime,
                        **{
                            f"train/{k}": v
                            for k, v in unreplicated_train_metric.items()
                        },
                    },
                    commit=True,
                )

        if log_wandb:
            wandb.log(
                data={
                    "walltime": walltime,
                    "train/epoch": epoch,
                    "train/secs_per_epoch": walltime / (epoch + 1),
                },
                commit=True,
            )

        if (epoch % 10 == 0) and repo_id is not None:

            save_to_repository(
                output_dir,
                unet,
                state.params,
                repo_id,
            )
