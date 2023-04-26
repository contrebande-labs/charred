import time

import jax
from flax import jax_utils
from flax.training.common_utils import shard

from wandb import wandb_log_epoch, wandb_log_step
from batch import setup_dataloader
from dataset import setup_dataset
from repository import save_to_local_directory
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
    global_walltime = time.monotonic()
    for epoch in range(num_train_epochs):
        epoch_walltime = time.monotonic()
        epoch_steps = 0
        unreplicated_train_metric = None

        for batch in train_dataloader:
            batch_walltime = time.monotonic()
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
                global_walltime = time.monotonic() - t0
                delta_time = time.monotonic() - batch_walltime
                wandb_log_step(
                    global_walltime,
                    epoch_steps,
                    global_training_steps,
                    delta_time,
                    epoch,
                    unreplicated_train_metric,
                )

        if log_wandb:
            epoch_walltime = global_walltime - epoch_walltime
            wandb_log_epoch(epoch_walltime, global_training_steps)

        if epoch % 10 == 0:
            save_to_local_directory(
                f"{ output_dir }/{ str(epoch).zfill(6) }",
                unet,
                state.params,
            )
