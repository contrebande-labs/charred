import time

import jax
from flax import jax_utils
from flax.training.common_utils import shard

from monitoring import get_wandb_log_step_lambda
from batch import setup_dataloader
from dataset import setup_dataset
from repository import save_to_local_directory
from training_step import get_training_step_lambda


def get_training_state_params_from_devices(params):
    return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))


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
    get_validation_predictions,
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

    wandb_log_step = get_wandb_log_step_lambda(
        get_validation_predictions,
    )

    # Epoch setup
    t0 = time.monotonic()
    global_training_steps = 0
    global_walltime = time.monotonic()
    for epoch in range(num_train_epochs):
        unreplicated_train_metric = None

        for batch in train_dataloader:
            batch_walltime = time.monotonic()

            batch = shard(batch)

            state, train_rngs, train_metrics = jax_pmap_train_step(
                state, text_encoder_params, vae_params, batch, train_rngs
            )

            global_training_steps += 1

            if log_wandb:
                unreplicated_train_metric = jax_utils.unreplicate(train_metrics)
                global_walltime = time.monotonic() - t0
                delta_time = time.monotonic() - batch_walltime
                wandb_log_step(
                    global_walltime,
                    global_training_steps,
                    delta_time,
                    epoch,
                    unreplicated_train_metric,
                    state.params,
                )

        if epoch % 10 == 0:
            save_to_local_directory(
                f"{ output_dir }/{ str(epoch).zfill(6) }",
                unet,
                get_training_state_params_from_devices(state.params),
            )
