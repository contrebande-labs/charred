import time

from jax import pmap
from jax.tree_util import tree_map
from jax.random import split
from flax.training.common_utils import shard
from flax.jax_utils import replicate

from monitoring import get_wandb_log_batch_lambda
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
    unet_training_state,
    rng,
    max_train_steps,
    num_train_epochs,
    train_batch_size,
    output_dir,
    log_wandb,
    get_validation_predictions,
    num_devices,
):

    # replication setup
    unet_training_state = replicate(unet_training_state)
    rng = split(rng, num_devices)

    # dataset setup
    train_dataset = setup_dataset(max_train_steps)
    print("dataset loaded...")

    # batch setup
    total_train_batch_size = train_batch_size * num_devices
    train_dataloader = setup_dataloader(train_dataset, total_train_batch_size)
    print("dataloader setup...")

    # milestone setup
    milestone_step_count = min(100_000, max_train_steps)
    print(f"milestone step count: {milestone_step_count}")

    # wandb setup
    if log_wandb:
        wandb_log_batch = get_wandb_log_batch_lambda(
            get_validation_predictions,
        )
        print("wand log batch setup...")

    # Epoch setup
    t0 = time.monotonic()
    global_training_steps = 0
    global_walltime = time.monotonic()
    for epoch in range(num_train_epochs):

        is_first_step = global_training_steps == 0

        if is_first_step:
            print("entering first epoch...")

        for batch in train_dataloader:

            if is_first_step:
                print("entering first batch...")

            batch_walltime = time.monotonic()

            # Create parallel version of the train step
            # TODO: Should we try "axis_size=num_devices" or "axis_size=total_train_batch_size" ?
            unet_training_state, rng, train_metrics = pmap(
                # cannot send these as static broadcasted arguments because they are not hashable
                # TODO: rewrite text_encoder, vae and unet as pure
                fun=get_training_step_lambda(
                    text_encoder, text_encoder_params, vae, vae_params, unet
                ),
                axis_name="batch",
                in_axes=(0, 0, 0),
                out_axes=(0, 0, 0),
                static_broadcasted_argnums=(),
                # We cannot donate the "batch" argument. Otherwise, we get this:
                #   /site-packages/jax/_src/interpreters/mlir.py:711: UserWarning: Some donated buffers were not usable: ShapedArray(int32[8,1024]), ShapedArray(float32[8,3,512,512]).
                #   See an explanation at https://jax.readthedocs.io/en/latest/faq.html#buffer-donation.
                #   warnings.warn(f"Some donated buffers were not usable: {', '.join(unused_donations)}.\n{msg}")
                donate_argnums=(
                    1,
                    2,
                ),  # donating rng and training state
            )(
                unet_training_state,
                rng,
                shard(batch),
            )

            # TODO: should we "average" and re-replicate the training state here before it's sent back again in the next step, or used for evaluation or saved to disc?

            if is_first_step:
                print("computed first batch...")

            global_training_steps += num_devices

            is_milestone = (
                True if global_training_steps % milestone_step_count == 0 else False
            )

            if log_wandb:
                # TODO: is this correct? was only unreplicated before, with no averaging
                global_walltime = time.monotonic() - t0
                delta_time = time.monotonic() - batch_walltime
                wandb_log_batch(
                    global_walltime,
                    global_training_steps,
                    delta_time,
                    epoch,
                    train_metrics,
                    unet_training_state.params,
                    is_milestone,
                )
                if is_first_step:
                    print("logged first batch...")

            if is_milestone:
                save_to_local_directory(
                    f"{ output_dir }/{ str(global_training_steps).zfill(12) }",
                    unet,
                    # TODO: is this ok?
                    # was: jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params))
                    # then: jax.device_get(flax.jax_utils.unreplicate(state.params))
                    # and then, also: jax.device_get(state.params)
                    # and then, again: unreplicate(state.params)
                    # Finally found a way to average along the splits/device/partition/shard axis
                    tree_map(
                        f=lambda x: x.mean(axis=0), tree=unet_training_state.params
                    ),
                )
