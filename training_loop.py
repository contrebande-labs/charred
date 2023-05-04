import time

from jax import pmap
from jax.random import split
from flax.training.common_utils import shard
from flax.jax_utils import replicate, unreplicate
from jax.profiler import start_trace, stop_trace, save_device_memory_profile, device_memory_profile

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

    # Create parallel version of the train step
    # TODO: Should we try "axis_size=num_devices" or "axis_size=total_train_batch_size" ?
    jax_pmapped_training_step = pmap(
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
        # donating rng and training state
        donate_argnums=(
            0,
            1,
        ),
    )

    # Epoch setup
    t0 = time.monotonic()
    global_training_steps = 0
    global_walltime = time.monotonic()
    is_compilation_step = True
    is_first_compiled_step = False
    loss = None
    for epoch in range(num_train_epochs):

        for batch in train_dataloader:

            # getting batch start time
            batch_walltime = time.monotonic()   

            if is_compilation_step:
                print("computing compilation batch...")
                device_memory_profile()
                start_trace(log_dir="./profiling/compilation_step", create_perfetto_link=False, create_perfetto_trace=True)
            elif is_first_compiled_step:
                print("computing first compiled batch...")
                device_memory_profile()
                start_trace(log_dir="./profiling/first_compiled_step", create_perfetto_link=False, create_perfetto_trace=True)

            # training step
            # TODO: Fix this jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED: Error loading program: Attempting to allocate 1.28G. That was not possible. There are 785.61M free.; (0x0x0_HBM0): while running replica 0 and partition 0 of a replicated computation (other replicas may have failed as well).
            unet_training_state, rng, loss = jax_pmapped_training_step(
                unet_training_state,
                rng,
                shard(batch),
            )

            # block until train step has completed
            loss.block_until_ready()

            if is_compilation_step:
                stop_trace()
                save_device_memory_profile(filename="./profiling/compilation_step/pprof_memory_profile.pb")
                print("computed compilation batch...")
            elif is_first_compiled_step:
                stop_trace()
                save_device_memory_profile(filename="./profiling/first_compiled_step/pprof_memory_profile.pb")
                print("computed first compiled batch...")

            global_training_steps += num_devices

            # checking if current batch is a milestone
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
                    loss,
                    unet_training_state.params,
                    is_milestone,
                )

            if is_milestone:
                save_to_local_directory(
                    f"{ output_dir }/{ str(global_training_steps).zfill(12) }",
                    unet,
                    # TODO: is this ok?
                    # was: jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params))
                    # then: jax.device_get(flax.jax_utils.unreplicate(state.params))
                    # and then, also: jax.device_get(state.params)
                    # and then, again: unreplicate(state.params)
                    # Finally found a way to average along the splits/device/partition/shard axis: jax.tree_util.tree_map(f=lambda x: x.mean(axis=0), tree=unet_training_state.params),
                    unreplicate(tree=unet_training_state.params)
                )


        if is_compilation_step:
            is_compilation_step = False
            is_first_compiled_step = True
        elif is_first_compiled_step:
            is_first_compiled_step = False
