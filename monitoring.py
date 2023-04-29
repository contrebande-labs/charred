import wandb
import jax


def wandb_inference_init():
    wandb.init(
        entity="charred",
        project="charred-inference",
        job_type="inference",
    )
    wandb.config.update(
        {
            "num_devices": jax.device_count(),
        }
    )

    print("WandB inference init...")


def wandb_inference_log(log: list):
    wandb_log = []

    for entry in log:
        wandb_log.append(wandb.Image(entry["image"], caption=entry["prompt"]))

    wandb.log({"inference": wandb_log})

    print("WandB inference log...")


def wandb_init(args):
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
    wandb.define_metric("*", step_metric="step")
    wandb.define_metric("step", step_metric="walltime")

    print("WandB setup...")


def wandb_close():
    wandb.finish()

    print("WandB closed...")


def get_wandb_log_step_lambda(
    get_predictions,
):
    def __wandb_log_step(
        global_walltime,
        global_training_steps,
        delta_time,
        epoch,
        unreplicated_train_metric,
        unet_params,
    ):
        is_milestone = True if global_training_steps % 10_000 == 0 else False

        log_data = {
            "walltime": global_walltime,
            "step": global_training_steps,
            "batch_delta_time": delta_time,
            "epoch": epoch,
            **{k: v for k, v in unreplicated_train_metric.items()},
        }

        if is_milestone:
            log_data["validation"] = [
                wandb.Image(image, caption=prompt)
                for prompt, image in get_predictions(unet_params)
            ]

        wandb.log(
            data=log_data,
            commit=is_milestone,
        )

    return __wandb_log_step
