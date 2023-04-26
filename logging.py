import logging
import jax


def wandb_init(args):

    logging.init(
        entity="charred",
        project="charred",
        job_type="train",
        config=args,
    )
    logging.config.update(
        {
            "num_devices": jax.device_count(),
        }
    )
    logging.define_metric("*", step_metric="train/global_step")
    logging.define_metric("train/global_step", step_metric="walltime")
    logging.define_metric("train/epoch", step_metric="train/global_step")
    logging.define_metric("train/secs_per_epoch", step_metric="train/epoch")

    print("WandB setup...")


def wandb_close():

    logging.finish()

    print("WandB closed...")


def wandb_log_step(
    global_walltime,
    epoch_steps,
    global_training_steps,
    delta_time,
    epoch,
    unreplicated_train_metric,
):
    logging.log(
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


def wandb_log_epoch(epoch_walltime, global_training_steps):
    logging.log(
        data={
            "train/secs_per_epoch": epoch_walltime,
            "train/global_step": global_training_steps,
        },
        commit=True,
    )


def wandb_log_validation(image_logs):
    formatted_images = []
    for log in image_logs:
        images = log["images"]
        validation_prompt = log["validation_prompt"]
        validation_image = log["validation_image"]

        formatted_images.append(
            logging.Image(validation_image, caption="Controlnet conditioning")
        )
        for image in images:
            image = logging.Image(image, caption=validation_prompt)
            formatted_images.append(image)

    logging.log({"validation": formatted_images})
