import jax
from monitoring import wandb_init, wandb_log_validation, wandb_close
from flax.training.common_utils import shard

from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel,
)

from transformers import ByT5Tokenizer, FlaxT5Model


def validate(
    pipeline: FlaxStableDiffusionPipeline,
    num_devices: int,
    rng,
    validation_prompts: list[str],
    validation_images: list,
):

    image_logs = []

    for i, validation_prompt in enumerate(validation_prompts):

        text_inputs = shard(
            pipeline.prepare_text_inputs(num_devices * [validation_prompt])
        )

        output_images = pipeline.numpy_to_pil(
            pipeline(
                prompt_ids=text_inputs,
                prng_seed=rng,
                num_inference_steps=50,
                jit=True,
            ).images.reshape(
                (output_images.shape[0] * output_images.shape[1],)
                + output_images.shape[-3:]
            )
        )

        image_logs.append(
            {
                "validation_image": validation_images[i]
                if validation_images is not None and i < len(validation_images)
                else None,
                "images": output_images,
                "validation_prompt": validation_prompt,
            }
        )

    wandb_log_validation(image_logs)


def get_inference_validate_lambda(pretrained_unet_path, seed):

    tokenizer = ByT5Tokenizer()

    text_encoder = FlaxT5Model.from_pretrained("google/byt5-base")

    vae = FlaxAutoencoderKL.from_pretrained(
        "flax/stable-diffusion-2-1", subfolder="vae"
    )

    unet = FlaxUNet2DConditionModel.from_pretrained(pretrained_unet_path)

    scheduler = FlaxDDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    pipeline = FlaxStableDiffusionPipeline(
        text_encoder=text_encoder.encode,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=scheduler,
        feature_extractor=None,
        safety_checker=None,
    )

    num_devices = jax.device_count()

    rng = jax.random.split(jax.random.PRNGKey(seed), num_devices)

    return lambda validation_prompts, validation_images: validate(
        pipeline, num_devices, rng, validation_prompts, validation_images
    )


if __name__ == "__main__":

    wandb_init(None)
    inference_validate = get_inference_validate_lambda(
        "character-aware-diffusion/charred", 87
    )
    inference_validate(
        ["a running shoe"],
        None,
    )
    wandb_close()
