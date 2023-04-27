import jax
from flax.training.common_utils import shard
import jax.numpy as jnp

from monitoring import wandb_init, wandb_log_validation, wandb_close

from diffusers import (
    FlaxAutoencoderKL,
    FlaxDPMSolverMultistepScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel,
)

from transformers import ByT5Tokenizer, FlaxT5ForConditionalGeneration


def predict(
    pipeline: FlaxStableDiffusionPipeline,
    tokenizer,
    rng,
    validation_prompts: list[str],
    num_inference_steps: int,
):

    output_images = []

    for validation_prompt in validation_prompts:

        text_input_ids = tokenizer(
            text=validation_prompt,
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        ).input_ids[0]

        output_images.append(
            pipeline.numpy_to_pil(
                pipeline(
                    params={},
                    prompt_ids=text_input_ids,
                    prng_seed=rng,
                    num_inference_steps=num_inference_steps,
                    jit=True,
                ).images.reshape(
                    (output_images.shape[0] * output_images.shape[1],)
                    + output_images.shape[-3:]
                )
            )
        )

    return output_images


def log_validate(
    pipeline: FlaxStableDiffusionPipeline,
    tokenizer,
    rng,
    validation_prompts: list[str],
    validation_images: list,
    num_inference_steps: int,
):

    predicted_images = predict(
        pipeline,
        tokenizer,
        rng,
        validation_prompts,
        num_inference_steps,
    )

    image_logs = []

    for i, predicted_image in enumerate(predicted_images):

        image_logs.append(
            {
                "validation_image": validation_images[i]
                if validation_images is not None and i < len(validation_images)
                else None,
                "images": predicted_image,
                "validation_prompt": validation_prompts[i],
            }
        )

    wandb_log_validation(image_logs)


def get_inference_log_validate_lambda(pretrained_unet_path, seed):

    tokenizer = ByT5Tokenizer()

    language_model = FlaxT5ForConditionalGeneration.from_pretrained(
        "google/byt5-base",
        dtype=jnp.bfloat16,
    )

    vae, _ = FlaxAutoencoderKL.from_pretrained(
        "flax/stable-diffusion-2-1",
        subfolder="vae",
        dtype=jnp.bfloat16,
    )

    unet, _ = FlaxUNet2DConditionModel.from_pretrained(
        pretrained_unet_path,
        dtype=jnp.bfloat16,
    )

    scheduler = FlaxDPMSolverMultistepScheduler.from_config(
        config={
            "_diffusers_version": "0.16.0",
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "clip_sample": False,
            "num_train_timesteps": 1000,
            "prediction_type": "v_prediction",
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "trained_betas": None,
        }
    )

    pipeline = FlaxStableDiffusionPipeline(
        tokenizer=tokenizer,
        text_encoder=language_model.encode,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        feature_extractor=None,
        safety_checker=None,
    )

    rng = jax.random.PRNGKey(seed)

    return (
        lambda validation_prompts, validation_images, num_inference_steps: log_validate(
            pipeline,
            tokenizer,
            rng,
            validation_prompts,
            validation_images,
            num_inference_steps,
        )
    )


if __name__ == "__main__":

    wandb_init(None)
    get_inference_log_validate_lambda("character-aware-diffusion/charred", 87)(
        ["a running shoe"], None, 20
    )
    wandb_close()
