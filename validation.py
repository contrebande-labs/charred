import jax
from logging import wandb_log_validation
from flax.training.common_utils import shard

from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel,
)

from transformers import ByT5Tokenizer, FlaxT5Model

def get_validate_lambda(pretrained_unet_path, pipeline_params, rng):

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
        safety_checker=None,
        feature_extractor=None,
    )

    num_samples = jax.device_count()

    prng_seed = jax.random.split(rng, num_samples)

    def __validate_lambda(validation_prompts, validation_images):

        image_logs = []

        for validation_prompt, validation_image in zip(
            validation_prompts, validation_images
        ):

            text_inputs = shard(
                pipeline.prepare_text_inputs(
                    num_samples * [validation_prompt]
                )
            )

            image_inputs = shard(
                pipeline.prepare_image_inputs(
                    num_samples * [validation_image]
                )
            )

            output_images = pipeline(
                prompt_ids=text_inputs,
                image=image_inputs,
                params=pipeline_params,
                prng_seed=prng_seed,
                num_inference_steps=50,
                jit=True,
            ).images

            reshaped_output_pil_images = pipeline.numpy_to_pil(
                output_images.reshape(
                    (output_images.shape[0] * output_images.shape[1],) + output_images.shape[-3:]
                )
            )

            image_logs.append(
                {
                    "validation_image": validation_image,
                    "images": reshaped_output_pil_images,
                    "validation_prompt": validation_prompt,
                }
            )

        wandb_log_validation(image_logs)

    return lambda validation_prompts, validation_images: __validate_lambda(validation_prompts, validation_images)
