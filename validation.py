import jax
from logging import Image, wandb_log_validation
from flax.training.common_utils import shard

from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel,
)

from transformers import AutoTokenizer, FlaxT5Model


def validate():

    # pipeline_params = pipeline_params.copy()
    # pipeline_params["controlnet"] = controlnet_params

    vae = FlaxAutoencoderKL.from_pretrained(
        "flax/stable-diffusion-2-1", subfolder="vae"
    )
    unet = FlaxUNet2DConditionModel.from_pretrained("character-aware-diffusion/charred")

    scheduler = FlaxDDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")
    lm = FlaxT5Model.from_pretrained("google/byt5-base")

    pipeline = FlaxStableDiffusionPipeline(
        text_encoder=lm.encode,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
    )

    num_samples = jax.device_count()
    prng_seed = jax.random.split(rng, jax.device_count())

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []

    for validation_prompt, validation_image in zip(
        validation_prompts, validation_images
    ):
        prompts = num_samples * [validation_prompt]
        prompt_ids = pipeline.prepare_text_inputs(prompts)
        prompt_ids = shard(prompt_ids)

        validation_image = Image.open(validation_image).convert("RGB")
        processed_image = pipeline.prepare_image_inputs(
            num_samples * [validation_image]
        )
        processed_image = shard(processed_image)
        images = pipeline(
            prompt_ids=prompt_ids,
            image=processed_image,
            params=pipeline_params,
            prng_seed=prng_seed,
            num_inference_steps=50,
            jit=True,
        ).images

        images = images.reshape(
            (images.shape[0] * images.shape[1],) + images.shape[-3:]
        )
        images = pipeline.numpy_to_pil(images)

        image_logs.append(
            {
                "validation_image": validation_image,
                "images": images,
                "validation_prompt": validation_prompt,
            }
        )

    wandb_log_validation(image_logs)

    return image_logs
