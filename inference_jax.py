import jax
from flax.jax_utils import replicate
from flax.training.common_utils import shard, shard_prng_key
import jax.numpy as jnp

import torch

from diffusers import (
    FlaxAutoencoderKL,
    FlaxDPMSolverMultistepScheduler,
    FlaxUNet2DConditionModel,
)

from transformers import ByT5Tokenizer, FlaxT5ForConditionalGeneration


def get_inference_lambda(pretrained_unet_path, seed):

    tokenizer = ByT5Tokenizer()

    language_model = FlaxT5ForConditionalGeneration.from_pretrained(
        "google/byt5-base",
        dtype=jnp.float32,
    )
    text_encoder = language_model.encode
    text_encoder_params = replicate(language_model.params)
    # print(len(language_model.params["encoder"]["block"].keys()))
    # print(language_model.params["encoder"]["final_layer_norm"]["weight"].shape)
    # print(language_model.params["shared"]["embedding"].shape)

    # vae, vae_params = FlaxAutoencoderKL.from_pretrained(
    #     "flax/stable-diffusion-2-1",
    #     subfolder="vae",
    #     dtype=jnp.float32,
    # )
    # vae_params = replicate(vae_params)

    # unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
    #     pretrained_unet_path,
    #     dtype=jnp.float32,
    # )
    # unet_params = replicate(unet_params)

    # scheduler = FlaxDPMSolverMultistepScheduler.from_config(
    #     config={
    #         "_diffusers_version": "0.16.0",
    #         "beta_end": 0.012,
    #         "beta_schedule": "scaled_linear",
    #         "beta_start": 0.00085,
    #         "clip_sample": False,
    #         "num_train_timesteps": 1000,
    #         "prediction_type": "v_prediction",
    #         "set_alpha_to_one": False,
    #         "skip_prk_steps": True,
    #         "steps_offset": 1,
    #         "trained_betas": None,
    #     }
    # )

    # rng = shard_prng_key(jax.random.PRNGKey(seed))

    def __tokenize_prompt(prompt):

        return shard(
            torch.stack(
                [
                    tokenizer(
                        text=[prompt],
                        max_length=1024,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids
                ]
            )
            .to(memory_format=torch.contiguous_format)
            .float()
            .numpy()
        )

    def __predict_image(tokenized_prompt):

        # Get the text embedding
        text_encoder(
            tokenized_prompt,
            params=text_encoder_params,
            train=False,
        )[0]

        # print(text_embedding.shape)

        # output_images.append(
        #     pipeline.numpy_to_pil(
        #         pipeline(
        #             params={},
        #             prompt_ids=text_input_ids,
        #             prng_seed=rng,
        #             num_inference_steps=num_inference_steps,
        #             jit=True,
        #         ).images.reshape(
        #             (output_images.shape[0] * output_images.shape[1],)
        #             + output_images.shape[-3:]
        #         )
        #     )
        # )

    jax_pmap_predict_image = jax.pmap(__predict_image)

    return lambda prompt: jax_pmap_predict_image(__tokenize_prompt(prompt))


if __name__ == "__main__":

    # wandb_init(None)
    get_inference_lambda("character-aware-diffusion/charred", 87)("a running shoe")
    # wandb_close()
