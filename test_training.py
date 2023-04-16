
from architecture import setup_model


def main():

    tokenizer, text_encoder, vae, vae_params, unet, unet_params = setup_model(
        seed=None, mixed_precision=None,
        pretrained_text_encoder_model_name_or_path="google/byt5-base", pretrained_text_encoder_model_revision=None, 
        pretrained_diffusion_model_name_or_path="stabilityai/stable-diffusion-2-1", pretrained_diffusion_model_revision=None)

if __name__ == "__main__":
    main()