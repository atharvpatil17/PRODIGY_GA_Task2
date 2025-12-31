from diffusers import StableDiffusionPipeline
import torch

def generate_image(prompt, output_name="generated_image.png"):
    # Load pre-trained Stable Diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5"
    )

    # Use CPU (works in all systems)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    # Generate image
    image = pipe(prompt).images[0]

    # Save output
    image.save(output_name)
    print(f"Image saved as {output_name}")

if __name__ == "__main__":
    prompt = "A futuristic city at sunset, ultra realistic, digital art"
    generate_image(prompt)

