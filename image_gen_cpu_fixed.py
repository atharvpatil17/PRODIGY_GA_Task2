from diffusers import StableDiffusionPipeline
import torch
import os

# Load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)

# Run on CPU
pipe = pipe.to("cpu")

# Text prompt
prompt = "A futuristic city at sunset, digital art"

# Generate image
image = pipe(prompt).images[0]

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# Save image
image.save("outputs/generated_image.png")

print("✅ Image saved successfully in outputs/generated_image.png")



