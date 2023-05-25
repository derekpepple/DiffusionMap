from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a top down image texture of a black and white checkerboard ceramic tile"
images = pipe(prompt).images
print(len(images))
image = images[0]
    
image.save("generated.png")