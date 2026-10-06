from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler
import torch
from PIL import Image, ImageDraw, ImageFont
import os
from copy import deepcopy
from transformers import CLIPTextModel, CLIPTokenizer
import requests
from huggingface_hub import notebook_login, hf_hub_url
import zipfile

model_id = "models/sd-v1-5_vae-pruned"
gen_count = 5
seed = 2

# prompt = "demoura in a boxing ring, light cascading down, professional photo, ufc fight, professional demoura fighter, 125mm long shot, demoura demoura demoura demoura"
# prompt = "demoura highly detailed cinematic extremely colored headshot portrait photography photo of demoura person (((extremely demoura))) with rough detailed skin looking straight at me, wearing a turtleneck, deep gaze, (beautiful (round) ((((demoura eyes)))), depth, masterpiece very focused and (sharp fair eyes and skin), blur background, 125mm f1.8, ((photo of demoura)) symmetrical demoura face"
prompt = "the woman loab"
# prompt = "highly detailed cinematic extremely colored headshot portrait photography of an attractive person looking straight at me, wearing a turtleneck, deep gaze, beautiful eyes, depth, black and white, masterpiece very focused and sharp eyes and skin, blur background, 125mm f1.8"
# prompt = "wax cartoon of disfigured person with extremely destroyed malformed eyes"
# neg_prompt = "negembed negembed negembed negembed negembed negembed negembed negembed"
# neg_prompt = ""

# neg_prompt = "negembed style, smooth wax doll, side angle, extremely ugly, font, text, make up, extremely destroyed negembed eyes, hands, blurry, low resolution, animated, cartoon, lowres, cropped, worst quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned negembed face, disfigured, gross proportions, long neck, vizca, extremely black & white, negembed negembed negembed negembed negembed"

neg_prompt = ""

tokenizer = CLIPTokenizer.from_pretrained(
    model_id,
    subfolder="tokenizer",
    use_auth_token=False
)

text_encoder = CLIPTextModel.from_pretrained(
    model_id,
    subfolder="text_encoder",
    use_auth_token=False
)

def main():
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                                   custom_pipeline="lpw_stable_diffusion",
                                                   scheduler=scheduler,
                                                   text_encoder=text_encoder,
                                                   tokenizer=tokenizer,
                                                   revision="fp16",
                                                   torch_dtype=torch.float16,
                                                   safety_checker=None)
    
    pipe = pipe.to("cuda")
    vocab_size = tokenizer.vocab_size
    
    g = torch.Generator(device='cuda')
    g.manual_seed(seed)
    
    for i in range(gen_count):
        with torch.autocast("cuda"):
            image = pipe(prompt=prompt,
                         negative_prompt=neg_prompt,
                         # latents=latent_batched,
                         generator=g,
                         num_inference_steps=30, 
                         guidance_scale=7.5,
                         height=512,
                         width=512,)["images"]
            image[0].save(f"results/portrait-photo-of-nathan-EARLYSTOP-{i}.jpg")

def img2img():
    init_image = Image.open("img2img-pics/B&WGOD-816x816.jpg")
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    # scheduler = LMSDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

    # load the pipeline
    device = "cuda"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id,
                                                          custom_pipeline="lpw_stable_diffusion",
                                                          torch_dtype=torch.float16,
                                                          safety_checker=None,
                                                          scheduler=scheduler,
                                                          text_encoder=text_encoder,
                                                          tokenizer=tokenizer,
                                                          revision="fp16",
                                                         ).to(device)
    pipe.to(device)
    
    pipe.enable_attention_slicing()
    pipe.set_use_memory_efficient_attention_xformers
    # pipe.enable_sequential_cpu_offload()
    
    vocab_size = tokenizer.vocab_size
    
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    
    for i in range(gen_count):
        with torch.autocast(device):
            image = pipe(prompt=prompt,
                         negative_prompt=neg_prompt,
                         image=init_image, 
                         strength=.8, 
                         guidance_scale=7,
                         num_inference_steps=30,
                         generator=g
                        ).images
            image[0].save(f"results/nathan-816x816-{i}.jpg")

if __name__ == "__main__":
    main()
    # img2img()
