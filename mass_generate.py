from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler, UniPCMultistepScheduler
import torch
from PIL import Image, ImageDraw, ImageFont
import os
from copy import deepcopy
from transformers import CLIPTextModel, CLIPTokenizer
import requests
from huggingface_hub import notebook_login, hf_hub_url
import zipfile
import json
import argparse
import boto3
from deepface import DeepFace
import glob
import shutil
import tempfile
import time
import warnings
import sentry_sdk
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from compel import Compel
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from diffusers.models.attention_processor import AttnProcessor2_0

# options = {
#     "MODEL_ID": os.getenv("MODEL_ID", default=args.model_id),
#     "PRECISION": str(os.getenv("PRECISION", default=args.precision)),
#     "BETA_START": float(os.getenv("BETA_START", default=args.beta_start)),
#     "BETA_END": float(os.getenv("BETA_END", default=args.beta_end)),
#     "NUM_TRAIN_TIMESTEPS": int(
#         os.getenv("NUM_TRAIN_TIMESTEPS", default=args.num_train_timesteps)
#     ),
# }

def parse_args(input_args=None):
    parser = argparse.ArgumentParser()    
    parser.add_argument("--generation_id", type=str, default=None, required=True)
    parser.add_argument("--user_id", type=str, default=None, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True)
    parser.add_argument("--templates_path", type=str, default=None, required=True)
    parser.add_argument("--embeddings_path", type=str, default=None, required=True)
    parser.add_argument("--gen_count", type=int, default=1, required=False)
    parser.add_argument("--output_dir", type=str, default=None, required=True)
    parser.add_argument("--run_facial_filter", default=False, action="store_true", required=False)
    parser.add_argument("--facial_filter_basis_img", type=str, default=None, required=False)
    parser.add_argument("--dev", action="store_true")


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.gen_count > 15:
        warnings.warn("gen_count is beyond hard limit of 15, defaulting to 15")
        args.gencount = 15

    if (args.run_facial_filter == True) and (args.facial_filter_basis_img == None):
        warnings.warn("need to have a specified basis image path for facial filter to work")

    return args


class TrainingException(Exception):
    """
    Stores additional information about the error for updating Firebase

    message (str)
    error (dict): { "code": "...", "message": "..." }
    """
    def __init__(self, message, error):
        super().__init__(message)
        self.error = error


def log_prometheus_metrics(templates_path, gen_count, duration):

    # templates.json will be ['template_1', 'template_2', etc.]
    with open(templates_path, 'r') as f:
        templates = json.load(f)

    template_count = len(templates)

    total_images = gen_count * template_count

    seconds_per_image = duration / total_images

    os.makedirs("/tmp", exist_ok=True)
    duration_fd = "/tmp/inference_duration.txt"
    seconds_per_image_fd = "/tmp/inference_spi.txt"

    with open(duration_fd, 'w+', encoding="utf-8") as f:
        f.write(str(duration))

    with open(seconds_per_image_fd, 'w+', encoding="utf-8") as f:
        f.write(str(seconds_per_image))


    return


class inference():

    def __init__(self, model_path, embeddings_path, gen_count, templates_path, output_dir, run_facial_filter, facial_filter_basis_img):
        self.model_path = model_path
        self.gen_count = gen_count
        self.templates_path = templates_path
        self.output_dir = output_dir

        self.run_facial_filter = run_facial_filter
        self.facial_filter_basis_img = facial_filter_basis_img

        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            model_path,
            subfolder="scheduler")

        # self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
        #     model_path, 
        #     subfolder="scheduler", repo_type="directory")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_path,
            subfolder="tokenizer",
            use_auth_token=False,
            repo_type="directory"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_path,
            subfolder="text_encoder",
            use_auth_token=False,
        )
        try:
            self.embeddings_path = embeddings_path
            if self.embeddings_path is not None:
                # Loading all of the .bin embedding files
                for filename in os.listdir(self.embeddings_path):
                    if filename.endswith(".bin"):
                        print(filename)
                        self.load_learned_embeds_in_clip(os.path.join(self.embeddings_path, filename),
                                                         self.text_encoder, 
                                                         self.tokenizer)
                print("Loaded embeddings: " , self.tokenizer.added_tokens_encoder)
        except:
            print("Loaded embeddings: " , self.tokenizer.added_tokens_encoder)

        # Initializing pipeline
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.model_path,
                                                      torch_dtype=torch.float16,
                                                      safety_checker=None,
                                                      scheduler=self.scheduler,
                                                      text_encoder=self.text_encoder,
                                                      tokenizer=self.tokenizer,
                                                      revision="fp16",
                                                      repo_type="directory").to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()
        # self.pipe.unet = torch.compile(self.pipe.unet)
        # self.pipe.unet.set_attn_processor(AttnProcessor2_0())
        
        self.vocab_size = self.tokenizer.vocab_size
        self.compel_proc = Compel(tokenizer=self.pipe.tokenizer, text_encoder=self.pipe.text_encoder, truncate_long_prompts=False)


    def __call__(self):
        # Open templates.json file for reading
        with open(self.templates_path, 'r') as file:
            data = json.load(file)

        for template in data:
            template_name = template['result_name']
            prompt = template['prompt']
            neg_prompt = template['neg_prompt']
            init_image = Image.open(template['init_image_path'])
            strength = float(template['strength'])
            output_path = os.path.join(self.output_dir, template_name)
            os.makedirs(output_path, exist_ok=True)

            self.generate(prompt=prompt,
                          neg_prompt=neg_prompt,
                          init_image=init_image,
                          strength=strength,
                          template_name=template_name,
                          output_path=output_path)


    def generate(self, prompt, neg_prompt, init_image, strength, template_name, output_path):
        device = "cuda"
        g = torch.Generator(device=device)
        g.manual_seed(69)
        with tempfile.TemporaryDirectory() as tmpdir:
            with torch.autocast(device):
                for i in range(self.gen_count):
                    conditioning = self.compel_proc.build_conditioning_tensor(prompt)
                    negative_conditioning = self.compel_proc.build_conditioning_tensor(neg_prompt)
                    [conditioning, negative_conditioning] = self.compel_proc.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])
                    image = self.pipe(prompt_embeds=conditioning,
                                     negative_prompt_embeds=negative_conditioning,
                                     image=init_image,
                                     strength=strength,
                                     guidance_scale=9,
                                     num_inference_steps=40,
                                     generator=g
                                     ).images[0]
                    image.save(f"{tmpdir}/{template_name}_{i}.jpg")

                if self.run_facial_filter == True:
                    self.facial_filter(self.facial_filter_basis_img, tmpdir)
                
                file_list = os.listdir(tmpdir)
                print("File list:", file_list)
                if file_list is not None:
                    for f in file_list:
                        print("Current file:", f)
                        if f is not None:
                            shutil.copy(os.path.join(tmpdir, f), output_path)
                        else:
                            print("Skipping None file")
                else:
                    print("No files found in the directory")


    def load_learned_embeds_in_clip(self, learned_embeds_path, text_encoder, tokenizer, token=None):
        loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

        # separate token and the embeds
        trained_token = list(loaded_learned_embeds.keys())[0]
        print("trained_token: ", trained_token, "\n")
        embeds = loaded_learned_embeds[trained_token]

        # cast to dtype of text_encoder
        dtype = text_encoder.get_input_embeddings().weight.dtype
        embeds.to(dtype)

        # add the token in tokenizer
        token = token if token is not None else trained_token
        num_added_tokens = tokenizer.add_tokens(token)
        if num_added_tokens == 0:
            raise ValueError(f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")

        # resize the token embeddings
        text_encoder.resize_token_embeddings(len(tokenizer))

        # get the id for the token and assign the embeds
        token_id = tokenizer.convert_tokens_to_ids(token)
        text_encoder.get_input_embeddings().weight.data[token_id] = embeds


    def facial_filter(self, basis_image_path, path_to_images):
        # path = r'outputs/txt2img-samples/final_results/*.png'
        png_files = glob.glob(fr"{path_to_images}/*.png")
        jpg_files = glob.glob(fr"{path_to_images}/*.jpg")
        img_paths = png_files + jpg_files

        # If the path to the basis image is a directory, just take the first image file in there.
        # Todo: use method to pick best pic from bunch and use that as the basis image.
        if os.path.isdir(basis_image_path):
            basis_png_files = glob.glob(fr"{path_to_images}/*.png")
            basis_jpg_files = glob.glob(fr"{path_to_images}/*.jpg")
            basis_img_path = basis_png_files + basis_jpg_files
            basis_img_path = basis_img_path[0]

        likeness_distances= []
        for img_path in img_paths:
            try:
                results = DeepFace.verify(img1_path = basis_image_path, img2_path = img_path)
                likeness_distances.append(results.get("distance"))
            except ValueError as e:
                likeness_distances.append(1.0)
                print("Error, could not find a face")
            except:
                likeness_distances.append(1.0)
                print("Error")
        print(likeness_distances)

        # Keep image with best(smallest) likeness distance metric, remove all others
        best_img_path = img_paths[likeness_distances.index(min(likeness_distances))]
        for img_path in img_paths:
            if img_path is not best_img_path:
                os.system(fr"rm '{img_path}'")




if __name__ == "__main__":
    args = parse_args()
    
    sentry_sdk.init(
    dsn="https://696c9eea81594c9dae532f577c09e1da@o4505322473324544.ingest.sentry.io/4505322474307584",

    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    # We recommend adjusting this value in production.
    traces_sample_rate=1.0,
    )
    sentry_sdk.set_context("workflow", {"Stage": "inference"})

    # Firebase realtime database
    if args.dev:
        cred = credentials.Certificate("service_key.json")
    else:
        cred = credentials.Certificate("/var/secrets/google/serviceAccountKey")
        

    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://dreamify-42741-default-rtdb.firebaseio.com/",
        "databaseAuthVariableOverride": {
            "uid": args.user_id
        }
    })

    # We handle the special case of trial generations and update the 
    # corresponding database entry.
    if args.generation_id == "trial":
        ref_path = f"/users/{args.user_id}/training_workflow"
    else:
        ref_path = f"/users/{args.user_id}/inference_workflow"


    workflow_ref = db.reference(ref_path)
    workflow_ref.update({
        "status": "inference"
    })


    try:
        inference = inference(
            model_path = args.pretrained_model_name_or_path,
            embeddings_path = args.embeddings_path,
            gen_count = args.gen_count,
            templates_path = args.templates_path,
            output_dir = args.output_dir,
            run_facial_filter = args.run_facial_filter,
            facial_filter_basis_img = args.facial_filter_basis_img
        )
        start = time.time()
        inference()
        end = time.time()

        duration = end - start
        log_prometheus_metrics(args.templates_path, args.gen_count, duration) # writing the metrics data to .txt files
        print(f"Time elapsed: {end - start} seconds")

    # We may want more granular error handling in the future, when we have a better
    except TrainingException as e:
        sentry_sdk.capture_exception(e)
        print(e)

        workflow_ref.update({
            "error": e.error,
            "status": "failed"
        })

    # Here we capture all unanticipated exceptions and log them with an unknown status.
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(e)

        workflow_ref.update({
            "error": {"code": "unknown", "message": "An unknown error occurred."},
            "status": "failed"
        })
