import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

st.title("🎨 AI Image Generator with Stable Diffusion")

prompt = st.text_input("Enter your image prompt:", "A cinematic view of a futuristic city at sunset")

if st.button("Generate Image"):
    with st.spinner("Loading model and generating image..."):
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float32,
            use_auth_token=True  # only needed if private model
        )
        pipe.to("cpu")

        image = pipe(prompt).images[0]
        st.image(image, caption="Generated Image", use_column_width=True)
        image.save("generated_image.jpg")
