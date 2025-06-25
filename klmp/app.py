import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os

# Load model (CPU-friendly)
@st.cache_resource
def load_pipe():
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32
    )
    pipe.to("cpu")
    return pipe

pipe = load_pipe()

# Helper: Split script into scenes (naive split by 2 sentences)
def split_script_to_scenes(script, lines_per_scene=2):
    sentences = script.strip().split(". ")
    scenes = [" ".join(sentences[i:i + lines_per_scene]) for i in range(0, len(sentences), lines_per_scene)]
    return scenes

# App UI
st.set_page_config(page_title="Script to AI Images", layout="centered")
st.title("🎬 AI Script Visualizer")
st.write("Turn your script into vertical cinematic AI-generated images — fully automated!")

script_text = st.text_area("✏️ Paste your video/news script here", height=250)

if st.button("🚀 Generate Images"):
    if not script_text.strip():
        st.warning("Please enter a script to continue.")
    else:
        with st.spinner("Generating images for scenes..."):
            scenes = split_script_to_scenes(script_text)
            for i, scene in enumerate(scenes):
                prompt = f"A vertical cinematic photo of {scene}, realistic, hyperdetailed, 4K"
                image = pipe(prompt, height=1024, width=576).images[0]
                image_path = f"scene_{i+1}.png"
                image.save(image_path)
                st.image(image, caption=f"🖼️ Scene {i+1}", use_column_width=True)
        st.success("✅ All images generated successfully!")
