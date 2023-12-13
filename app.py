import streamlit as st
import torch
from diffusers import LCMScheduler, AutoPipelineForText2Image
from PIL import Image
import io
from st_keyup import st_keyup

@st.cache_resource
def load_model():
    model_id = "segmind/Segmind-Vega"
    adapter_id = "segmind/Segmind-VegaRT"
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.load_lora_weights(adapter_id)
    pipe.fuse_lora()
    return pipe

pipe = load_model()

def generate_image(pipe, prompt, num_inference_steps, guidance_scale):
    image = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    return image

def main():
    st.title("Real-Time Text to Image Generation with Keyup")
    
    debounce = st.sidebar.slider("Debounce", 0, 1000, 500)
    prompt = st_keyup("Enter your prompt:", debounce=debounce, key="prompt_input")

    # Parameters for image generation in the sidebar
    num_inference_steps = st.sidebar.slider("Number of Inference Steps", 2, 8, 4)
    guidance_scale = st.sidebar.slider("Guidance Scale", 0.0, 2.0, 0.0)

    # Initialize an empty buffer for the image
    img_buffer = None

    # Displaying the image
    if prompt:  # Check if prompt is not empty
        with st.spinner("Generating Image..."):
            image = generate_image(pipe, prompt, num_inference_steps, guidance_scale)
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="JPEG")
            img_buffer.seek(0)
            st.image(img_buffer, use_column_width=True)

    # Save Image button in the sidebar
    if img_buffer is not None:
        st.sidebar.download_button(
            label="Save Current Image",
            data=img_buffer,
            file_name="outputs/generated_image.jpeg",
            mime="image/jpeg"
        )

if __name__ == "__main__":
    main()
