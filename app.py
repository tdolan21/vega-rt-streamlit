from diffusers import LCMScheduler, AutoPipelineForText2Image
from st_keyup import st_keyup
import streamlit as st
from PIL import Image
import datetime
import torch
import uuid
import io
import os

st.set_page_config(
    page_title="Real-Time Text to Image Generation",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def save_image(image_buffer, filename):
    # Save the image
    try:
        with open(filename, 'wb') as f:
            f.write(image_buffer.getbuffer())
        return filename
    except IOError as e:
        return f"Error saving file: {e}"

def export_all_images(image_history):
    # Create a unique subfolder in the outputs directory
    export_folder = os.path.join('outputs', uuid.uuid4().hex)
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)

    saved_files = []
    for item in image_history:
        file_path = os.path.join(export_folder, item['filename'])
        result = save_image(item['image'], file_path)
        if "Error" not in result:
            saved_files.append(result)

    return saved_files, export_folder

def main():

    st.title("Segmind Vega-RT")
    st.sidebar.image("assets/header.png")
    st.sidebar.title("Parameters")
    
    debounce = st.sidebar.slider("Debounce", 0, 1000, 500)
    prompt = st_keyup("Enter your prompt:", debounce=debounce, key="prompt_input")

    num_inference_steps = st.sidebar.slider("Number of Inference Steps", 2, 8, 4)
    guidance_scale = st.sidebar.slider("Guidance Scale", 0.0, 2.0, 0.0)

    if 'image_history' not in st.session_state:
        st.session_state.image_history = []

    # Generate image only if the prompt changes
    if prompt:
        if 'last_prompt' not in st.session_state or prompt != st.session_state.last_prompt:
            with st.spinner("Generating Image..."):
                image = generate_image(pipe, prompt, num_inference_steps, guidance_scale)
                img_buffer = io.BytesIO()
                image.save(img_buffer, format="JPEG")
                img_buffer.seek(0)

               
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = uuid.uuid4().hex
                filename = f"generated_image_{timestamp}_{unique_id}.jpeg"

                st.session_state.image_history.insert(0, {'prompt': prompt, 'image': img_buffer, 'filename': filename})

                st.session_state.image_history = st.session_state.image_history[:10]

                st.session_state.last_prompt = prompt

    # Display and save the current image
    if st.session_state.image_history:
        current_image = st.session_state.image_history[0]['image']
        current_image.seek(0)  # Reset the buffer to the start
        st.image(current_image, use_column_width=True)

        st.sidebar.divider()

        # Save to 'outputs' folder and provide download link
        if st.sidebar.button('Save Current Image to Folder'):
            filename = st.session_state.image_history[0]['filename']
            save_result = save_image(current_image, os.path.join('outputs', filename))
            if "Error" in save_result:
                st.sidebar.error(save_result)
            else:
                st.sidebar.success(f"Image saved as {save_result}")

        # Button to export all images
        if st.sidebar.button('Export All Images'):
            saved_files, export_folder = export_all_images(st.session_state.image_history)
            if saved_files:
                st.sidebar.success(f"All images exported to {export_folder}")
            else:
                st.sidebar.error("Error exporting images")
        
        st.sidebar.divider()

if __name__ == "__main__":
    main()


