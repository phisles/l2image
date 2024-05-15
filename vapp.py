import streamlit as st
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2Tokenizer
import warnings
import cv2
import tempfile
import subprocess
import json
import os
import glob

# Ignore specific FutureWarning from huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub.file_download')

# Setting up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer once using Streamlit's caching
@st.cache_resource
def load_model():
    model_id = "nlpconnect/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model, tokenizer

model, tokenizer = load_model()
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def generate_caption(image):
    image_tensor = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)
    gen_kwargs = {
        "max_length": 32,
        "min_length": 20,
        "num_beams": 8,
        "no_repeat_ngram_size": 2
    }
    output_ids = model.generate(image_tensor, **gen_kwargs)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()

def get_video_info(video_path):
    command = [
        'ffprobe', 
        '-v', 'error', 
        '-show_format', 
        '-show_streams',
        '-print_format', 'json', 
        video_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Debugging information
    st.write("FFprobe command:", " ".join(command))
    st.write("FFprobe stdout:", result.stdout)
    st.write("FFprobe stderr:", result.stderr)
    
    if result.returncode != 0:
        st.error("FFprobe command failed.")
        return None
    
    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        st.error("JSON decoding failed.")
        st.write("Error message:", e.msg)
        st.write("Error at:", e.pos)
        st.write("Raw output:", result.stdout)
        return None
    
    return info

def extract_frames(video_path, fps_target=1.0):
    info = get_video_info(video_path)
    if info is None:
        return [], 0, 0
    
    duration = float(info['format']['duration'])
    frame_interval_sec = 1 / fps_target
    
    # Create a temporary directory to store the frames
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_dir_path = tmp_dir.name
    
    # Use ffmpeg to extract frames at specified intervals
    output_pattern = os.path.join(tmp_dir_path, "frame_%04d.jpg")
    command = [
        'ffmpeg', 
        '-i', video_path,
        '-vf', f'fps={fps_target}', 
        output_pattern
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Load the extracted frames
    frames = []
    frame_paths = sorted(glob.glob(os.path.join(tmp_dir_path, "frame_*.jpg")))
    for frame_path in frame_paths:
        frame = Image.open(frame_path)
        frames.append(frame)
    
    return frames, frame_interval_sec, duration

st.title('Video Captioning App')
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

# Variable to control the number of frames per second to extract
fps_target = st.number_input("Frames per second to extract", min_value=0.1, max_value=10.0, value=1.5)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()  # Ensure the file is closed and written completely
    st.write(f"Temporary video file path: {tfile.name}")
    st.write(f"File exists: {os.path.exists(tfile.name)}")
    frames, time_interval, duration = extract_frames(tfile.name, fps_target=fps_target)
    
    st.write(f"Total video duration: {duration:.2f} seconds")
    expected_frames = int(duration * fps_target)
    st.write(f"Expected number of frames: {expected_frames}")
    st.write(f"Extracted {len(frames)} frames from the video.")
    
    captions = []
    
    # Define number of columns
    num_columns = 3
    columns = st.columns(num_columns)
    
    for i, frame in enumerate(frames):
        col = columns[i % num_columns]
        with col:
            timestamp = i * time_interval
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            st.image(frame, caption=f'Time: {minutes:02}:{seconds:02}', use_column_width=True)
            with st.spinner(f'Generating caption for frame {i+1}...'):
                caption = generate_caption(frame)
                st.write(f"{minutes:02}:{seconds:02} - {caption}")
                captions.append(caption)
    
    # Optional: Summarize the captions using an LLM (not implemented in this code)
    # summary = summarize_captions(captions)
    # st.write("Video Summary:", summary)