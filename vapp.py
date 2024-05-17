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
import spacy
from collections import Counter
import whisper
from datetime import timedelta
import ollama  # Import the Ollama client
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


st.set_page_config(layout="wide")

# Ignore specific FutureWarning from huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub.file_download')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

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

# Load Whisper model
whisper_model = whisper.load_model("large")

def generate_caption(image):
    image_tensor = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)
    gen_kwargs = {
        "max_length": 15,  # Further reduced maximum length
        "min_length": 5,   # Further reduced minimum length
        "num_beams": 15,   # Further increased number of beams for better accuracy
        "no_repeat_ngram_size": 1,
        "length_penalty": 2.0,  # Increased penalty for longer captions
        "early_stopping": True
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
    
    if result.returncode != 0:
        st.error("FFprobe command failed.")
        return None
    
    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        st.error("JSON decoding failed.")
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

def extract_nouns(caption):
    doc = nlp(caption)
    return [token.text.lower() for token in doc if token.pos_ == 'NOUN']

def remove_infrequent_nouns(caption, common_nouns, rare_nouns):
    doc = nlp(caption)
    filtered_caption = ' '.join([token.text for token in doc if token.pos_ != 'NOUN' or (token.text.lower() in common_nouns and token.text.lower() not in rare_nouns)])
    return filtered_caption

def extract_transcript(video_path):
    result = whisper_model.transcribe(video_path)
    return result["segments"]

def format_timedelta(seconds):
    return str(timedelta(seconds=seconds)).split(".")[0]

def convert_to_seconds(timestamp):
    parts = timestamp.split(":")
    parts = list(map(int, parts))
    return parts[0] * 3600 + parts[1] * 60 + parts[2] if len(parts) == 3 else parts[0] * 60 + parts[1]

def generate_feedback(transcript_segments, interlaced_text):
    # Log the interlaced text
    logging.info(f"Interlaced text:\n{interlaced_text}")

    # Prepare the improved prompt text
    prompt_text = f"""
    Review the combined transcripts (t), image captions (c), and corresponding time codes ('interlaced text') from a video recording of a single incident.
    Each caption represents a single frame of the video. Each video encapsulates one incident. Videos are usually body worn camera footage or interview room recordings.
    Some captions will be incorrect so you must look at the captions as a whole to infer the narrative of the described frames.

    Instructions:
    Infer the context and overall narrative of the incident by reviewing the captions and transcript.
    Write a detailed, cohesive, and formal report that accurately describes the specific incident recorded in the video.
    Your report will tell the narrative of the incident using the provided information.
    
    Write your entire response in the tone and style of a police officer writing an incident report for their department's records.
    Do not refer to the captions in the report--use the captions to tell the overall narrative by comparing them to the transcript.
    Do not tell the reader that you need more information--make your best estimated guess.
    
    1. Review the provided transcripts and image captions.
    2. Summarize the events of the specific incident in the video using formal language.
    3. Provide a detailed description of the actions, interactions, and context based on the captions and transcripts.
    4. To avoid false positives, only mention specific nouns or actions if they reoccur multiple times in the captions.
    5. Do not include opinions, suggestions for further investigation, or general observations.

    Here is the interlaced text:
    {interlaced_text}
    """

    # Log the full prompt text
    logging.info(f"Prompt text sent to LLaMA:\n{prompt_text}")

    # Sending the prompt to LLaMA
    response = ollama.chat(
        model='llama3',
        messages=[{'role': 'user', 'content': prompt_text}]
    )

    # Extracting the content from the response
    response_content = response['message']['content']
    logging.info(f"LLaMA response:\n{response_content}")
    return response_content


st.title('Video Captioning and Transcription App')
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

# Variable to control the number of frames per second to extract
fps_target = st.number_input("Frames per second to extract", min_value=0.1, max_value=10.0, value=1.5)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()  # Ensure the file is closed and written completely

    frames, time_interval, duration = extract_frames(tfile.name, fps_target=fps_target)

    st.write(f"Total video duration: {duration:.2f} seconds")
    expected_frames = int(duration * fps_target)
    st.write(f"Expected number of frames: {expected_frames}")
    st.write(f"Extracted {len(frames)} frames from the video.")

    initial_captions = []
    timestamps = []

    # Define columns for images
    num_columns = 3
    columns = st.columns(num_columns)

    for i, frame in enumerate(frames):
        col = columns[i % num_columns]
        with col:
            timestamp = i * time_interval
            formatted_timestamp = format_timedelta(timestamp)
            st.image(frame, caption=f'Time: {formatted_timestamp}', use_column_width=True)
            with st.spinner(f'Generating caption for frame {i+1}...'):
                caption = generate_caption(frame)
                initial_captions.append(caption)
                timestamps.append(formatted_timestamp)
                st.write(f"{formatted_timestamp} - {caption}")

    # Show initial captions
    initial_caption_text = "\n".join([f"{timestamps[i]} - {initial_captions[i]}" for i in range(len(initial_captions))])

    # Extract nouns from all captions
    all_nouns = []
    for caption in initial_captions:
        all_nouns.extend(extract_nouns(caption))

    # Determine common and rare nouns
    noun_counts = Counter(all_nouns)
    common_nouns = {noun for noun, count in noun_counts.items() if count > 1}
    rare_nouns = {noun for noun, count in noun_counts.items() if count == 1}

    # Filter captions by removing infrequent nouns
    filtered_captions = [remove_infrequent_nouns(caption, common_nouns, rare_nouns) for caption in initial_captions]

    # Update full_caption string with filtered captions
    filtered_caption_text = "\n".join([f"{timestamps[i]} - {filtered_captions[i]}" for i in range(len(filtered_captions))])

    # Extract transcript using Whisper
    transcript_segments = extract_transcript(tfile.name)

    # Combine captions and transcripts based on time
    combined_text = []
    for segment in transcript_segments:
        start_time = format_timedelta(segment['start'])
        end_time = format_timedelta(segment['end'])
        text = segment['text']
        combined_text.append(f"{start_time} - {end_time}: {text}")

    combined_text = "\n".join(combined_text)

    # Interlace transcripts and captions
    interlaced_text = []
    i = j = 0
    while i < len(transcript_segments) and j < len(timestamps):
        ts_seconds = convert_to_seconds(timestamps[j])
        segment = transcript_segments[i]
        segment_start_seconds = segment['start']

        if ts_seconds <= segment_start_seconds:
            interlaced_text.append(f"c: {timestamps[j]} - {filtered_captions[j]}")
            j += 1
        else:
            interlaced_text.append(f"t: {format_timedelta(segment_start_seconds)} - {segment['text']}")
            i += 1

    # Append any remaining captions or transcript segments
    while j < len(timestamps):
        interlaced_text.append(f"c: {timestamps[j]} - {filtered_captions[j]}")
        j += 1
    while i < len(transcript_segments):
        segment = transcript_segments[i]
        interlaced_text.append(f"t: {format_timedelta(segment['start'])} - {segment['text']}")
        i += 1

    interlaced_text = "\n".join(interlaced_text)


    # Define fourth and fifth columns for additional information
    col4, col5 = st.columns(2)

    with col4:
        st.markdown("## Initial Captions")
        st.text_area("Initial Captions", initial_caption_text, height=200)

        st.markdown("## Filtered Captions")
        st.text_area("Filtered Captions", filtered_caption_text, height=200)

        st.markdown("## Transcript")
        st.text_area("Transcript", combined_text, height=300)

        st.markdown("## Interlaced Captions and Transcript")
        st.text_area("Interlaced", interlaced_text, height=300)

    # Generate feedback using the combined interlaced text and transcripts
    feedback = generate_feedback(transcript_segments, interlaced_text)

    with col5:
        st.markdown("## LLaMA3 Output")
        st.text_area("LLaMA3 Output", feedback, height=800)
