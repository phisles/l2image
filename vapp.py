import streamlit as st
import torch
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration, BlipProcessor, BlipForConditionalGeneration
from moviepy.editor import VideoFileClip
from PIL import Image
import numpy as np
import os
import re
import spacy

# Hugging Face token
HF_TOKEN = "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"

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Function to extract frames from the video
def extract_frames(video_path, num_frames=5):
    clip = VideoFileClip(video_path)
    duration = clip.duration
    times = np.linspace(0, duration, num_frames)
    frames = [clip.get_frame(t) for t in times]
    return frames

# Load the BLIP captioning model and processor
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

blip_processor, blip_model = load_blip_model()

# Function to generate captions for frames
def generate_captions(frames):
    captions = []
    for frame in frames:
        image = Image.fromarray(frame.astype('uint8'), 'RGB')
        inputs = blip_processor(images=image, return_tensors="pt")
        out = blip_model.generate(**inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        captions.append(caption)
    return captions

# Load the Whisper model and processor
@st.cache_resource
def load_whisper_model():
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    return processor, model

whisper_processor, whisper_model = load_whisper_model()

# Function to transcribe audio from video
def transcribe_audio(video_path):
    clip = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    clip.audio.write_audiofile(audio_path, codec='pcm_s16le')
    audio_input = whisper_processor(audio_path, return_tensors="pt", sampling_rate=16000)
    forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(language="en", task="transcribe")
    transcription = whisper_model.generate(**audio_input, forced_decoder_ids=forced_decoder_ids)
    transcript = whisper_processor.batch_decode(transcription, skip_special_tokens=True)[0]
    os.remove(audio_path)
    return transcript

# Function to filter captions
def filter_captions(captions):
    docs = [nlp(caption) for caption in captions]
    noun_counts = {}
    for doc in docs:
        for token in doc:
            if token.pos_ == 'NOUN':
                if token.text in noun_counts:
                    noun_counts[token.text] += 1
                else:
                    noun_counts[token.text] = 1

    common_nouns = {noun for noun, count in noun_counts.items() if count > 1}
    filtered_captions = []
    for doc in docs:
        filtered_caption = ' '.join([token.text for token in doc if token.pos_ != 'NOUN' or token.text in common_nouns])
        filtered_captions.append(filtered_caption)
    return filtered_captions

# Convert timestamp to seconds
def convert_to_seconds(timestamp):
    t = timestamp.split(":")
    return int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2])

# Interlace transcript and captions
def interlace_transcript_and_captions(transcript, captions):
    lines = transcript.split("\n")
    timestamps = [line.split(": ")[0] for line in lines]
    sentences = [line.split(": ")[1] for line in lines]
    timestamps_seconds = [convert_to_seconds(ts) for ts in timestamps]
    
    interlaced_text = ""
    for i, caption in enumerate(captions):
        caption_time = i * (timestamps_seconds[-1] / len(captions))
        for j, ts in enumerate(timestamps_seconds):
            if ts >= caption_time:
                interlaced_text += f"{timestamps[j]}: Caption: {caption}\n"
                interlaced_text += f"{timestamps[j]}: Transcript: {sentences[j]}\n"
                break
    return interlaced_text

# Load the LLaMA model
@st.cache_resource
def load_llama_pipeline():
    model_id = "meta-llama/Meta-Llama-3-8B"
    llama_pipeline = pipeline(
        "text-generation", 
        model=model_id, 
        model_kwargs={"torch_dtype": torch.bfloat16},
        use_auth_token=HF_TOKEN,
        device_map="auto"
    )
    return llama_pipeline

llama_pipeline = load_llama_pipeline()

# Function to generate summary with LLaMA
def generate_summary(interlaced_text):
    result = llama_pipeline(interlaced_text, max_length=512)
    return result[0]['generated_text']

# Streamlit app interface
st.title("Video Captioning and Transcription Summary")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    st.video(uploaded_file)

    st.header("Extracted Frames and Captions")
    frames = extract_frames(uploaded_file.name)
    captions = generate_captions(frames)
    for i, frame in enumerate(frames):
        st.image(frame, caption=captions[i])

    st.header("Filtered Captions")
    filtered_captions = filter_captions(captions)
    for caption in filtered_captions:
        st.write(caption)

    st.header("Transcription")
    transcript = transcribe_audio(uploaded_file.name)
    st.write(transcript)

    st.header("Interlaced Transcript and Captions")
    interlaced_text = interlace_transcript_and_captions(transcript, filtered_captions)
    st.write(interlaced_text)

    st.header("Generated Summary")
    summary = generate_summary(interlaced_text)
    st.write(summary)
