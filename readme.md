# Video Captioning and Transcription App

## Overview

This Streamlit app processes uploaded video files to extract frames, generate captions, and transcribe audio. It utilizes machine learning models to provide a detailed, cohesive report on video incidents by combining image captions and audio transcripts.

## Features

- Upload video files in MP4, MOV, or AVI formats.
- Extract frames from the video at a user-defined rate (frames per second).
- Generate image captions for each extracted frame using a VisionEncoderDecoderModel.
- Transcribe audio from the video using Whisper.
- Interlace captions with transcripts to form a comprehensive narrative.
- Provide feedback on the incident based on the combined captions and transcripts using the LLaMA3 model.

## Requirements

- Python 3.7+
- Streamlit
- Pillow
- Torch
- Transformers
- OpenCV
- Whisper
- SpaCy
- Ollama


