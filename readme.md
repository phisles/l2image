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

## Installation

Install the required Python packages using pip:

```bash
pip install streamlit pillow torch transformers opencv-python-headless whisper spacy ollama
Download the SpaCy model:

bash
Copy code
python -m spacy download en_core_web_sm
Usage
Run the app with the following command:

bash
Copy code
streamlit run your_script_name.py
Open the app in your web browser.

Upload a video file.

Adjust the frames per second (fps) setting to control the number of frames extracted.

Review the generated captions, filtered captions, transcript, and interlaced text.

View the feedback generated by the LLaMA3 model.

Functions
generate_caption(image): Generates a caption for a given image frame.
get_video_info(video_path): Retrieves video metadata using FFprobe.
extract_frames(video_path, fps_target): Extracts frames from a video at the specified frame rate.
extract_nouns(caption): Extracts nouns from a given caption using spaCy.
remove_infrequent_nouns(caption, common_nouns, rare_nouns): Filters out infrequent nouns from a caption.
extract_transcript(video_path): Extracts audio transcript from a video using Whisper.
format_timedelta(seconds): Formats a duration in seconds into a string.
convert_to_seconds(timestamp): Converts a timestamp into seconds.
generate_feedback(transcript_segments, interlaced_text): Generates feedback based on transcripts and interlaced text using LLaMA3.
Logging
The app logs detailed information including interlaced text and responses from LLaMA3. The logging level is set to INFO.

License
This project is licensed under the MIT License - see the LICENSE file for details.

csharp
Copy code

You can copy and paste this snippet directly into your README file.
