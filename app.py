import streamlit as st
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2Tokenizer
import warnings


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
    # Assume other setup is correct and already provided
    image_tensor = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)
    gen_kwargs = {
        "max_length": 32,  # Increased max_length
        "min_length": 20,  # Encourage longer outputs
        "num_beams": 8,
        "no_repeat_ngram_size": 2  # Prevent short repetitive phrases
    }
    output_ids = model.generate(image_tensor, **gen_kwargs)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()


st.title('Image Captioning App')
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    with st.spinner('Generating caption...'):
        caption = generate_caption(image)
        st.write("Caption:", caption)

