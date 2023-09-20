# Libraries 
import streamlit as st
from transformers import AutoProcessor, BlipForConditionalGeneration, AutoTokenizer
import openai
from itertools import cycle
from tqdm import tqdm
from PIL import Image
import torch
import os
from dotenv import load_dotenv
from gtts import gTTS
from googletrans import Translator
from urllib.request import urlopen
from io import BytesIO


descriptions = []

# Object creation model, tokenizer and processor from HuggingFace
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")

# Setting for the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

load_dotenv()
# Getting the key from env
openai.api_key = os.environ.get('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') ## you Openai key
openai_model = "text-davinci-002" # OpenAI model 

def translate_captions(captions, target_languages):
    translator = Translator()
    translated_captions = {}
    
    for language in target_languages:
        translated_captions[language] = []
        for caption in captions:
            translation = translator.translate(caption, dest=language)
            translated_captions[language].append(translation.text)
    
    return translated_captions

def prediction(img_list):
    
    max_length = 30
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    
    img = []
    
    for image in tqdm(img_list):
        
        i_image = Image.open(image) # Storing of Image
        st.image(i_image,width=200) # Display of Image

        if i_image.mode != "RGB": # Check if the image is in RGB mode
            i_image = i_image.convert(mode="RGB")

        img.append(i_image) # Add image to the list

    # Image data to pixel values
    pixel_val = processor(images=img, return_tensors="pt").pixel_values
    pixel_val = pixel_val.to(device)

    # Using model to generate output from the pixel values of Image
    output = model.generate(pixel_val, **gen_kwargs)

    # To convert output to text
    predict = tokenizer.batch_decode(output, skip_special_tokens=True)
    predict = [pred.strip() for pred in predict]

    return predict
        
    
def sample():
    # Sample Images in the 
    sp_images = {'Sample 1':'image\\beach.png','Sample 2':'image\\coffee.png','Sample 3':'image\\footballer.png','Sample 4':'image\\mountain.jpg'} 
    
    colms = cycle(st.columns(4)) # No of Columns 
    
    for sp in sp_images.values(): # To display the sample images
        next(colms).image(sp, width=150)
        
    for i, sp in enumerate(sp_images.values()): # loop to generate caption and hashtags for the sample images
        
        if next(colms).button("Generate",key=i): # Prediction is called only on the selected image
            
            description = prediction([sp])
            st.subheader("Description for the Image:")
            st.write(description[0])

def upload():
    # Form uploader inside tab
    with st.form("uploader"):
        # Image URL input (moved to the top)
        st.subheader("Enter Image URL")
        image_urls = st.text_area("Enter Image URL", value="", help="Enter image URLs separated by line breaks.")

        # Use HTML to center the "OR" text
        st.markdown("<h6 style='text-align:center;'>OR</h6>", unsafe_allow_html=True)

        # Image input (file upload)
        image = st.file_uploader("Upload Image", accept_multiple_files=True, type=["jpg", "png", "jpeg"])

        target_languages = st.multiselect("Select Target Languages for Translation", ["en", "fr", "es", "de", "zh-CN"])
        # Generate button
        submit = st.form_submit_button("Generate")

        if submit:
            # Process uploaded images
            if image:
                image_paths = [BytesIO(img.read()) for img in image]
            else:
                # Process image URLs
                image_urls = image_urls.split("\n")
                image_paths = [urlopen(url) for url in image_urls if url.strip()]

            description = prediction(image_paths)
            st.subheader("Description for the Image(s):")
            for i, caption in enumerate(description):
                st.write(f"Original Caption ({i + 1}): {caption}")
            if target_languages:
                st.subheader("Translated Captions:")
                translated_captions = translate_captions(description, target_languages)
                for language, captions in translated_captions.items():
                    st.write(f"Language: {language}")
                    for i, caption in enumerate(captions):
                        st.write(f"Caption ({i + 1}): {caption}")


def main():
    # title on the tab
    st.set_page_config(page_title="Caption generation") 
    # Title of the page
    st.title("Image Captioner")
    
    # Tabs on the page 
    tab1, tab2= st.tabs(["Upload Image", "Sample"])
    
    # Selection of Tabs
    with tab1: # Sample images tab
        upload()

    with tab2: # Upload images tab
        sample()

        # Sub-title of the page
    st.subheader('By Beh Teck Sian')
    
if __name__ == '__main__': 
    main()