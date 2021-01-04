import streamlit as st
import pathlib
from PIL import Image
from tensorflow.keras.preprocessing.image import array_to_img
from style_transform import style_transform


def list_content_images():
    content = pathlib.Path("images/content")
    return sorted(list(content.iterdir()))


def list_style_images():
    content = pathlib.Path("images/styles")
    return sorted(list(content.iterdir()))


content_images = list_content_images()
style_images = list_style_images()

st.title('Style transfer')

def display_image(img_path, caption=None):
    content_img = Image.open(img_path)
    return st.image(content_img, caption=None, use_column_width=True)

@st.cache
def cached_style(content_image, style_image):
    return array_to_img(style_transform(content_image, style_image))

col1, col2 = st.beta_columns(2)
with col1:
    content_image = st.selectbox('Content image', content_images)
    display_image(content_image, caption="Content image")
with col2:
    style_image = st.selectbox('Style image', style_images)
    display_image(style_image, caption="Style image")
st.title("Styled image")
st.image(cached_style(content_image, style_image), caption=None)
