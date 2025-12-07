# app.py
import io
from PIL import Image
import streamlit as st
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

st.set_page_config(page_title="Fast Style Transfer (MVP)", layout="wide")
st.title("Fast Style Transfer — Upload content & style images")

@st.cache_resource
def load_model():
    # TF-Hub model handle for arbitrary-image-stylization
    model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")
    return model

def load_image_bytes(uploaded_file, max_dim=1024):
    img = Image.open(uploaded_file).convert("RGB")
    # keep reasonable size for fast inference
    img.thumbnail((max_dim, max_dim))
    return img

def tf_image_from_pil(pil_img):
    arr = np.array(pil_img).astype(np.float32)[np.newaxis, ...] / 255.0
    return tf.convert_to_tensor(arr)

model = load_model()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Content image")
    content_file = st.file_uploader("Upload a content image", type=["png","jpg","jpeg"], key="content")
    if content_file:
        content_pil = load_image_bytes(content_file)
        st.image(content_pil, caption="Content", use_column_width=True)

with col2:
    st.subheader("Style image")
    style_file = st.file_uploader("Upload a style image (painting, sketch, etc.)", type=["png","jpg","jpeg"], key="style")
    if style_file:
        style_pil = load_image_bytes(style_file, max_dim=256)  # style can be smaller
        st.image(style_pil, caption="Style", use_column_width=True)

if content_file and style_file:
    st.write("Generating stylized image — one forward pass (fast).")
    content_tf = tf_image_from_pil(content_pil)
    style_tf = tf_image_from_pil(style_pil)

    # model returns a batch of 1
    outputs = model(tf.constant(content_tf), tf.constant(style_tf))
    stylized = outputs[0][0]  # stylized image tensor (H,W,3)

    # convert to uint8 PIL
    stylized_np = np.clip(stylized.numpy() * 255.0, 0, 255).astype(np.uint8)
    stylized_pil = Image.fromarray(stylized_np)

    st.subheader("Stylized result")
    st.image(stylized_pil, use_column_width=True)

    # allow download
    buf = io.BytesIO()
    stylized_pil.save(buf, format="PNG")
    st.download_button("Download result (PNG)", data=buf.getvalue(), file_name="stylized.png", mime="image/png")
