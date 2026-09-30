import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from weed_info import WEED_INFO

st.set_page_config(page_title="Weed Identifier", page_icon="🌿", layout="centered")

IMG = 224
NOT_WEED = "Not a weed"
CLASS_NAMES = json.load(open("class_names.json"))


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("weed_model.keras")


def predict(img: Image.Image):
    arr = np.asarray(img.convert("RGB").resize((IMG, IMG)), dtype=np.float32)[None, ...]
    return load_model().predict(arr, verbose=0)[0]


st.title("🌿 Weed Identifier")
st.write(f"Upload or take a photo of a plant. The model recognises **{len(WEED_INFO)} weed species** "
         "and tells you its scientific name, family and category.")

with st.expander("Which weeds can it identify?"):
    st.write(", ".join(sorted(WEED_INFO)))

tab_upload, tab_camera = st.tabs(["Upload photo", "Use camera"])
with tab_upload:
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
with tab_camera:
    captured = st.camera_input("Take a photo")

source = uploaded or captured
if source is None:
    st.info("Add a photo to get a result. A close shot of one plant works best.")
    st.stop()

image = Image.open(source)
st.image(image, use_container_width=True)

with st.spinner("Identifying..."):
    probs = predict(image)

top = int(np.argmax(probs))
label, conf = CLASS_NAMES[top], float(probs[top])

if conf < 0.5:
    st.warning(f"Not sure ({conf:.0%} confidence). Try a clearer, closer photo of a single plant.")
elif label == NOT_WEED:
    st.success(f"✅ Not a weed ({conf:.0%} confidence)")
    st.caption(f"It doesn't match any of the {len(WEED_INFO)} known weeds. "
               "It could still be a weed species the model wasn't trained on.")
else:
    info = WEED_INFO.get(label)
    st.error(f"⚠️ Weed detected: **{label}** ({conf:.0%} confidence)")
    if info:
        st.markdown(f"""
| | |
|---|---|
| **Common name** | {label} |
| **Scientific name** | *{info['scientific']}* |
| **Family** | {info['family']} |
| **Order** | {info['order']} |
| **Category** | {info['category']} |
| **Native to** | {info['native']} |
""")
        st.write(info["notes"])

with st.expander("Top 3 predictions"):
    for i in np.argsort(probs)[::-1][:3]:
        st.write(f"{CLASS_NAMES[i]}: {probs[i]:.1%}")
        st.progress(float(probs[i]))

st.caption("Trained on DeepWeeds, CottonWeedID15, Plant Seedlings (Aarhus) and Early Crop Weed datasets. "
           "For guidance only — confirm with an agronomist before taking control measures.")
