import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import onnxruntime as ort
from pathlib import Path

MODEL_PATH  = "simple_fruit_cnn_1.onnx"  
LABELS_PATH = "labels.txt"               
IMG_SIZE    = (100, 100)                 

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp_name  = sess.get_inputs()[0].name
    out_name  = sess.get_outputs()[0].name
    return sess, inp_name, out_name

session, INPUT_NAME, OUTPUT_NAME = load_model(MODEL_PATH)

LABELS = [ln.strip() for ln in Path(LABELS_PATH).read_text(encoding="utf-8").splitlines()]

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = ImageOps.fit(img, (140, 140), centering=(0.5, 0.5))
    img = ImageEnhance.Brightness(img).enhance(1.10)
    img = ImageEnhance.Contrast(img).enhance(1.10)
    img = img.resize((100, 100), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0   
    x = (x - 0.5) / 0.5                
    return x.transpose(2, 0, 1)[None, ...]     


def softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max())
    return e / e.sum()

def predict(tensor: np.ndarray):
    logits = session.run([OUTPUT_NAME], {INPUT_NAME: tensor})[0][0]
    probs  = softmax(logits)
    idx    = int(probs.argmax())
    return idx, float(probs[idx])

st.set_page_config(page_title="Klasyfikacja", page_icon="🍎")
st.title("🍎 Klasyfikacja obrazów z modelem ONNX")

uploaded = st.file_uploader("Wgraj obraz JPG / PNG", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Podgląd", use_container_width=True)

    with st.spinner("Analizuję…"):
        idx, conf = predict(preprocess(img))
    label = LABELS[idx] if idx < len(LABELS) else f"class {idx}"
    st.success(f"**Prediction:** {label}  ({conf:.2%})")
