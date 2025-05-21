# app.py  ── uproszczona inferencja ONNX + Streamlit
import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
from pathlib import Path

MODEL_PATH  = "simple_fruit_cnn_1.onnx"   # ← jeśli użyjesz innej nazwy, zmień tutaj
LABELS_PATH = "labels.txt"                # ← masz ten plik obok skryptu
IMG_SIZE    = (100, 100)                  # Fruit-360 trenowany na 100×100

# -------------------------------------------------- 1. załaduj model
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp_name  = sess.get_inputs()[0].name
    out_name  = sess.get_outputs()[0].name
    return sess, inp_name, out_name

session, INPUT_NAME, OUTPUT_NAME = load_model(MODEL_PATH)

# -------------------------------------------------- 2. etykiety
LABELS = [ln.strip() for ln in Path(LABELS_PATH).read_text(encoding="utf-8").splitlines()]

# -------------------------------------------------- 3. preprocessing = to samo co w Torch transforms
def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    x   = np.asarray(img, dtype=np.float32) / 255.0      # H,W,C in [0,1]
    x   = (x - 0.5) / 0.5                                # -> [-1,1]   (Normalize 0.5/0.5)
    return x.transpose(2, 0, 1)[None, ...]               # 1,C,H,W

def softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max())
    return e / e.sum()

def predict(tensor: np.ndarray):
    logits = session.run([OUTPUT_NAME], {INPUT_NAME: tensor})[0][0]
    probs  = softmax(logits)
    idx    = int(probs.argmax())
    return idx, float(probs[idx])

# -------------------------------------------------- 4. Streamlit UI
st.set_page_config(page_title="Fruit-360 classifier", page_icon="🍎")
st.title("🍎 Fruit-360 classifier")

uploaded = st.file_uploader("Wgraj obraz JPG / PNG", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Podgląd", use_container_width=True)

    with st.spinner("Analizuję…"):
        idx, conf = predict(preprocess(img))
    label = LABELS[idx] if idx < len(LABELS) else f"class {idx}"
    st.success(f"**Prediction:** {label}  ({conf:.2%})")
