# app.py – lekka wersja
import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
from pathlib import Path

MODEL_PATH  = "simple_fruit_cnn_1.onnx"   # nazwa Twojego pliku ONNX
LABELS_PATH = "labels.txt"                # już masz ten plik!

# 1️⃣ model + parametry wejścia
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp  = sess.get_inputs()[0]
    out  = sess.get_outputs()[0]
    _, _, h, w = inp.shape                  # h, w mogą być None (dynamiczne H×W)
    target = (w, h) if w and h else None
    return sess, inp.name, out.name, target

session, INPUT_NAME, OUTPUT_NAME, TARGET_SIZE = load_model(MODEL_PATH)

# 2️⃣ etykiety – tylko z pliku
LABELS = [ln.strip() for ln in Path(LABELS_PATH).read_text(encoding="utf-8").splitlines()]

# 3️⃣ preprocessing → tensor (1, C, H, W) 0-1
def preprocess(img: Image.Image) -> np.ndarray:
    if TARGET_SIZE:
        img = img.resize(TARGET_SIZE, Image.BILINEAR)
    x = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0      # H,W,C
    return x.transpose(2, 0, 1)[None, ...]                            # 1,C,H,W

# 4️⃣ softmax i predykcja
def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()

def predict(tensor: np.ndarray):
    logits = session.run([OUTPUT_NAME], {INPUT_NAME: tensor})[0][0]   # (n_classes,)
    probs  = softmax(logits)
    idx    = int(np.argmax(probs))
    conf   = float(probs[idx])
    return idx, conf

# 5️⃣ interfejs Streamlit
st.set_page_config(page_title="Fruit-360 classifier", page_icon="🍓", layout="centered")
st.title("🍓 Fruit-360 classifier")

if TARGET_SIZE:
    st.caption(f"Model input size : **{TARGET_SIZE[1]} × {TARGET_SIZE[0]} px**")

uploaded = st.file_uploader("Upload a JPG/PNG", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_container_width=True)

    with st.spinner("Running inference…"):
        idx, conf = predict(preprocess(img))

    label = LABELS[idx] if idx < len(LABELS) else f"class {idx}"
    st.success(f"**Prediction:** {label} ({conf:.2%})")
