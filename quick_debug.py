# quick_debug.py  –  ONNX only
import onnxruntime as ort
import numpy as np
from PIL import Image

IMG_PATH   = "apple100.jpg"
IMG_SIZE   = (100, 100)          # tak jak w app.py
MODEL_PATH = "simple_fruit_cnn_1.onnx"

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    x   = np.asarray(img, dtype=np.float32) / 255.0
    x   = (x - 0.5) / 0.5          # ta sama normalizacja co w app.py
    return x.transpose(2,0,1)[None,...]

sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
probs = sess.run(None, {"input": preprocess(IMG_PATH)})[0][0]
print("Argmax   :", probs.argmax())
print("Top-3 ids:", probs.argsort()[-3:][::-1])
