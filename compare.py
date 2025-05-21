import torch, onnxruntime as ort, numpy as np
from PIL import Image
from model_arch import SimpleCNN

# ---------- pliki ----------
IMG_PATH  = "apple100.jpg"            # dowolny testowy obraz z oryginalnego zbioru
PTH_FILE  = "simple_fruit_cnn_1.pth"
ONNX_FILE = "simple_fruit_cnn_1.onnx"
IMG_SIZE  = 100                       # Fruit-360

# ---------- 1. PyTorch ----------
n_classes = len(open("labels.txt", encoding="utf-8").read().splitlines())
net = SimpleCNN(n_classes, IMG_SIZE)
state_dict = torch.load(PTH_FILE, map_location="cpu")      # OrderedDict
net.load_state_dict(state_dict)
net.eval()

def torch_pre(img_path):
    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x   = np.asarray(img, dtype=np.float32) / 255.0
    x   = (x - 0.5) / 0.5                          # identyczna normalizacja!
    x   = torch.from_numpy(x.transpose(2, 0, 1))[None]  # 1,C,H,W
    return x

torch_idx = net(torch_pre(IMG_PATH)).argmax(1).item()

# ---------- 2. ONNX ----------
sess = ort.InferenceSession(ONNX_FILE, providers=["CPUExecutionProvider"])

def onnx_pre(img_path):
    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x   = np.asarray(img, dtype=np.float32) / 255.0
    x   = (x - 0.5) / 0.5
    return x.transpose(2, 0, 1)[None]              # 1,C,H,W

onnx_idx = sess.run(None, {"input": onnx_pre(IMG_PATH)})[0].argmax(1).item()

print("PyTorch index :", torch_idx)
print("ONNX   index :", onnx_idx)
