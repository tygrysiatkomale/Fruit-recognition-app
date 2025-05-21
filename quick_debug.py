# w osobnym pliku quick_debug.py
import onnxruntime as ort, numpy as np
from PIL import Image
from your_torch_model import model as torch_model        # tylko jeśli masz jeszcze PyTorch
torch_model.eval()

def preprocess_pytorch(path):
    import torchvision.transforms as T
    tf = T.Compose([
        T.Resize((100,100)),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])
    return tf(Image.open(path)).unsqueeze(0)

def preprocess_onnx(path):
    img = Image.open(path).convert("RGB").resize((100,100))
    x = np.asarray(img,dtype=np.float32)/255.0
    x = (x-0.5)/0.5
    return x.transpose(2,0,1)[None,...]

pth_pred = torch_model(preprocess_pytorch("apple.jpg")).argmax(1).item()
sess = ort.InferenceSession("simple_fruit_cnn_1.onnx")
onnx_pred = sess.run(None, {"input": preprocess_onnx("apple.jpg")})[0].argmax(1).item()
print("PyTorch:", pth_pred, "ONNX:", onnx_pred)
