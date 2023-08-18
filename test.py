import os
import io
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import onnxruntime
import torch.nn.functional as F
import pandas as pd
import os
import sys

app = Flask(__name__)
CORS(app)
template = "class:{:<15} probability:{:.3f}"
class_path = "\\imagenet_class_index.csv"
onnx_path = "\\resnet18_imagenet.onnx"

cur_path = os.path.dirname(os.path.abspath(__file__))
class_path = cur_path + class_path
onnx_path = cur_path + onnx_path

# import onnx model
ort_session = onnxruntime.InferenceSession(onnx_path)

# import reflat
df = pd.read_csv(class_path)

# select device
device = torch.device("cpu")


def transform_image(image_bytes):
    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(256),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                                         ])
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        raise ValueError("input file does not RGB image...")
    return test_transform(image).unsqueeze(0).numpy()


def get_prediction(image_bytes):
    try:
        input_tensor = transform_image(image_bytes=image_bytes)
        # inference
        # ONNX Runtime input
        ort_inputs = {'in': input_tensor}
        # ONNX Runtime output
        pred_logits = ort_session.run(['out'], ort_inputs)[0]
        pred_logits = torch.tensor(pred_logits)
        # do softmax
        pred_softmax = F.softmax(pred_logits, dim=1)
        # sort probability
        # analyze inference result
        k = 5
        top_k = torch.topk(pred_softmax, k)
        # get class and score, has been ranked
        pred_ids, confs = top_k.indices.numpy()[0], top_k.values.numpy()[0]
        idx_to_labels = {}
        for idx, row in df.iterrows():
            idx_to_labels[row['ID']] = row['class']  # build {ID:class}

        output = list()
        for i in range(k):
            class_name = idx_to_labels[pred_ids[i]]  # get name
            confidence = confs[i] * 100  # get score
            output.append((class_name,confidence))
        text = [template.format(k, v) for k, v in output]
        # print(text)
        return_info = {"result": text}
    except Exception as e:
        return_info = {"result": [str(e)]}
    return return_info


@app.route("/predict", methods=["POST"])
@torch.no_grad()
def predict():
    image = request.files["file"]
    img_bytes = image.read()
    info = get_prediction(image_bytes=img_bytes)
    return jsonify(info)


@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("up.html")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)




