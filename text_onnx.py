# import package
import onnxruntime
import numpy as np
import torch
import torch.nn.functional as F

import pandas as pd

# import onnx model
ort_session = onnxruntime.InferenceSession('resnet18_imagenet.onnx')
# ort_session = onnxruntime.InferenceSession('resnet50_imagenet.onnx')
# ort_session = onnxruntime.InferenceSession('resnet152_imagenet.onnx')


# construct input to text
x = torch.randn(1, 3, 256, 256).numpy()
# print(x.shape)

# onnx runtime input
ort_inputs = {'in': x}

# onnx runtime output
ort_output = ort_session.run(['out'], ort_inputs)[0]
# print(ort_output.shape)

# real text image
img_path = 'orange.jpg'

# use pillow import
from PIL import Image
img_pil = Image.open(img_path)
# img_pil.show()

# pre-deal
from torchvision import transforms

# pre deal
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(256),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])

input_img = test_transform(img_pil)
# print(input_img.shape)

input_tensor = input_img.unsqueeze(0).numpy()
# print(input_tensor.shape)

# inference
# ONNX Runtime input
ort_inputs = {'in': input_tensor}

# ONNX Runtime output
pred_logits = ort_session.run(['out'], ort_inputs)[0]
pred_logits = torch.tensor(pred_logits)
# no softmax
# print(pred_logits.shape)

# do softmax
pred_softmax = F.softmax(pred_logits, dim=1)

# analyze inference result
k = 5
top_k = torch.topk(pred_softmax, k)

# get class and score
pred_ids, confs = top_k.indices.numpy()[0], top_k.values.numpy()[0]

# reflat
df = pd.read_csv('imagenet_class_index.csv')
idx_to_labels = {}
for idx, row in df.iterrows():
    idx_to_labels[row['ID']] = row['class']

for i in range(k):
    class_name = idx_to_labels[pred_ids[i]] # 获取类别名称
    confidence = confs[i] * 100             # 获取置信度
    text = '{:<20} {:>.3f}'.format(class_name, confidence)
    print(text)