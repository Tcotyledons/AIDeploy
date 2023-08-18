# use gpu
import torch
from torchvision import models

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print('device', device)

# use resnet train in imageNet
model = models.resnet18(pretrained=True)
model = model.eval()

# build a tensor input shape
x = torch.randn(1, 3, 256, 256)
# output 1000 class
output = model(x)
# print(output.shape)

# pytorch convert to onnx
with torch.no_grad():
    torch.onnx.export(
        model,                       # model to change
        x,                           # random input of model
        'resnet18_imagenet.onnx',    # export name
        input_names=['in'],       # input name
        output_names=['out']      # output name
    )


# check if export successfully also in https://netron.app
import onnx

onnx_model = onnx.load('resnet18_imagenet.onnx')
onnx.checker.check_model(onnx_model)
print('onnx OK')