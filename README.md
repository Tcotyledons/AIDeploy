# ONNX+Flask+Web三件套部署ResNet 18/50/152

## 0 环境配置

本项目默认你有conda虚拟环境+python3.6 + pytorch1.2.0 + torchvision0.4.0，并且安装了pycharm专业版，另外需要执行以下的指令安装所需要的依赖：

```python
pip install -r requirements.txt
```

如果安装失败，可以添加清华镜像源。以下所有的实验均用windows10实现，没有什么命令行操作。

## 1 onnx文件的生成

ONNX是一个开放的深度学习模型表示标准，旨在实现深度学习框架之间的互操作性和跨平台部署的便利性。 可以作为深度学习模型部署在安卓端，本地端，Web端的中间格式。这里就是部署在Web端的demo。

首先，项目主目录下有三个python文件：imgnet_resnet18.py，imgnet_resnet50.py，imgnet_resnet152.py，分别使用了ImageNet上预训练的ResNet18/50/152三个模型进行对应onnx文件的生成，可以根据需要运行，这里假设需要生成ResNet18的onnx文件。如果pycharm终端显示：

![image](https://github.com/Tcotyledons/AIDeploy/blob/main/pic/onnxOK.png)

则证明ResNet18的onnx文件成功生成。

## 2 测试ONNX文件的可用性

生成ONNX文件后，需要测试ONNX的可用性，这里用了目录文件的一个orange.jpg橙子的图片进行测试，运行目录中的text_onnx.py文件，如果得到以下的输出，则证明模型测试没问题。

![image](https://github.com/Tcotyledons/AIDeploy/blob/main/pic/testOK.png)

如果生成的是ResNet50/152，需要修改text_onnx.py中的一行即可：

```
ort_session = onnxruntime.InferenceSession('onnx文件名')
```

## 3 部署到Web端，浏览器进行验证

那么到这里为止，部署前的准备工作都已经完成了，这时候只需要启动项目就可以了，这里需要检查在pycharm的Edit Configurations的Script path是不是指向项目里的text.py文件，没问题的话启动，点击pycharm终端显示的网页如下图

![image](https://github.com/Tcotyledons/AIDeploy/blob/main/pic/allview.png)

就进入了项目的浏览器界面。

## 4 上传自己的图片进行预测

然后就进入到了浏览器的界面，这里展示的是谷歌浏览器，模型是ResNet152。

![image](https://github.com/Tcotyledons/AIDeploy/blob/main/pic/web.png)

点击"选择文件"，在点击预测，就可以得到图片的预测置信度top5的score和对应类别。

![image](https://github.com/Tcotyledons/AIDeploy/blob/main/pic/predict.jpg)
