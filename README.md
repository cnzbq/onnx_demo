# onnx

### tensorflow 2.x saved model（*.pb）to onnx

安装依赖

```bash
  pip install tensorflow
  pip install tf2onnx
```

以转换sj为例，目录结构如下

```text
└─demo # 模型文件夹
   ├─...
   ├─variables
   │  ├─...
   │  └─...
   ├─saved_model.pb
   └─...
```

通过以下命令实现转换

```bash
  python -m tf2onnx.convert --saved-model "./demo" --output demo.onnx 
```

tensorflow-onnx: [GitHub](https://github.com/onnx/tensorflow-onnx/)

### pyTorch to onnx

