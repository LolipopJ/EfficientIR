# EfficientIR - 基于 EfficientNet 的图片检索工具

> 基于内容的图片检索方法有很多，即便是传统算法也能达到不错的效果。 本项目起源于要求使用传统算法（SIFT）完成的课程作业，但实现过程中遇到了一些影响实际使用的问题（例如建立索引慢）。 经过简单调研（指咨询群友和 Google）后发现 PC 上并没有什么好用的本地图片检索工具，于是干脆在作业外另外弄了这个小工具。
>
> 使用到的主要工具是 EfficientNet 和 Hnswlib，使用前者在 ImageNet 上的预训练模型进行特征抽取，使用后者进行特征索引及检索。
>
> @Sg4Dylan

## 功能特性

- 以图搜图
- 图片相似度计算
- 重复图片查找

## 本地构建

构建于 `python>=3.12`，其它版本未测试。执行以下命令安装依赖与构建可执行文件：

```bash
pip install -r requirements.txt
pyinstaller build.spec
```

## 使用

图形化界面请移步项目 [LolipopJ/dupimg-finder](https://github.com/LolipopJ/dupimg-finder)。

## 更换模型

目前包含以下模型：

- EfficientNet-B2：`models/imagenet-b2-opti.onnx`

关于将 PyTorch 或其他机器学习框架模型导出为 ONNX 模型的方法此处不再赘述。需要注意的是，导出的 ONNX 模型必须经过优化过程，可以使用 [onnx-simplifier](https://github.com/daquexian/onnx-simplifier) （推荐）或本项目包含的 `opti.py`。

更换模型前可以使用 [Netron](https://lutzroeder.github.io/netron/) 检查模型的输入矩阵形状是否为 `1x3xNxN`,输出向量是否为 `1xN`。其中输入矩阵的形状对应 `efficient_ir.py` 中的常量 `img_size`，输出向量的形状对应 `init_index()` 方法的初始化维度 以及 `get_fv()` 方法的返回值。

若只是希望更换模型为更大的 EfficientNet 模型，那么只需要确认并修改 `efficient_ir.py` 中的 `img_size` 和 `model_path`。

但如果需要更换为 Once For All 模型，虽然其输入与 EfficientNet-B2 相同，但输出是 `N` 并不是 `1xN`，故除修改 `model_path` 外，还需将 `get_fv()` 中返回所在行做出如下修改:

```python
# with EfficientNet-B2 模型
return self.session.run([], {self.model_input: norm_img_data})[0][0]
# with Once For All 模型
return self.session.run([], {self.model_input: norm_img_data})[0]
```

**注意:** 更换模型后一定要重新建立索引。

## GPU 加速

当前选择的模型均对性能进行了权衡，在支持 AVX 指令集的 CPU 上索引速度略高于甜品级 GPU。

若期待通过 GPU 加速获得更好的性能及加速比，可以将模型换成更大规模的或增加索引时同时处理的图片数量。

具体的操作请自行阅读并修改代码实现。

如果是 NVIDIA 显卡，切换 GPU 推理的步骤：

1. 安装 `onnxruntime-gpu` ；
2. 取消 `efficient_ir.py` 的相应注释；
3. 将 provider 需要换成 `GPUExecutionProvider`。

支持 DX12 Compute 的任意显卡（包括集成显卡），切换 GPU 推理的步骤：

1. 安装 `onnxruntime-dml` ；
2. 取消 `efficient_ir.py` 的相应注释。

## Q&A

> Q：可承载最大索引数量是多少？如何修改？  
> A：目前是 1000000。可以在 `efficient_ir.py` 中修改，数值将在下一次加载时生效。

> Q：检索效果不佳怎么解决？  
> A：当前代码中使用 EfficientNet-b2 模型是经过权衡后决定的，若追求更佳检索效果请自行更换更大规模的 EfficientNet 模型或其他的 SOTA 模型。本项目将持续关注 SOTA 模型的发展，并在 [Wiki](https://github.com/Sg4Dylan/EfficientIR/wiki) 中更新相关测试结果。

## References

- [EfficientNet PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
- [Once For All](https://github.com/mit-han-lab/once-for-all)
- [Hnswlib](https://github.com/nmslib/hnswlib)
