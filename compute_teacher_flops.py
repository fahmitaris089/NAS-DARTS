"""Hitung FLOPs (MMACs, konvensi thop) + params untuk semua teacher di 224x224, 834 kelas.
FLOPs independen terhadap nilai bobot → pakai weights=None (tanpa download).
"""
import torch
import torchvision.models as m
from thop import profile

NUM = 834
INP = torch.randn(1, 3, 224, 224)

def build(name):
    if name == "ResNet50":        return m.resnet50(weights=None, num_classes=NUM)
    if name == "VGG16":           return m.vgg16(weights=None, num_classes=NUM)
    if name == "DenseNet121":     return m.densenet121(weights=None, num_classes=NUM)
    if name == "EfficientNetB4":  return m.efficientnet_b4(weights=None, num_classes=NUM)
    if name == "EfficientNetV2M": return m.efficientnet_v2_m(weights=None, num_classes=NUM)
    if name == "MobileNetV3Large":return m.mobilenet_v3_large(weights=None, num_classes=NUM)
    if name == "ConvNeXtBase":    return m.convnext_base(weights=None, num_classes=NUM)
    if name == "RegNetY16GF":     return m.regnet_y_16gf(weights=None, num_classes=NUM)
    if name == "InceptionV3":     return m.inception_v3(weights=None, num_classes=NUM,
                                                        aux_logits=False, init_weights=False)
    raise ValueError(name)

names = ["EfficientNetV2M","ResNet50","ConvNeXtBase","RegNetY16GF","DenseNet121",
         "MobileNetV3Large","InceptionV3","EfficientNetB4","VGG16"]

print(f"{'Model':<18}{'Params(M)':>12}{'MMACs(224)':>14}")
for n in names:
    net = build(n).eval()
    macs, params = profile(net, inputs=(INP,), verbose=False)
    print(f"{n:<18}{params/1e6:>12.2f}{macs/1e6:>14.1f}")
