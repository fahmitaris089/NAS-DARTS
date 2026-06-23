"""
Model Factory — 9 CNN Models for Palm Vein Recognition
======================================================
All models use pretrained ImageNet weights.
Classifier head is replaced for num_classes.
"""

import torch
import torch.nn as nn
from torchvision import models

try:
    import timm
except ImportError:  # GhostNet baseline uses timm; keep other torchvision models usable.
    timm = None


# ─── Model Registry ──────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "InceptionV3": {
        "input_size": 299,
        "description": "Multi-scale CNN with auxiliary classifier",
    },
    "ResNet50": {
        "input_size": 224,
        "description": "50-layer Residual Network",
    },
    "VGG16": {
        "input_size": 224,
        "description": "16-layer Very Deep CNN",
    },
    "DenseNet121": {
        "input_size": 224,
        "description": "121-layer Densely Connected CNN",
    },
    "EfficientNetB4": {
        "input_size": 224,
        "description": "EfficientNet B4 with compound scaling",
    },
    "EfficientNetV2M": {
        "input_size": 224,
        "description": "EfficientNet V2 Medium — improved training",
    },
    "MobileNetV3Large": {
        "input_size": 224,
        "description": "MobileNet V3 Large — lightweight CNN",
    },
    "MobileNetV3Small": {
        "input_size": 224,
        "description": "MobileNet V3 Small — lightweight CNN",
    },
    "ShuffleNetV2_x0_5": {
        "input_size": 224,
        "description": "ShuffleNet V2 0.5x — lightweight CNN",
    },
    "ShuffleNetV2_x1_0": {
        "input_size": 224,
        "description": "ShuffleNet V2 1.0x — lightweight CNN",
    },
    "GhostNet_050": {
        "input_size": 224,
        "description": "GhostNet 0.5x — cheap-operation lightweight CNN (no ImageNet pretrained weights in timm)",
    },
    "GhostNet_100": {
        "input_size": 224,
        "description": "GhostNet 1.0x — cheap-operation lightweight CNN",
    },
    "EfficientNetLite0": {
        "input_size": 224,
        "description": "EfficientNet-Lite0 — mobile-optimized EfficientNet baseline",
    },
    "ConvNeXtBase": {
        "input_size": 224,
        "description": "ConvNeXt Base — modernized ResNet (2022)",
    },
    "RegNetY16GF": {
        "input_size": 224,
        "description": "RegNet Y 16GF — NAS-designed CNN with SE blocks",
    },
}


def get_available_models():
    """Return list of available model names."""
    return list(MODEL_CONFIGS.keys())


def get_input_size(model_name):
    """Return required input size for a model."""
    assert model_name in MODEL_CONFIGS, f"Unknown model: {model_name}"
    return MODEL_CONFIGS[model_name]["input_size"]


# ─── Head Replacement Helpers ────────────────────────────────────────────────

def _replace_fc(model, num_classes, dropout=0.3):
    """Replace model.fc (ResNet, RegNet style)."""
    in_feat = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_feat, num_classes),
    )
    return model


def _replace_classifier_index(model, idx, num_classes, dropout=None):
    """Replace model.classifier[idx]."""
    in_feat = model.classifier[idx].in_features
    if dropout is not None:
        # Insert dropout before the linear layer if not already present
        model.classifier[idx] = nn.Linear(in_feat, num_classes)
    else:
        model.classifier[idx] = nn.Linear(in_feat, num_classes)
    return model


# ─── Model Builders ─────────────────────────────────────────────────────────

def _build_inception_v3(num_classes):
    model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)
    # Main head
    in_feat = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_feat, num_classes),
    )
    # Auxiliary head
    aux_in = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(aux_in, num_classes),
    )
    return model


def _build_resnet50(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    return _replace_fc(model, num_classes, dropout=0.5)


def _build_vgg16(num_classes):
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    # VGG16 classifier: [Linear, ReLU, Dropout, Linear, ReLU, Dropout, Linear]
    #                     0       1     2        3       4     5        6
    in_feat = model.classifier[6].in_features
    model.classifier[5] = nn.Dropout(p=0.5)  # replace existing dropout (0.5 vs default 0.5)
    model.classifier[6] = nn.Linear(in_feat, num_classes)
    return model


def _build_densenet121(num_classes):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    in_feat = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_feat, num_classes),
    )
    return model


def _build_efficientnet_b4(num_classes):
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    # classifier: [Dropout, Linear]
    in_feat = model.classifier[1].in_features
    model.classifier[0] = nn.Dropout(p=0.5)
    model.classifier[1] = nn.Linear(in_feat, num_classes)
    return model


def _build_efficientnet_v2m(num_classes):
    model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
    in_feat = model.classifier[1].in_features
    model.classifier[0] = nn.Dropout(p=0.5)
    model.classifier[1] = nn.Linear(in_feat, num_classes)
    return model


def _build_mobilenet_v3_large(num_classes):
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    # classifier: [Linear, Hardswish, Dropout, Linear]
    #              0       1          2        3
    in_feat = model.classifier[3].in_features
    model.classifier[2] = nn.Dropout(p=0.5)
    model.classifier[3] = nn.Linear(in_feat, num_classes)
    return model


def _build_mobilenet_v3_small(num_classes):
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    # classifier: [Linear, Hardswish, Dropout, Linear]
    in_feat = model.classifier[3].in_features
    model.classifier[2] = nn.Dropout(p=0.5)
    model.classifier[3] = nn.Linear(in_feat, num_classes)
    return model


def _build_shufflenet_v2_x0_5(num_classes):
    model = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
    return _replace_fc(model, num_classes, dropout=0.5)


def _build_shufflenet_v2_x1_0(num_classes):
    model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
    return _replace_fc(model, num_classes, dropout=0.5)


def _build_ghostnet(model_name, num_classes, pretrained=True):
    if timm is None:
        raise ImportError(
            "GhostNet baseline requires timm. Install with: python3 -m pip install timm"
        )
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=0.5)


def _build_ghostnet_050(num_classes):
    # timm does not provide ImageNet pretrained weights for ghostnet_050.
    # Use only as an extra from-scratch ablation, not as the main fair baseline.
    return _build_ghostnet("ghostnet_050", num_classes, pretrained=False)


def _build_ghostnet_100(num_classes):
    return _build_ghostnet("ghostnet_100", num_classes)


def _build_efficientnet_lite0(num_classes):
    if timm is None:
        raise ImportError(
            "EfficientNet-Lite baseline requires timm. Install with: python3 -m pip install timm"
        )
    return timm.create_model(
        "tf_efficientnet_lite0",
        pretrained=True,
        num_classes=num_classes,
        drop_rate=0.5,
    )


def _build_convnext_base(num_classes):
    model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
    # classifier: [LayerNorm2d, Flatten, Linear]
    #              0             1       2
    in_feat = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        model.classifier[0],  # LayerNorm2d
        model.classifier[1],  # Flatten
        nn.Dropout(p=0.5),
        nn.Linear(in_feat, num_classes),
    )
    return model


def _build_regnety_16gf(num_classes):
    model = models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.DEFAULT)
    return _replace_fc(model, num_classes, dropout=0.5)


# ─── Builder Map ─────────────────────────────────────────────────────────────

_BUILDERS = {
    "InceptionV3":      _build_inception_v3,
    "ResNet50":         _build_resnet50,
    "VGG16":            _build_vgg16,
    "DenseNet121":      _build_densenet121,
    "EfficientNetB4":   _build_efficientnet_b4,
    "EfficientNetV2M":  _build_efficientnet_v2m,
    "MobileNetV3Large": _build_mobilenet_v3_large,
    "MobileNetV3Small": _build_mobilenet_v3_small,
    "ShuffleNetV2_x0_5": _build_shufflenet_v2_x0_5,
    "ShuffleNetV2_x1_0": _build_shufflenet_v2_x1_0,
    "GhostNet_050": _build_ghostnet_050,
    "GhostNet_100": _build_ghostnet_100,
    "EfficientNetLite0": _build_efficientnet_lite0,
    "ConvNeXtBase":     _build_convnext_base,
    "RegNetY16GF":      _build_regnety_16gf,
}


# ─── Public API ──────────────────────────────────────────────────────────────

def create_model(model_name, num_classes):
    """
    Create a pretrained model with classifier head replaced.

    Args:
        model_name: one of get_available_models()
        num_classes: number of output classes

    Returns: (model, input_size)
    """
    assert model_name in _BUILDERS, \
        f"Unknown model '{model_name}'. Available: {get_available_models()}"

    print(f"Creating {model_name} (pretrained=ImageNet, num_classes={num_classes})")
    print(f"  {MODEL_CONFIGS[model_name]['description']}")

    model      = _BUILDERS[model_name](num_classes)
    input_size = MODEL_CONFIGS[model_name]["input_size"]

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {train_params:,}")
    print(f"  Input size      : {input_size}x{input_size}")

    return model, input_size


def get_backbone_and_head_params(model, model_name):
    """
    Separate model parameters into backbone vs head for differential LR.

    Returns: (backbone_params, head_params)
    """
    head_params     = []
    backbone_params = []

    if model_name == "InceptionV3":
        head_names = {"fc", "AuxLogits"}
    elif model_name in ("ResNet50", "RegNetY16GF", "ShuffleNetV2_x0_5", "ShuffleNetV2_x1_0"):
        head_names = {"fc"}
    elif model_name in ("VGG16", "DenseNet121", "EfficientNetB4",
                        "EfficientNetV2M", "MobileNetV3Large", "MobileNetV3Small",
                        "GhostNet_050", "GhostNet_100", "EfficientNetLite0",
                        "ConvNeXtBase"):
        head_names = {"classifier"}
    else:
        head_names = {"fc", "classifier"}

    for name, param in model.named_parameters():
        top_level = name.split(".")[0]
        if top_level in head_names:
            head_params.append(param)
        else:
            backbone_params.append(param)

    return backbone_params, head_params


def freeze_backbone(model, model_name):
    """Freeze backbone parameters (only head is trainable)."""
    backbone_params, _ = get_backbone_and_head_params(model, model_name)
    for p in backbone_params:
        p.requires_grad = False

    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    total  = sum(1 for p in model.parameters())
    print(f"  Frozen: {frozen}/{total} parameter groups")


def unfreeze_backbone(model):
    """Unfreeze all parameters."""
    for p in model.parameters():
        p.requires_grad = True
    print("  Unfroze all parameters")


# ─── Quick Test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Available models:")
    for name in get_available_models():
        cfg = MODEL_CONFIGS[name]
        print(f"  {name:20s}  input={cfg['input_size']}  {cfg['description']}")

    # Test one model
    print("\n--- Testing ResNet50 ---")
    model, input_size = create_model("ResNet50", num_classes=834)
    x = torch.randn(2, 3, input_size, input_size)
    with torch.no_grad():
        out = model(x)
    print(f"  Output shape: {out.shape}")  # (2, 834)
    print("Done.")
