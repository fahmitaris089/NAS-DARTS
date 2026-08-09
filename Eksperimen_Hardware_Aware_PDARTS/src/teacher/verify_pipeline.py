"""Quick verification — run: python3 verify_pipeline.py"""
from palm_vein_dataset import create_subject_split, get_transforms
from model_factory import create_model, get_available_models, freeze_backbone, unfreeze_backbone

print("Available models:", get_available_models())

model, inp_size = create_model("MobileNetV3Large", num_classes=834)
print(f"MobileNetV3Large created — input_size={inp_size}")

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params: {total:,}  Trainable: {trainable:,}")

freeze_backbone(model, "MobileNetV3Large")
frozen_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  After freeze — trainable: {frozen_train:,}")

unfreeze_backbone(model)
unfrozen_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  After unfreeze — trainable: {unfrozen_train:,}")

t_train = get_transforms("train", 224, use_augmentation=True)
t_val = get_transforms("val", 224, use_augmentation=False)
print(f"Transforms OK — train: {len(t_train.transforms)} ops, val: {len(t_val.transforms)} ops")

print("\nAll imports and module tests passed!")
