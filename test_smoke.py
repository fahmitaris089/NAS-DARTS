"""Quick test for dataset and full pipeline smoke test."""
import sys
sys.path.insert(0, ".")

from palm_vein_dataset import create_search_dataloaders, create_retrain_dataloaders

print("=== Search DataLoaders ===")
tr, vs, vl, tl, info = create_search_dataloaders(num_workers=0, batch_size=4)
bx, by = next(iter(tr))
print(f"Batch: {bx.shape}, labels: {by.tolist()}")
print(f"Classes: {info['num_classes']}")
print(f"Train search: {info['search_train_size']}, Val search: {info['search_val_size']}")

print("\n=== Retrain DataLoaders ===")
tr2, vl2, tl2, info2 = create_retrain_dataloaders(num_workers=0, batch_size=4)
bx2, by2 = next(iter(tr2))
print(f"Batch: {bx2.shape}")
print(f"Train: {info2['train_size']}, Val: {info2['val_size']}, Test: {info2['test_size']}")

print("\n=== Verify same split as teacher ===")
assert info2['train_size'] == 6672, f"Expected 6672 train, got {info2['train_size']}"
assert info2['val_size'] == 834, f"Expected 834 val, got {info2['val_size']}"
assert info2['test_size'] == 834, f"Expected 834 test, got {info2['test_size']}"
assert info['num_classes'] == 834, f"Expected 834 classes, got {info['num_classes']}"
print("[OK] Split matches teacher (6672/834/834, 834 classes)")

print("\nAll dataset tests passed!")
