import unittest
import torch

from palm_vein_dataset import PKBatchSampler
from knowledge_distilation.adaptive_center_relation import (
    AdaptiveCenterRelationLoss, load_center_cache, save_center_cache,
)


def make_samples(classes=834, per_class=8):
    return [(f"{label}_{index}.bmp", label) for label in range(classes) for index in range(per_class)]


def make_loss(relation_weight=0.05):
    torch.manual_seed(1)
    centers = torch.nn.functional.normalize(torch.randn(6, 7), dim=1)
    return AdaptiveCenterRelationLoss(
        student_dim=5, teacher_dim=7, num_classes=6, initial_centers=centers,
        relation_weight=relation_weight, topk_negatives=2,
    )


class PKAdaptiveTests(unittest.TestCase):
  def test_pk_sampler_scut_schedule_and_determinism(self):
    samples = make_samples()
    left = PKBatchSampler(samples, p_classes=16, k_samples=4, seed=42)
    right = PKBatchSampler(samples, p_classes=16, k_samples=4, seed=42)
    left.set_epoch(0); right.set_epoch(0)
    batches_left = list(left); batches_right = list(right)
    self.assertEqual(len(batches_left), 105)
    self.assertEqual(batches_left, batches_right)
    for batch in batches_left:
        labels = [samples[index][1] for index in batch]
        self.assertEqual(len(batch), 64)
        self.assertEqual(len(set(labels)), 16)
        self.assertTrue(all(labels.count(label) == 4 for label in set(labels)))
    self.assertEqual(min(left.last_epoch_class_counts.values()), 2)
    self.assertEqual(sum(count == 3 for count in left.last_epoch_class_counts.values()), 12)


  def test_pk_sampler_epoch_rotation_and_resume(self):
    samples = make_samples(34, 8)
    sampler = PKBatchSampler(samples, p_classes=16, k_samples=4, seed=7)
    sampler.set_epoch(3)
    state = sampler.state_dict()
    expected = list(sampler)
    resumed = PKBatchSampler(samples, p_classes=16, k_samples=4, seed=7)
    resumed.load_state_dict(state)
    self.assertEqual(list(resumed), expected)
    resumed.set_epoch(4)
    self.assertNotEqual(list(resumed), expected)


  def test_pk_sampler_replacement_metadata(self):
    samples = make_samples(16, 3)
    sampler = PKBatchSampler(samples, p_classes=16, k_samples=4)
    self.assertEqual(sampler.replacement_labels, list(range(16)))
    self.assertEqual(len(next(iter(sampler))), 64)

  def test_adaptive_loss_finite_gradient_and_normalized_centers(self):
    criterion = make_loss()
    logits = torch.randn(6, 6, requires_grad=True)
    student = torch.randn(6, 5, requires_grad=True)
    teacher = torch.randn(6, 7)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    loss, parts = criterion(logits, student, teacher, labels, epoch=10)
    loss.backward()
    self.assertTrue(torch.isfinite(loss))
    self.assertTrue(torch.isfinite(student.grad).all())
    self.assertTrue(torch.isfinite(criterion.adapter.weight.grad).all())
    self.assertTrue(torch.allclose(criterion.centers.norm(dim=1), torch.ones(6), atol=1e-5))
    self.assertGreater(parts["positive_pairs"], 0)


  def test_center_update_is_order_invariant(self):
    first, second = make_loss(), make_loss()
    second.load_state_dict(first.state_dict())
    logits = torch.randn(4, 6)
    student = torch.randn(4, 5)
    teacher = torch.randn(4, 7)
    labels = torch.tensor([0, 0, 1, 1])
    first(logits, student, teacher, labels)
    order = torch.tensor([1, 0, 3, 2])
    second(logits[order], student[order], teacher[order], labels[order])
    self.assertTrue(torch.allclose(first.centers, second.centers, atol=1e-6))


  def test_relation_handles_batch_one_and_no_valid_pair(self):
    criterion = make_loss()
    logits = torch.randn(1, 6, requires_grad=True)
    student = torch.randn(1, 5, requires_grad=True)
    teacher = torch.randn(1, 7)
    loss, parts = criterion(logits, student, teacher, torch.tensor([0]))
    loss.backward()
    self.assertTrue(torch.isfinite(loss))
    self.assertEqual(parts["positive_pairs"], 0)
    self.assertEqual(parts["mined_negative_pairs"], 0)


  def test_center_cache_rejects_stale_metadata(self):
    import tempfile
    from pathlib import Path
    path = Path(tempfile.mkdtemp()) / "centers.pth"
    centers = torch.nn.functional.normalize(torch.randn(3, 4), dim=1)
    save_center_cache(path, centers, {"split": "abc", "num_classes": 3})
    loaded = load_center_cache(path, {"split": "abc", "num_classes": 3})
    self.assertEqual(loaded.shape, (3, 4))
    with self.assertRaisesRegex(ValueError, "Stale"):
        load_center_cache(path, {"split": "different", "num_classes": 3})

  def test_center_dimension_mismatch_is_rejected(self):
    centers = torch.nn.functional.normalize(torch.randn(5, 7), dim=1)
    with self.assertRaisesRegex(ValueError, "center shape"):
        AdaptiveCenterRelationLoss(5, 7, 6, centers)

  def test_subtle_relations_are_ignored(self):
    criterion = make_loss()
    embeddings = torch.nn.functional.normalize(torch.randn(4, 7), dim=1)
    loss, positives, negatives = criterion._relation_loss(
        embeddings, embeddings.clone(), torch.tensor([0, 0, 1, 1])
    )
    self.assertEqual(float(loss), 0.0)
    self.assertEqual(positives, 0)
    self.assertEqual(negatives, 0)

  def test_progressive_stages_and_gradient_ratio_calibration(self):
    torch.manual_seed(19)
    centers = torch.nn.functional.normalize(torch.randn(6, 7), dim=1)
    criterion = AdaptiveCenterRelationLoss(
        student_dim=5, teacher_dim=7, num_classes=6, initial_centers=centers,
        progressive_staging=True, center_start_epoch=2, relation_start_epoch=3,
        calibration_batches=1, warmup_epochs=1, relation_weight=0.05,
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    classifier = torch.randn(5, 6)
    def run(epoch):
        student = torch.randn(6, 5, requires_grad=True)
        logits = student @ classifier
        teacher = torch.randn(6, 7)
        return criterion(logits, student, teacher, labels, epoch=epoch, batch_index=0)
    _, stage1 = run(1)
    self.assertEqual(stage1["adaptive_stage"], 1)
    self.assertEqual(stage1["loss_kd"], 0.0)
    loss2, stage2 = run(2)
    self.assertEqual(stage2["adaptive_stage"], 2)
    self.assertGreater(stage2["center_weight_effective"], 0.0)
    loss2.backward()
    loss3, stage3 = run(3)
    self.assertEqual(stage3["adaptive_stage"], 3)
    self.assertGreater(stage3["relation_weight_effective"], 0.0)
    loss3.backward()
    self.assertTrue(bool(criterion.center_feature_calibrated))
    self.assertTrue(bool(criterion.relation_calibrated))


if __name__ == "__main__":
    unittest.main()
