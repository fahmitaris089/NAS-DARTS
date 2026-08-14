from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from adaface import AdaFaceHead
from knowledge_distilation.kd_loss import DecoupledKDLoss


class DKDTests(unittest.TestCase):
    def test_dkd_finite_for_834_classes_and_batch_one(self):
        torch.manual_seed(7)
        student = torch.randn(1, 834, requires_grad=True)
        teacher = torch.randn(1, 834)
        target = torch.tensor([17])
        criterion = DecoupledKDLoss(temperature=4, alpha=1, beta=8, warmup_epochs=20)
        loss, parts = criterion(student, teacher, target, epoch=10)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(parts["dkd_warmup"], 0.5)
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_dkd_zero_when_teacher_and_student_match_except_ce(self):
        torch.manual_seed(3)
        logits = torch.randn(4, 834)
        targets = torch.tensor([0, 1, 2, 3])
        criterion = DecoupledKDLoss(warmup_epochs=0)
        total, parts = criterion(logits, logits.clone(), targets)
        self.assertLess(abs(parts["loss_tckd"]), 1e-5)
        self.assertLess(abs(parts["loss_nckd"]), 1e-5)
        self.assertAlmostEqual(total.item(), F.cross_entropy(logits, targets).item(), places=5)


class AdaFaceTests(unittest.TestCase):
    def test_training_margin_and_inference_are_finite(self):
        torch.manual_seed(11)
        head = AdaFaceHead(32, 834)
        embeddings = torch.randn(8, 32, requires_grad=True)
        labels = torch.arange(8)
        inference = head(embeddings)
        margin = head(embeddings, labels)
        self.assertEqual(tuple(inference.shape), (8, 834))
        self.assertFalse(torch.allclose(inference, margin))
        loss = F.cross_entropy(margin, labels)
        loss.backward()
        self.assertTrue(torch.isfinite(embeddings.grad).all())
        self.assertTrue(torch.isfinite(head.batch_mean))
        self.assertTrue(torch.isfinite(head.batch_std))

    def test_weight_normalization_does_not_mutate_parameter(self):
        head = AdaFaceHead(16, 10)
        before = head.weight.detach().clone()
        _ = head(torch.randn(2, 16))
        self.assertTrue(torch.equal(before, head.weight.detach()))

    def test_checkpoint_resume_preserves_head_and_statistics(self):
        torch.manual_seed(13)
        source = AdaFaceHead(16, 10)
        _ = source(torch.randn(8, 16), torch.arange(8) % 10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.pth"
            torch.save(source.state_dict(), path)
            restored = AdaFaceHead(16, 10)
            restored.load_state_dict(torch.load(path, map_location="cpu"))
        self.assertTrue(torch.equal(source.weight, restored.weight))
        self.assertTrue(torch.equal(source.batch_mean, restored.batch_mean))
        self.assertTrue(torch.equal(source.batch_std, restored.batch_std))

    def test_onnx_inference_parity_when_runtime_is_available(self):
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime unavailable")
        torch.manual_seed(17)
        head = AdaFaceHead(16, 10).eval()
        sample = torch.randn(3, 16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adaface.onnx"
            torch.onnx.export(head, sample, path, input_names=["embedding"], output_names=["logits"], opset_version=13)
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            actual = session.run(None, {"embedding": sample.numpy()})[0]
        expected = head(sample).detach().numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
        self.assertTrue(np.array_equal(actual.argmax(1), expected.argmax(1)))


if __name__ == "__main__":
    unittest.main()
