from __future__ import annotations

import torch
import torch.nn.functional as F

from knowledge_distilation.kd_loss import LogitStandardizationKDLoss, get_kd_loss


def _official_standardize(logits: torch.Tensor) -> torch.Tensor:
    return (logits - logits.mean(dim=-1, keepdim=True)) / (
        1e-7 + logits.std(dim=-1, keepdim=True)
    )


def test_matches_public_reference_formula() -> None:
    student = torch.tensor(
        [[1.0, -0.5, 2.0], [0.2, 0.8, -1.0]], requires_grad=True
    )
    teacher = torch.tensor([[3.0, -1.0, 0.5], [-0.2, 1.4, 0.7]])
    targets = torch.tensor([2, 1])
    temperature = 2.0
    criterion = LogitStandardizationKDLoss(
        temperature=temperature,
        ce_weight=1.0,
        kd_weight=9.0,
        label_smoothing=0.2,
    )
    loss, breakdown = criterion(student, teacher, targets)

    expected_ce = F.cross_entropy(student, targets, label_smoothing=0.2)
    expected_kd = F.kl_div(
        F.log_softmax(_official_standardize(student) / temperature, dim=1),
        F.softmax(_official_standardize(teacher) / temperature, dim=1),
        reduction="batchmean",
    ) * temperature**2
    expected = expected_ce + 9.0 * expected_kd
    torch.testing.assert_close(loss, expected)
    assert abs(breakdown["loss_ls_kd"] - expected_kd.item()) < 1e-7
    assert abs(breakdown["loss_ls_kd_weighted"] - 9.0 * expected_kd.item()) < 1e-6


def test_standardized_kd_is_invariant_to_positive_affine_logit_scale() -> None:
    generator = torch.Generator().manual_seed(42)
    student = torch.randn(4, 11, generator=generator)
    teacher = torch.randn(4, 11, generator=generator)
    targets = torch.tensor([0, 2, 4, 6])
    criterion = LogitStandardizationKDLoss(ce_weight=0.0, kd_weight=1.0)
    loss_a, _ = criterion(student, teacher, targets)
    loss_b, _ = criterion(3.5 * student + 17.0, 0.25 * teacher - 8.0, targets)
    torch.testing.assert_close(loss_a, loss_b, atol=2e-6, rtol=2e-6)


def test_gradient_reaches_student_but_not_teacher() -> None:
    student = torch.randn(2, 834, requires_grad=True)
    teacher = torch.randn(2, 834, requires_grad=True)
    targets = torch.tensor([1, 700])
    criterion = LogitStandardizationKDLoss()
    loss, _ = criterion(student, teacher, targets)
    loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert teacher.grad is None


def test_zero_kd_weight_reduces_to_ordinary_ce() -> None:
    student = torch.randn(3, 9, requires_grad=True)
    teacher = torch.randn(3, 9)
    targets = torch.tensor([0, 4, 8])
    criterion = LogitStandardizationKDLoss(
        ce_weight=1.0, kd_weight=0.0, label_smoothing=0.2
    )
    loss, _ = criterion(student, teacher, targets)
    expected = F.cross_entropy(student, targets, label_smoothing=0.2)
    torch.testing.assert_close(loss, expected)


def test_factory_exposes_logit_standardization() -> None:
    criterion = get_kd_loss(
        "logit_standardization",
        temperature=2,
        ce_weight=1,
        ls_kd_weight=9,
        ls_eps=1e-7,
        label_smoothing=0.2,
    )
    assert isinstance(criterion, LogitStandardizationKDLoss)
