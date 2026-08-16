import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "knowledge_distilation"))

from icd_compactness import ICDCompactnessLoss, SynchronizedClassFeatureBank


def test_feature_bank_replacement_and_validity_expiry():
    bank = SynchronizedClassFeatureBank(
        num_classes=2, bank_size=2, student_dim=3, teacher_dim=4, valid_steps=2
    )
    bank.update(torch.tensor([[1.0, 0, 0]]), torch.tensor([[1.0, 0, 0, 0]]), torch.tensor([0]))
    assert bank.validity[0].tolist() == [1, 0]
    bank.update(torch.tensor([[0.0, 1, 0]]), torch.tensor([[0.0, 1, 0, 0]]), torch.tensor([1]))
    assert bank.validity[0].tolist() == [0, 0]
    assert bank.validity[1].tolist() == [1, 0]


def test_positive_pairs_are_same_class_and_student_path_is_differentiable():
    bank = SynchronizedClassFeatureBank(
        num_classes=3, bank_size=2, student_dim=2, teacher_dim=3, valid_steps=10
    )
    student = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    teacher = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    labels = torch.tensor([0, 1])
    bank.update(student, teacher, labels)
    student_scores, teacher_scores = bank.positive_similarities(student, teacher, labels)
    assert student_scores.numel() == teacher_scores.numel() == 2
    student_scores.sum().backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_icd_full_loss_is_finite_and_projector_receives_gradient():
    loss_fn = ICDCompactnessLoss(
        student_dim=4, teacher_dim=6, num_classes=3, mode="full",
        bank_size=2, valid_steps=5, delta=0.1, gamma=10,
        sdc_start_epoch=1, sdc_weight=0.5, classification_weight=0.1,
    )
    student = torch.randn(6, 4, requires_grad=True)
    teacher = torch.randn(6, 6)
    inference_logits = torch.randn(6, 3, requires_grad=True)
    classification_logits = torch.randn(6, 3, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    total, breakdown = loss_fn(
        inference_logits=inference_logits,
        classification_logits=classification_logits,
        student_embeddings=student,
        teacher_embeddings=teacher,
        targets=labels,
        epoch=1,
    )
    assert torch.isfinite(total)
    assert breakdown["icd_positive_pairs"] > 0
    assert breakdown["icd_sdc_active"] == 1.0
    total.backward()
    assert loss_fn.projector.weight.grad is not None
    assert torch.isfinite(loss_fn.projector.weight.grad).all()


def test_icd_state_dict_restores_projector_and_banks_exactly():
    first = ICDCompactnessLoss(
        student_dim=2, teacher_dim=3, num_classes=2, mode="full",
        bank_size=2, valid_steps=5, delta=0.2, gamma=5, sdc_start_epoch=1,
    )
    first.bank.update(torch.randn(2, 2), torch.randn(2, 3), torch.tensor([0, 1]))
    state = first.state_dict()
    second = ICDCompactnessLoss(
        student_dim=2, teacher_dim=3, num_classes=2, mode="full",
        bank_size=2, valid_steps=5, delta=0.2, gamma=5, sdc_start_epoch=1,
    )
    second.load_state_dict(state, strict=True)
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key]), key


def test_sdc_schedule_and_fcd_ablation_are_locked():
    kwargs = dict(
        student_dim=2, teacher_dim=3, num_classes=2, bank_size=2,
        valid_steps=5, delta=0.2, gamma=5, sdc_start_epoch=2,
    )
    inputs = dict(
        inference_logits=torch.randn(2, 2),
        classification_logits=torch.randn(2, 2, requires_grad=True),
        student_embeddings=torch.randn(2, 2, requires_grad=True),
        teacher_embeddings=torch.randn(2, 3),
        targets=torch.tensor([0, 1]),
    )
    full = ICDCompactnessLoss(mode="full", **kwargs)
    _, before = full(**inputs, epoch=1)
    _, after = full(**inputs, epoch=2)
    fcd = ICDCompactnessLoss(mode="fcd", **kwargs)
    _, ablation = fcd(**inputs, epoch=99)
    assert before["icd_sdc_active"] == 0.0
    assert after["icd_sdc_active"] == 1.0
    assert ablation["icd_sdc_active"] == 0.0
