import torch

from scripts.evaluate_prototype_classifier import (
    accumulate_class_prototypes,
    prototype_logits,
    selection_key,
)


def test_prototypes_are_normalized_class_means():
    embeddings = torch.tensor([
        [1.0, 0.0],
        [0.8, 0.2],
        [0.0, 1.0],
        [0.2, 0.8],
    ])
    labels = torch.tensor([0, 0, 1, 1])
    prototypes, counts = accumulate_class_prototypes(
        embeddings, labels, num_classes=2
    )
    assert counts.tolist() == [2, 2]
    assert torch.allclose(prototypes.norm(dim=1), torch.ones(2), atol=1e-6)
    logits = prototype_logits(embeddings, prototypes, scale=64.0)
    assert logits.argmax(1).tolist() == labels.tolist()


def test_missing_class_is_rejected():
    try:
        accumulate_class_prototypes(
            torch.randn(2, 3), torch.tensor([0, 0]), num_classes=2
        )
    except ValueError as error:
        assert "missing classes" in str(error)
    else:
        raise AssertionError("missing prototype class must be rejected")


def test_selection_is_lexicographic():
    baseline = {
        "errors": 0,
        "ordinary_ce_loss": 0.2,
        "mean_true_class_margin": 3.0,
    }
    lower_loss = {
        "errors": 0,
        "ordinary_ce_loss": 0.1,
        "mean_true_class_margin": 2.0,
    }
    extra_error = {
        "errors": 1,
        "ordinary_ce_loss": 0.01,
        "mean_true_class_margin": 10.0,
    }
    assert selection_key(lower_loss) < selection_key(baseline)
    assert selection_key(extra_error) > selection_key(baseline)
