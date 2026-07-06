from __future__ import annotations

import unittest

import torch

from driftlab.losses import (
    class_positive_mask,
    multi_positive_clip_loss,
    orthogonalize_gradient,
)


class LossTests(unittest.TestCase):
    def test_duplicate_captions_are_positives(self) -> None:
        labels = torch.tensor([0, 0, 1, 2])
        mask = class_positive_mask(labels)
        self.assertTrue(mask[0, 1].item())
        self.assertTrue(mask[1, 0].item())
        self.assertFalse(mask[0, 2].item())
        self.assertEqual(mask[0].sum().item(), 2)

    def test_multi_positive_loss_is_finite_and_differentiable(self) -> None:
        image = torch.randn(6, 4, requires_grad=True)
        text = torch.randn(6, 4, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        loss = multi_positive_clip_loss(image, text, labels)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(image.grad)
        self.assertIsNotNone(text.grad)

    def test_gradient_projection_is_orthogonal_to_constraint(self) -> None:
        primary = torch.tensor([2.0, 3.0])
        constraint = torch.tensor([1.0, 0.0])
        projected = orthogonalize_gradient(primary, constraint)
        self.assertTrue(torch.allclose(projected, torch.tensor([0.0, 3.0])))
        self.assertAlmostEqual(float(torch.dot(projected, constraint)), 0.0)

    def test_zero_constraint_leaves_gradient_unchanged(self) -> None:
        primary = torch.tensor([2.0, 3.0])
        projected = orthogonalize_gradient(primary, torch.zeros(2))
        self.assertTrue(torch.equal(primary, projected))


if __name__ == "__main__":
    unittest.main()
