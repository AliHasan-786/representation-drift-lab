from __future__ import annotations

import torch
import torch.nn.functional as functional


def class_positive_mask(labels: torch.Tensor) -> torch.Tensor:
    """Return all same-class pairs, including duplicate-caption positives."""
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    return labels[:, None].eq(labels[None, :])


def _multi_positive_directional_loss(
    logits: torch.Tensor, positive_mask: torch.Tensor
) -> torch.Tensor:
    if logits.shape != positive_mask.shape:
        raise ValueError("logits and positive_mask must have identical shapes")
    if not positive_mask.any(dim=1).all():
        raise ValueError("every anchor must have at least one positive")
    negative_infinity = torch.finfo(logits.dtype).min
    positive_logits = logits.masked_fill(~positive_mask, negative_infinity)
    numerator = torch.logsumexp(positive_logits, dim=1)
    denominator = torch.logsumexp(logits, dim=1)
    return (denominator - numerator).mean()


def multi_positive_clip_loss(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric class-aware contrastive loss.

    Unlike diagonal-only CLIP loss, all samples sharing a class caption are
    positives. This prevents duplicate class captions from becoming false
    negatives within the batch.
    """
    if image_embeddings.shape != text_embeddings.shape:
        raise ValueError("image and text embeddings must have identical shapes")
    if image_embeddings.ndim != 2 or image_embeddings.shape[0] != labels.numel():
        raise ValueError("embeddings must be [batch, dim] and match labels")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    image_embeddings = functional.normalize(image_embeddings, dim=-1)
    text_embeddings = functional.normalize(text_embeddings, dim=-1)
    logits = image_embeddings @ text_embeddings.T / temperature
    mask = class_positive_mask(labels)
    image_to_text = _multi_positive_directional_loss(logits, mask)
    text_to_image = _multi_positive_directional_loss(logits.T, mask.T)
    return 0.5 * (image_to_text + text_to_image)


def orthogonalize_gradient(
    primary: torch.Tensor, constraint: torch.Tensor, *, epsilon: float = 1e-12
) -> torch.Tensor:
    """Remove the component of a gradient aligned with a constraint gradient."""
    if primary.shape != constraint.shape:
        raise ValueError("primary and constraint gradients must have identical shapes")
    denominator = torch.sum(constraint * constraint)
    if float(denominator.detach().cpu()) <= epsilon:
        return primary.clone()
    coefficient = torch.sum(primary * constraint) / denominator
    return primary - coefficient * constraint
