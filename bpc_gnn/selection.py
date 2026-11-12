import torch
from torch.nn import functional as F

from .bpc_adapter import sparsemax_support


def adaptive_support(scores, scale=1.):
    """Exact existing sparsemax selection, without a maximum support size."""
    if scores.ndim != 1 or not len(scores):
        raise ValueError('expected a nonempty vector of scores')
    def topk(count, rows):
        values, indices = scores.topk(count)
        return indices[None], values[None]
    return sparsemax_support(topk, len(scores), float(scale))[0][0]


def stochastic_logits(scores, scale, temperature):
    """Center logits at the sparsemax threshold, retaining its piecewise gradient."""
    if temperature <= 0:
        raise ValueError('positive exploration temperature required')
    with torch.no_grad():
        active = adaptive_support(scores, scale)
    scaled = scores * scale
    threshold = ((scaled[active].double().sum() - 1) / len(active)).to(scores.dtype)
    return (scaled - threshold) / temperature


class NonemptyBernoulli:
    """Independent inclusion gates conditioned on a nonempty support."""
    def __init__(self, logits):
        if logits.ndim != 1 or not len(logits) or not bool(torch.isfinite(logits).all()):
            raise ValueError('nonempty finite vector of inclusion logits required')
        self.logits = logits

    @torch.no_grad()
    def sample(self, count, generator=None):
        if count < 1:
            raise ValueError('positive sample count required')
        logits = self.logits.double()
        failure = F.logsigmoid(-logits)
        prefix = F.pad(failure.cumsum(0)[:-1], (1, 0))
        first_weights = torch.softmax(F.logsigmoid(logits) + prefix, dim=0)
        first = torch.multinomial(first_weights, count, replacement=True, generator=generator)
        # FP32 uniform zeros can admit vanishingly unlikely columns when the
        # bank contains millions of gates. Match the probability precision.
        selected = torch.rand(count, len(logits), device=logits.device, dtype=logits.dtype,
                              generator=generator) < logits.sigmoid()
        selected &= torch.arange(len(logits), device=logits.device)[None] > first[:, None]
        selected.scatter_(1, first[:, None], True)
        return selected

    def log_prob(self, selected):
        if selected.ndim != 2 or selected.shape[1] != len(self.logits) or selected.dtype != torch.bool:
            raise ValueError('support samples must be a boolean [samples, candidates] matrix')
        logits = self.logits.double()
        failure, success = F.logsigmoid(-logits), F.logsigmoid(logits)
        # Sum the mutually exclusive first-success events in log space. This
        # remains finite even if all raw inclusion probabilities underflow.
        prefix = F.pad(failure.cumsum(0)[:-1], (1, 0))
        log_normalizer = torch.logsumexp(success + prefix, dim=0)
        log_prob = torch.where(selected, success, failure).sum(-1) - log_normalizer
        return log_prob.masked_fill(~selected.any(-1), -torch.inf)


def reinforce_loss(log_probability, rewards):
    """Policy gradient with a leave-one-out reward baseline."""
    if log_probability.shape != rewards.shape or rewards.ndim != 1 or len(rewards) < 2:
        raise ValueError('at least two aligned samples required for leave-one-out baseline')
    rewards = rewards.detach()
    baseline = (rewards.sum() - rewards) / (len(rewards) - 1)
    return -((rewards - baseline) * log_probability).mean()


