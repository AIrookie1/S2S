"""Geometry and exact suffix replay for intervention/decision diagnostics."""
from __future__ import annotations

from contextlib import contextmanager
import torch
import torch.nn.functional as F

from methods.S2S_R.evaluate_S2S_R import adaptive_3way_weights, two_slerp_sym, norm
from methods.spherical_steering.steering import spherical_geometric_logic

# Fixed before evaluation. The same weights apply to every answer of a question.
GRID = ((.8, .1, .1), (.6, .3, .1), (.6, .1, .3),
        (.4, .5, .1), (.4, .1, .5), (.4, .3, .3))


def split_radial(vector, hidden):
    unit = norm(hidden)
    radial = (vector * unit).sum(-1, keepdim=True) * unit
    return radial, vector - radial


def equal_budget(vector, hidden, fraction=.01):
    return norm(vector) * hidden.norm(dim=-1, keepdim=True) * fraction


def cosine(x, y):
    return float(F.cosine_similarity(x.float().flatten()[None], y.float().flatten()[None]).item())


def dynamic_segment(hidden, average, residual, kappa, margin, weights=None, gate=None):
    a = norm(hidden)
    b = norm(average).expand_as(a)
    c = norm(residual).expand_as(a)
    alpha, beta, gamma, _ = adaptive_3way_weights(a, b, c, kappa=kappa)
    natural_gate = torch.maximum(beta, gamma) > alpha + margin
    if weights is not None:
        alpha, beta, gamma = [torch.full_like(alpha, w) for w in weights]
    active = natural_gate if gate is None else gate
    rotated = two_slerp_sym(a, b, c, alpha, beta, gamma) * hidden.norm(dim=-1, keepdim=True)
    edited = torch.where(active, rotated, hidden)
    return edited, active, torch.cat([alpha, beta, gamma], -1)


def spherical_segment(hidden, prototypes, hp):
    return torch.stack([spherical_geometric_logic(
        x, prototypes[0], prototypes[1], hp['kappa'], hp['alpha'], hp['beta'])[0]
        for x in hidden])


class SuffixEngine:
    """Replay the exact decoder suffix from an unedited block output.

    Full causal sequence, RoPE positions and attention mask are retained. No KV
    cache is used. Only the already-computed prefix blocks are omitted.
    """
    def __init__(self, model):
        self.model = model
        self.layers = model.model.layers
        self.device = next(model.parameters()).device
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    @contextmanager
    def tail(self, layer):
        self.model.model.layers = torch.nn.ModuleList(list(self.layers)[layer + 1:])
        try:
            yield
        finally:
            self.model.model.layers = self.layers

    def capture(self, ids, layer):
        from baukit import TraceDict
        name = f'layers.{layer}'
        with torch.no_grad(), TraceDict(self.model.model, [name], stop=True) as trace:
            self.model.model(input_ids=ids.to(self.device), use_cache=False)
        hidden = trace[name].output
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return hidden.detach()

    def scores(self, full_hidden, ids, start, layer, mean_score, grad=False):
        positions = torch.arange(start, ids.shape[1] - 1, device=self.device)
        if len(positions) == 0:
            raise ValueError('Empty answer')
        with self.tail(layer):
            out = self.model(inputs_embeds=full_hidden.to(self.model.dtype),
                             use_cache=False)
        logits = out.logits[:, positions].float()
        labels = ids.to(self.device)[:, start+1:].expand(logits.shape[0], -1)
        lp = logits.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
        score = lp.mean(-1) if mean_score else lp.sum(-1)
        if grad:
            gradient, = torch.autograd.grad(score.sum(), full_hidden)
            return score.detach(), gradient.detach()
        return score.detach()

    def score_edits(self, full_hidden, ids, start, layer, mean_score, edits, batch_size=4):
        result = {}
        names = list(edits)
        with torch.no_grad():
            for offset in range(0, len(names), batch_size):
                selected = names[offset:offset+batch_size]
                h = full_hidden.expand(len(selected), -1, -1).clone()
                for i, name in enumerate(selected):
                    h[i, start:] = edits[name].to(h.dtype)
                scores = self.scores(h, ids, start, layer, mean_score)
                result.update({name: float(score) for name, score in zip(selected, scores.cpu())})
        return result


def margin(scores, correct, negatives):
    return float(scores[correct] - max(scores[j] for j in negatives))
