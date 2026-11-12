from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class ModelConfig:
    history_dim: int = 10400
    text_dim: int = 384
    hidden_dim: int = 256
    width: int = 128
    heads: int = 2
    summaries: int = 32


class AttentionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        if width % heads:
            raise ValueError('Attention width must be divisible by head count')
        self.heads = heads
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.out = nn.Linear(width, width)
        self.ln1 = nn.LayerNorm(width)
        self.ln2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width))

    def split(self, values):
        return values.reshape(1, -1, self.heads, values.shape[-1]//self.heads).transpose(1, 2)

    def attend(self, residual, queries, keys, values):
        attended = F.scaled_dot_product_attention(queries, keys, values)
        attended = attended.transpose(1, 2).reshape(residual.shape)
        hidden = self.ln1(residual + self.out(attended))
        return self.ln2(hidden + self.ff(hidden))


class GraphRetriever(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.history = nn.Sequential(
            nn.Linear(config.history_dim, config.hidden_dim), nn.GELU(),
            nn.Linear(config.hidden_dim, config.width), nn.LayerNorm(config.width))
        self.goal = nn.Linear(config.text_dim, config.width)
        self.continuation = nn.Linear(config.text_dim, config.width, bias=False)
        self.seeds = nn.Parameter(torch.randn(config.summaries, config.width)/config.width**.5)
        self.condition = nn.Linear(config.width, config.width)
        self.gather = AttentionBlock(config.width, config.heads)
        self.broadcast = AttentionBlock(config.width, config.heads)
        self.score = nn.Linear(config.width, 1)
        self.register_buffer('support_scale', torch.tensor(1.))
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def query(self, history, instruction):
        return (self.history(history.to(self.history[0].weight.dtype))
                + self.goal(instruction.to(self.goal.weight.dtype)))

    def candidate(self, history, instruction, continuation):
        return (self.query(history, instruction)
                + self.continuation(continuation.to(self.continuation.weight.dtype)))

    def forward(self, history, instruction, candidates, *, checkpoint_graph=False):
        query = self.query(history, instruction)
        seeds = self.seeds + self.condition(query)
        summaries = self.gather.attend(
            seeds, self.gather.split(self.gather.q(seeds)),
            self.gather.split(self.gather.k(candidates)),
            self.gather.split(self.gather.v(candidates)))
        candidate_queries = self.broadcast.split(self.broadcast.q(candidates))

        def broadcast_and_score(candidates, candidate_queries, summaries, query):
            context = self.broadcast.attend(
                candidates, candidate_queries,
                self.broadcast.split(self.broadcast.k(summaries)),
                self.broadcast.split(self.broadcast.v(summaries)))
            # FP32 scores avoid BF16 ties at the sparsemax boundary.
            with torch.autocast(device_type=candidates.device.type, enabled=False):
                dtype = self.score.weight.dtype
                return self.score(context.to(dtype)*query.to(dtype)).flatten()

        arguments = (candidates, candidate_queries, summaries, query)
        if checkpoint_graph and torch.is_grad_enabled():
            return checkpoint(broadcast_and_score, *arguments, use_reentrant=False)
        return broadcast_and_score(*arguments)


@torch.no_grad()
def encode_bank(model, bank, chunk_size):
    keys = torch.empty(len(bank.columns), model.config.width, device=bank.device,
                       dtype=next(model.parameters()).dtype)
    for start in range(0, len(bank.columns), chunk_size):
        columns = bank.columns[start:start+chunk_size]
        keys[start:start+len(columns)] = model.candidate(*bank.inputs(columns)).to(keys.dtype)
    return keys.detach().requires_grad_(True)


def replay_bank_gradients(model, bank, gradient, chunk_size):
    # Recompute encoder activations after accumulating all query gradients.
    for start in range(0, len(bank.columns), chunk_size):
        columns = bank.columns[start:start+chunk_size]
        encoded = model.candidate(*bank.inputs(columns))
        torch.autograd.backward(encoded, gradient[start:start+len(columns)].to(encoded.dtype))
