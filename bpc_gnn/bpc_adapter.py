import os
from pathlib import Path
import sys

import torch

BPC_COMMIT = '4852cbb9bdd19f409c56de0f3b948ef6d01aad59'
BPC_ROOT = Path(os.environ.get('BPC_ROOT', Path(__file__).resolve().parents[1]/'vendor/bpc'))
if not (BPC_ROOT/'src/dpc_nldl/bpc.py').exists():
    raise ImportError('Initialize vendor/bpc with git submodule update --init, or set BPC_ROOT.')
sys.path.insert(0, str(BPC_ROOT/'src'))

from dpc_nldl.bpc import Policy
from dpc_nldl.fisher_summary_reference import FisherScoreReference
from dpc_nldl.sparse_qp import sparsemax_support


@torch.no_grad()
def continuation_errors(bank, query, columns):
    policy = bank.policy
    histories = policy._past_columns(columns[None])
    prediction, coefficients = policy._g_prior(
        histories, columns[None], query.history.to(policy.dtype))
    reconstruction = torch.einsum('bkh,bk->bh', histories, coefficients)
    action_error = (prediction-query.future).square().mean()
    history_error = (reconstruction-query.history).square().mean()
    coefficient_norm = coefficients.abs().sum()
    if not bool(torch.isfinite(torch.stack([action_error, history_error, coefficient_norm])).all()):
        raise FloatingPointError('Nonfinite affine continuation')
    if abs(float(coefficients.sum())-1) > 1e-6:
        raise FloatingPointError('Affine coefficients must sum to one')
    return action_error, history_error, coefficient_norm


def score_candidates(model, reference, query, keys, *, checkpoint_graph=False):
    eligible, fisher_scores = reference(query)
    correction = model(query.history, query.instruction, keys[eligible],
                       checkpoint_graph=checkpoint_graph)
    return eligible, fisher_scores + correction.double()
