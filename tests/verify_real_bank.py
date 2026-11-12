"""Compare both implementations using a fitted checkpoint and the real full bank."""
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from bpc_gnn.bpc_adapter import score_candidates, continuation_errors
from bpc_gnn.dataset import DemonstrationBank
from bpc_gnn.model import GraphRetriever, ModelConfig, encode_bank
from bpc_gnn.runtime import argument_parser, load_bank, precision, write_json
from bpc_gnn.selection import adaptive_support, NonemptyBernoulli, stochastic_logits
from dpc_nldl.summary_training import AnnotatedBank
from dpc_nldl.summary_retrieval import SummaryConfig
from dpc_nldl.summary_fisher_residual import FisherResidualRetriever


def main():
    parser = argument_parser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=16)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = saved['config']
    if config['model'].get('architecture') != 'bank_summary':
        parser.error('Use a parent-repository bank_summary checkpoint')
    config['mixed_precision'] = args.device.startswith('cuda')
    bank, reference = load_bank(args, config)
    torch.set_num_threads(args.threads)
    original_bank = AnnotatedBank(args.banks, args.annotations, bank.base_config, device=args.device)
    assert original_bank.identity == bank.identity
    queries = bank.fixed_training_queries(1, 20261113) + bank.heldout_queries('validation', 1, 20261114)
    previous_queries = (original_bank.fixed_training_queries(1, 20261113)
                        + original_bank.heldout_queries('validation', 1, 20261114))
    for actual, expected in zip(queries, previous_queries):
        assert (actual.task, actual.demo, actual.step) == (expected.task, expected.demo, expected.step)
        for name in ['history', 'instruction', 'future']:
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
    del original_bank, previous_queries
    gc.collect()
    model_config = {name: config['model'][name] for name in ModelConfig.__dataclass_fields__}
    model = GraphRetriever(ModelConfig(**model_config)).to(bank.device).eval()
    model.load_state_dict(saved['model'])
    original = FisherResidualRetriever(SummaryConfig(**config['model']), reference).to(bank.device).eval()
    original.load_state_dict(saved['model'])
    with torch.no_grad(), precision(bank, config):
        keys = encode_bank(model, bank, config['bank_chunk_size']).detach()
        for start in range(0, len(bank.columns), config['bank_chunk_size']):
            columns = bank.columns[start:start+config['bank_chunk_size']]
            torch.testing.assert_close(keys[start:start+len(columns)],
                original.candidate(*bank.inputs(columns)).to(keys.dtype), rtol=0, atol=0)
    rows = []
    with torch.no_grad():
        for query in queries:
            with precision(bank, config), original.query_context(query):
                eligible, scores = score_candidates(model, reference, query, keys)
                expected = original(query.history, query.instruction, keys[eligible])
            torch.testing.assert_close(scores, expected, rtol=0, atol=0)
            support = adaptive_support(scores)
            expected_support = adaptive_support(expected)
            assert torch.equal(support, expected_support)
            logits = stochastic_logits(scores, 1., config['temperature'])
            samples = NonemptyBernoulli(logits).sample(16, torch.Generator(device=bank.device).manual_seed(9))
            expected_samples = NonemptyBernoulli(stochastic_logits(expected, 1., config['temperature'])).sample(
                16, torch.Generator(device=bank.device).manual_seed(9))
            assert torch.equal(samples, expected_samples)
            metrics = continuation_errors(bank, query, eligible[support])
            rows.append(dict(task=query.task, demo=query.demo, step=query.step,
                             exclude_demo=query.exclude_demo, support=len(support),
                             action_mse=float(metrics[0]), history_mse=float(metrics[1])))
            print(json.dumps(rows[-1]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, dict(exact_match=True, bank_columns=len(bank.columns),
        queries=len(queries), data_identity=bank.identity, device=args.device, rows=rows))


if __name__ == '__main__':
    main()
