import json
from pathlib import Path

import numpy as np
import torch

from .bpc_adapter import continuation_errors, score_candidates
from .model import GraphRetriever, ModelConfig, encode_bank
from .runtime import argument_parser, load_bank, precision, write_json
from .selection import adaptive_support, stochastic_logits, NonemptyBernoulli


@torch.no_grad()
def evaluate(model, bank, reference, keys, queries, config, sampling_seed):
    generator = torch.Generator(device=bank.device).manual_seed(sampling_seed)
    rows = []
    for query in queries:
        with precision(bank, config):
            eligible, scores = score_candidates(model, reference, query, keys)
        selected = eligible[adaptive_support(scores)]
        action, history, coefficient_norm = continuation_errors(bank, query, selected)
        distribution = NonemptyBernoulli(stochastic_logits(scores, 1., config['temperature']))
        samples = distribution.sample(config['evaluation_samples'], generator)
        errors = [continuation_errors(bank, query, eligible[mask]) for mask in samples]
        rows.append(dict(task=query.task, demo=query.demo, step=query.step,
            action_mse=float(action), history_mse=float(history), support=len(selected),
            coefficient_l1=float(coefficient_norm),
            sampled_action_mse=float(torch.stack([error[0] for error in errors]).mean()),
            sampled_history_mse=float(torch.stack([error[1] for error in errors]).mean()),
            sampled_support=float(samples.sum(1).double().mean())))
    metrics = ['action_mse', 'history_mse', 'sampled_action_mse', 'sampled_history_mse',
               'support', 'sampled_support', 'coefficient_l1']
    per_task = {task: {metric: float(np.mean([row[metric] for row in rows if row['task']==task]))
                      for metric in metrics} for task in sorted({row['task'] for row in rows})}
    return dict(mean={metric: float(np.mean([row[metric] for row in rows])) for metric in metrics},
                per_task=per_task, rows=rows)


def main():
    parser = argument_parser('Evaluate deterministic and sampled GNN retrieval.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--queries-per-task', type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new evaluation output path')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = saved['config']
    bank, reference = load_bank(args, config)
    expected = saved['identity']
    if (expected['data'] != bank.identity or expected['fisher'] != reference.fisher_sha256
            or expected['base_config'] != bank.base_config):
        raise ValueError('Checkpoint and evaluation assets differ')
    model = GraphRetriever(ModelConfig(**config['model'])).to(bank.device)
    model.load_state_dict(saved['model'])
    model.eval()
    with precision(bank, config):
        keys = encode_bank(model, bank, config['bank_chunk_size']).detach()
    count = config['validation_per_task'] if args.queries_per_task is None else args.queries_per_task
    if count < 1:
        parser.error('Choose a positive number of queries per task')
    queries = bank.heldout_queries(args.split, count, config['validation_query_seed'])
    result = evaluate(model, bank, reference, keys, queries, config, config['validation_sampling_seed'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, dict(checkpoint=str(args.checkpoint), step=saved['step'],
                                 split=args.split, **result))
    print(json.dumps(result['mean'], indent=2))


if __name__ == '__main__':
    main()
