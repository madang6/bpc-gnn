import fcntl
import json
from pathlib import Path

import numpy as np
import torch

from .bpc_adapter import continuation_errors, score_candidates
from .evaluate import evaluate
from .model import GraphRetriever, ModelConfig, encode_bank, replay_bank_gradients
from .runtime import ROOT, argument_parser, identity, load_bank, precision, validate_config, write_json
from .selection import stochastic_logits, NonemptyBernoulli, reinforce_loss


def query_gradient(model, bank, reference, keys, query, config, generator):
    with precision(bank, config):
        eligible, scores = score_candidates(model, reference, query, keys, checkpoint_graph=True)
    distribution = NonemptyBernoulli(stochastic_logits(scores, 1., config['temperature']))
    samples = distribution.sample(config['samples_per_query'], generator)
    errors = [continuation_errors(bank, query, eligible[mask]) for mask in samples]
    action = torch.stack([error[0] for error in errors])
    history = torch.stack([error[1] for error in errors])
    scales = config['reward_scales']
    rewards = -action/scales['action_mse'] - history/scales['reconstruction_mse']
    loss = reinforce_loss(distribution.log_prob(samples), rewards)
    (loss/config['queries_per_update']).backward()
    return dict(task=query.task, demo=query.demo, step=query.step,
                action_mse=float(action.mean()), history_mse=float(history.mean()),
                loss=float(loss.detach()), support=float(samples.sum(1).double().mean()))


def train(bank, reference, config, output, *, resume=False):
    validate_config(config, bank)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return train_locked(bank, reference, config, output, resume)


def train_locked(bank, reference, config, output, resume):
    checkpoint_path = output/'checkpoint.pt'
    metrics_path = output/'metrics.jsonl'
    if not resume and (checkpoint_path.exists() or metrics_path.exists()):
        raise ValueError('Use --resume or choose a new run directory')
    provenance = identity(bank, reference)
    torch.manual_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])
    generator = torch.Generator(device=bank.device).manual_seed(config['seed'])
    model = GraphRetriever(ModelConfig(**config['model'])).to(bank.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.)
    begin = 0
    if resume:
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        previous = {key: value for key, value in saved['config'].items() if key != 'steps'}
        requested = {key: value for key, value in config.items() if key != 'steps'}
        if previous != requested or saved['identity'] != provenance:
            raise ValueError('Resume requires the same configuration, source, runtime and assets')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['numpy_rng']
        generator.set_state(saved['sampling_rng'])
        torch.set_rng_state(saved['torch_rng'])
        begin = saved['step']
        retained = []
        for line in metrics_path.read_text().splitlines(keepends=True):
            if line.endswith('\n') and json.loads(line)['update'] <= begin:
                retained.append(line)
        temporary = metrics_path.with_suffix('.tmp')
        temporary.write_text(''.join(retained))
        temporary.replace(metrics_path)
    if begin > config['steps']:
        raise ValueError('The requested step limit precedes the saved checkpoint')
    write_json(output/'config.json', config)
    write_json(output/'identity.json', provenance)
    validation = bank.heldout_queries('validation', config['validation_per_task'],
                                     config['validation_query_seed'])
    probes = bank.fixed_training_queries(config['probe_queries_per_task'], config['probe_query_seed'])

    def log(record):
        with metrics_path.open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
        print(json.dumps({key: value for key, value in record.items()
                          if key not in ['rows', 'queries', 'per_task']}), flush=True)

    def save(update):
        temporary = checkpoint_path.with_suffix('.tmp')
        torch.save(dict(config=config, identity=provenance, model=model.state_dict(),
            optimizer=optimizer.state_dict(), step=update, numpy_rng=rng.bit_generator.state,
            sampling_rng=generator.get_state(), torch_rng=torch.get_rng_state()), temporary)
        temporary.replace(checkpoint_path)
        write_json(output/'status.json', dict(step=update, complete=update == config['steps']))

    def validate(update):
        with precision(bank, config):
            keys = encode_bank(model, bank, config['bank_chunk_size']).detach()
        for event, queries, seed in [
                ('training_probes', probes, config['probe_sampling_seed']),
                ('validation', validation, config['validation_sampling_seed'])]:
            result = evaluate(model, bank, reference, keys, queries, config, seed)
            log(dict(event=event, update=update, **result))

    if not resume:
        validate(0)
        save(0)
    model.train()
    for update in range(begin+1, config['steps']+1):
        if (output/'STOP').exists():
            save(update-1)
            return model
        optimizer.zero_grad(set_to_none=True)
        with precision(bank, config):
            keys = encode_bank(model, bank, config['bank_chunk_size'])
        queries = [bank.training_query(task, rng) for task in range(config['queries_per_update'])]
        records = [query_gradient(model, bank, reference, keys, query, config, generator)
                   for query in queries]
        if keys.grad is None or not bool(torch.isfinite(keys.grad).all()):
            raise FloatingPointError('Missing or nonfinite bank gradients')
        with precision(bank, config):
            replay_bank_gradients(model, bank, keys.grad, config['bank_chunk_size'])
        if any(parameter.grad is None for parameter in model.parameters()):
            raise RuntimeError('Every model parameter must receive a gradient')
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        del keys
        log(dict(event='train', update=update, queries=records, gradient_norm=float(gradient_norm)))
        if update % config['validation_every'] == 0 or update == config['steps']:
            validate(update)
        if (update % config['checkpoint_every'] == 0 or update % config['validation_every'] == 0
                or update == config['steps']):
            save(update)
    return model


def main():
    parser = argument_parser('Train task-conditioned graph retrieval with BPC prediction rewards.')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/train.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for name in ['steps', 'seed']:
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    bank, reference = load_bank(args, config)
    train(bank, reference, config, args.output, resume=args.resume)


if __name__ == '__main__':
    main()
