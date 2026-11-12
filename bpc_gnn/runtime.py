import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path

import torch

from .bpc_adapter import BPC_ROOT, FisherScoreReference
from .dataset import DemonstrationBank

ROOT = Path(__file__).resolve().parents[1]


def argument_parser(description):
    parser = argparse.ArgumentParser(description=description)
    for name in ['banks', 'annotations', 'fisher', 'fisher-cache']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--base-config', type=Path, default=ROOT/'configs/bpc.json')
    parser.add_argument('--device', default='cuda:0')
    return parser


def load_bank(args, config):
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'], device)
    base_config = json.loads(args.base_config.read_text())
    bank = DemonstrationBank(args.banks, args.annotations, base_config, device=device)
    reference = FisherScoreReference(bank, args.fisher, args.fisher_cache)
    return bank, reference


def precision(bank, config):
    if config['mixed_precision'] and bank.device.type == 'cuda':
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return nullcontext()


def identity(bank, reference):
    source = {}
    for directory in [ROOT/'bpc_gnn', BPC_ROOT/'src']:
        for path in sorted(directory.rglob('*')):
            if path.suffix in ['.py', '.cpp', '.cu']:
                name = directory.name+'/'+str(path.relative_to(directory))
                source[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(data=bank.identity, fisher=reference.fisher_sha256,
                base_config=bank.base_config, source=source,
                torch=torch.__version__, cuda=torch.version.cuda)


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def validate_config(config, bank):
    positive = ['steps', 'bank_chunk_size', 'checkpoint_every', 'validation_every',
                'validation_per_task', 'probe_queries_per_task', 'queries_per_update',
                'evaluation_samples', 'learning_rate', 'temperature', 'gradient_clip']
    if any(not math.isfinite(config[name]) or config[name] <= 0 for name in positive):
        raise ValueError('Training settings must be finite and positive')
    if config['samples_per_query'] < 2:
        raise ValueError('Positive training settings and at least two support samples are required')
    if config['queries_per_update'] % len(bank.tasks):
        raise ValueError('Each batch must contain equally many queries from every task')
    policy = bank.policy
    if config['model']['history_dim'] != policy.history*(policy.state_dim+policy.action_dim):
        raise ValueError('History dimension differs from the bank')
    if config['model']['text_dim'] != bank.text.shape[1]:
        raise ValueError('Text dimension differs from the annotation embeddings')
    if any(not math.isfinite(value) or value <= 0 for value in config['reward_scales'].values()):
        raise ValueError('Reward normalization scales must be positive')
