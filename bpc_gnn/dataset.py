from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .bpc_adapter import Policy


@dataclass
class Query:
    history: torch.Tensor
    instruction: torch.Tensor
    future: torch.Tensor
    task: str
    demo: str
    step: int
    exclude_demo: int | None = None


class DemonstrationBank:
    def __init__(self, bank_dir, index_path, base_config, *, device='cuda'):
        self.index = torch.load(index_path, map_location='cpu', weights_only=False)
        if self.index['version'] != 1 or self.index['state_offset'] != 1:
            raise ValueError('unsupported annotation index')
        self.device = torch.device(device)
        self.base_config = base_config
        identity = hashlib.sha256()
        identity.update(self.index['annotation_identity'].encode())
        identity.update(json.dumps(dict(seed=self.index['seed'], instructions=self.index['instructions'],
            encoder=self.index['text_encoder'], split=[(d['task'], d['demo'], d['split'])
            for d in self.index['demonstrations']]), sort_keys=True).encode())
        identity.update(self.index['text_embeddings'].contiguous().numpy().tobytes())
        self.identity = identity.hexdigest()
        self.text = self.index['text_embeddings'].to(self.device)
        self.train_meta, self.heldout = [], {'validation': [], 'test': []}
        self.task_ranges, trajectories, goal_ids, continuation_ids = {}, [], [], []
        self.column_demo_starts, offset = [], 0
        for task in sorted(self.index['instructions']):
            data = torch.load(str(Path(bank_dir) / f'{task}.pt'), map_location='cpu',
                              weights_only=False, mmap=True)['data']
            first = offset
            records = [d for d in self.index['demonstrations'] if d['task'] == task]
            for meta in records:
                trajectory = data[meta['bank_index']]
                demo = str(trajectory['demo_id'])
                demo = demo if demo.startswith('demo_') else 'demo_' + demo
                if demo != meta['demo'] or trajectory['Uro'].shape[1] != meta['length']:
                    raise ValueError('bank and annotation cache identities do not match')
                if meta['split'] != 'train':
                    self.heldout[meta['split']].append((meta, trajectory))
                    continue
                self.column_demo_starts.append(offset)
                self.train_meta.append(meta)
                trajectories.append(trajectory)
                n = len(meta['continuation_text_ids'])
                goal_ids.append(torch.full((n,), meta['goal_text_id'], dtype=torch.long))
                continuation_ids.append(meta['continuation_text_ids'])
                offset += n
            self.task_ranges[task] = (first, offset)
            print(json.dumps(dict(stage='load_bank', task=task, training_columns=offset-first)), flush=True)
        config = deepcopy(base_config)
        method = config['method']
        method.update(device=str(device), history=self.index['history'])
        method['spc']['horizon'] = self.index['horizon']
        method['g'].update(lookup='action_ridge', lookup_rates=[1], state_offset=1,
            align_history=False, align_future=False, rate_scaled_action_dims=[],
            temporal_support_screen=False)
        method['residual']['action_source'] = 'native'
        self.policy = Policy(trajectories, config['data'], method, fit_residuals=False)
        self.goal_ids = torch.cat(goal_ids).to(self.device)
        self.continuation_ids = torch.cat(continuation_ids).to(self.device)
        self.columns = torch.arange(len(self.policy.hankel_starts), device=self.device)
        if len(self.columns) != offset:
            raise ValueError('annotation windows do not match BPC Hankel columns')
        self.column_demo_starts = np.asarray(self.column_demo_starts)
        self.tasks = sorted(self.task_ranges)

    def inputs(self, columns):
        return (self.policy._past_columns(columns, dtype=torch.float32),
                self.text[self.goal_ids[columns]], self.text[self.continuation_ids[columns]])

    def training_query(self, task_index, rng):
        task = self.tasks[task_index % len(self.tasks)]
        first, stop = self.task_ranges[task]
        col = int(rng.integers(first, stop))
        return self.training_query_at(col)

    def training_query_at(self, col):
        demo = int(np.searchsorted(self.column_demo_starts, col, side='right') - 1)
        columns = self.columns[col:col+1]
        history = self.policy._past_columns(columns)
        instruction = self.text[self.goal_ids[columns]]
        meta = self.train_meta[demo]
        return Query(history, instruction, self.policy.uf[:, col][None], meta['task'],
                     meta['demo'], col - int(self.column_demo_starts[demo]) + self.policy.history, demo)

    def fixed_training_queries(self, per_task, seed):
        """One history from each of distinct, reproducibly chosen demonstrations."""
        if per_task < 1:
            raise ValueError('positive fixed query count required')
        rng, output = np.random.default_rng(seed), []
        for task in self.tasks:
            demos = [i for i, meta in enumerate(self.train_meta) if meta['task'] == task]
            if per_task > len(demos):
                raise ValueError('not enough distinct training demonstrations')
            for demo in rng.choice(demos, per_task, replace=False):
                first = self.column_demo_starts[demo]
                stop = (self.column_demo_starts[demo+1] if demo+1 < len(self.train_meta)
                        else len(self.columns))
                output.append(self.training_query_at(int(rng.integers(first, stop))))
        return output

    def heldout_queries(self, split, per_task, seed):
        rng, output = np.random.default_rng(seed), []
        policy = self.policy
        for task in self.tasks:
            items = [item for item in self.heldout[split] if item[0]['task'] == task]
            if per_task > len(items):
                raise ValueError('not enough held-out demonstrations')
            for index in rng.choice(len(items), per_task, replace=False):
                meta, trajectory = items[index]
                t = int(rng.integers(policy.history, meta['length'] - policy.horizon))
                states = np.concatenate([trajectory[key] for key in
                                         ['Xro', 'r3m_agent', 'r3m_wrist']], 0).T
                states = torch.as_tensor(states[t-policy.history+1:t+1], device=self.device, dtype=policy.dtype)
                actions = torch.as_tensor(trajectory['Uro'][:, t-policy.history:t].T,
                                          device=self.device, dtype=policy.dtype)
                future = torch.as_tensor(trajectory['Uro'][:, t:t+policy.horizon].T,
                                         device=self.device, dtype=policy.dtype)
                history = torch.cat([((actions-policy.action_mean)/policy.action_scale).flatten(),
                    ((states-policy.state_mean)/policy.state_scale).flatten()])[None]
                future = ((future-policy.action_mean)/policy.action_scale).flatten()[None]
                output.append(Query(history, self.text[meta['goal_text_id']][None], future,
                                    task, meta['demo'], t))
        return output

    def eligible(self, query):
        if query.exclude_demo is None:
            return self.columns
        return self.columns[self.policy.hankel_trajectory != query.exclude_demo]
