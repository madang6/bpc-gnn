"""Annotation alignment and demonstration-level splits for learned retrieval."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def annotation_windows(annotation, length, history, horizon):
    """Ordered segment bundles overlapping [start+history, start+history+horizon).

    Labels describe recorded candidate futures. They must never be supplied as
    query inputs. State offset one leaves length-history-horizon valid columns.
    """
    segments = annotation['segments']
    cursor = 0
    for segment in segments:
        if segment['start_step'] != cursor or segment['end_step_exclusive'] <= cursor:
            raise ValueError('annotations must be contiguous, nonempty intervals')
        cursor = segment['end_step_exclusive']
        if not isinstance(segment['caption'], str) or not segment['caption'].strip():
            raise ValueError('empty annotation caption')
    if cursor != length or annotation['steps'] != length:
        raise ValueError('annotation length differs from demonstration')
    count = length - history - horizon
    if count <= 0:
        raise ValueError('demonstration too short for a complete BPC window')
    starts = np.arange(count) + history
    first = np.searchsorted([s['end_step_exclusive'] for s in segments], starts, side='right')
    last = np.searchsorted([s['start_step'] for s in segments], starts + horizon, side='left')
    bundles, texts, ids = {}, [], []
    for begin, end in zip(first, last):
        key = (int(begin), int(end))
        if key not in bundles:
            bundles[key] = len(texts)
            texts.append(' Then '.join(
                f"{s['primitive']}: {s['caption']}" for s in segments[begin:end]))
        ids.append(bundles[key])
    return texts, np.asarray(ids, dtype=np.int64)


def demo_split(count, seed, train_fraction=.9, validation_fraction=.05):
    if count < 3 or not 0 < train_fraction < 1 or not 0 < validation_fraction < 1-train_fraction:
        raise ValueError('need positive train, validation and test partitions')
    order = np.random.default_rng(seed).permutation(count)
    train = max(1, min(count-2, int(count * train_fraction)))
    valid = max(1, min(count-train-1, int(count * validation_fraction)))
    labels = np.full(count, 'test', dtype='<U10')
    labels[order[:train]], labels[order[train:train+valid]] = 'train', 'validation'
    return labels.tolist()


def prepare_index(bank_dir, annotation_dir, *, history=10, horizon=10, seed=20261104):
    bank_dir, annotation_dir = Path(bank_dir), Path(annotation_dir)
    instructions = json.loads((annotation_dir / 'manifest.json').read_text())['tasks']
    texts, text_index, demos, sources = [], {}, [], []
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(instructions=instructions, history=history,
                                  horizon=horizon, state_offset=1, split_seed=seed),
                             sort_keys=True).encode())

    def intern(text):
        if text not in text_index:
            text_index[text] = len(texts)
            texts.append(text)
        return text_index[text]

    for task_index, task in enumerate(sorted(instructions)):
        path = bank_dir / f'{task}.pt'
        bank = torch.load(str(path), map_location='cpu', weights_only=False, mmap=True)
        split = demo_split(len(bank['data']), seed + task_index)
        instruction_id = intern(instructions[task])
        source = dict(task=task, bytes=path.stat().st_size,
                      metadata=bank['metadata'], demos=len(bank['data']))
        sources.append(source)
        digest.update(json.dumps(source, sort_keys=True).encode())
        for number, trajectory in enumerate(bank['data']):
            demo = str(trajectory['demo_id'])
            demo = demo if demo.startswith('demo_') else 'demo_' + demo
            annotation = json.loads((annotation_dir / 'episodes' / task / demo / 'annotations.json').read_text())
            length = trajectory['Uro'].shape[1]
            bundles, window_ids = annotation_windows(annotation, length, history, horizon)
            mapping = np.array([intern(text) for text in bundles])
            aligned = mapping[window_ids]
            identity = dict(task=task, demo=demo, steps=length, segments=annotation['segments'])
            digest.update(json.dumps(identity, sort_keys=True).encode())
            demos.append(dict(task=task, task_index=task_index, bank_index=number, demo=demo,
                              length=length, split=split[number], goal_text_id=instruction_id,
                              continuation_text_ids=torch.from_numpy(aligned)))
        print(json.dumps(dict(stage='annotation_index', task=task, demos=len(bank['data']),
                              unique_texts=len(texts))), flush=True)
    return dict(version=1, history=history, horizon=horizon, state_offset=1,
                seed=seed, texts=texts, demonstrations=demos, sources=sources,
                annotation_identity=digest.hexdigest(), instructions=instructions)


@torch.no_grad()
def embed_texts(texts, model_name, *, device='cpu', batch_size=128, revision=None):
    """Encode text with masked mean pooling and unit-norm normalization."""
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModel.from_pretrained(model_name, revision=revision).to(device).eval()
    embeddings = []
    truncated = 0
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[start:start+batch_size], padding=True, truncation=False)
        truncated += sum(len(row) > 128 for row in tokens['input_ids'])
        tokens = tokenizer(texts[start:start+batch_size], padding=True, truncation=True,
                           max_length=128, return_tensors='pt').to(device)
        output = model(**tokens).last_hidden_state.float()
        mask = tokens['attention_mask'][..., None]
        pooled = (output * mask).sum(1) / mask.sum(1).clamp_min(1)
        embeddings.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
        if start % (batch_size * 20) == 0:
            print(json.dumps(dict(stage='text_embeddings', done=start, total=len(texts))), flush=True)
    return torch.cat(embeddings), dict(model=model_name, revision=model.config._commit_hash,
        pooling='attention_mask_mean_l2', max_length=128, truncated_texts=truncated)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Align segment labels and embed their text.')
    for name in ['banks', 'annotations', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=20261104)
    parser.add_argument('--revision', default='4ca70771034acceecb2e72475f72050fcdde4ddc')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new annotation-cache output path')
    index = prepare_index(args.banks, args.annotations, seed=args.seed)
    index['text_embeddings'], index['text_encoder'] = embed_texts(
        index['texts'], 'sentence-transformers/paraphrase-MiniLM-L3-v2',
        device=args.device, revision=args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    torch.save(index, temporary)
    temporary.replace(args.output)


if __name__ == '__main__':
    main()
