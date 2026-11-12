# BPC GNN retrieval

A small implementation of task-conditioned graph retrieval for behavior cloning.
It prepares annotated demonstration windows, trains a summary-node graph neural
network, and evaluates the neighborhoods through BPC's affine continuation.

## Method

Each candidate contains an observation–action history, the demonstration's task
instruction, and a caption describing its recorded continuation. A query contains
its observed history and supplied task instruction. A shared encoder creates
candidate and query vectors. Thirty-two query-conditioned summary nodes gather
messages from the eligible bank and send context back to its candidates.
The network produces one learned score correction per column.

A frozen Fisher retriever supplies the base scores. The correction starts at
zero, reproducing Fisher's initial neighborhood. Training samples a nonempty set
of columns using independent Bernoulli gates around the sparsemax threshold.
BPC reconstructs the query history from this set and continues its future
actions. The negative normalized sum of action and reconstruction MSE supplies
the policy-gradient reward. Deterministic evaluation uses adaptive sparsemax.

Training draws equally many queries from each task. The entire query
demonstration is excluded before graph attention. Chunked encoder evaluation and
gradient replay support the full bank without retaining all encoder activations.
Sampling uses float64 uniforms, including the precision correction from the
parent experiment.

## Read the code

| Module | Responsibility |
|---|---|
| `prepare.py` | Validate segment intervals, align captions, split demonstrations, embed text |
| `dataset.py` | Load compact trajectories, construct histories, sample queries |
| `model.py` | Encoders, gather/broadcast attention, chunked bank encoding |
| `selection.py` | Adaptive support selection and stochastic neighborhood probabilities |
| `bpc_adapter.py` | Frozen Fisher scoring and affine-continuation rewards |
| `train.py` | Policy-gradient updates, validation, checkpointing and resumption |
| `evaluate.py` | Deterministic and sampled errors, with per-task reporting |
| `runtime.py` | Arguments, precision, configuration checks and provenance |

BPC is pinned as a Git submodule at commit
`4852cbb9bdd19f409c56de0f3b948ef6d01aad59`. The dependency supplies the compact
physical bank, normalization, Fisher lookup, sparsemax support routine, and affine
solver. This repository owns data annotation alignment, the GNN, sampling, and
training/evaluation loops. Source-line counts cover `bpc_gnn/*.py`, including
comments and blank lines. Tests, configuration files and the BPC dependency have
separate counts.

## Setup

```bash
git clone --recurse-submodules https://github.com/madang6/bpc-gnn.git
cd bpc-gnn
python -m pip install -r requirements.txt
```

The pinned BPC repository is private, so Git must be authenticated for both
repositories. From an existing checkout, run `git submodule update --init`.
`BPC_ROOT` can point to a checkout of the pinned commit instead of `vendor/bpc`.
Use an appropriate CUDA-enabled PyTorch installation for GPU training. The
original experiments used PyTorch 2.10.0+cu130, Python 3.13, and 96 GiB GPUs.
A single full-bank training worker peaked at about 41 GiB in that campaign.

## Data inputs

Start with cached R3M trajectories and completed segment annotations. Each task
bank is a `.pt` file containing a `data` list. A trajectory contains `demo_id`,
`Xro`, `Uro`, `r3m_agent`, and `r3m_wrist`, with channels before time. The annotation
root contains `manifest.json` with a `tasks` instruction map, plus
`episodes/<task>/<demo>/annotations.json`. Each annotation contains `steps` and
contiguous segments with `start_step`, `end_step_exclusive`, `primitive`, and
`caption` fields.

```bash
python -m bpc_gnn.prepare \
  --banks /data/r3m/banks \
  --annotations /data/annotations \
  --output /data/annotation_embeddings.pt
```

Preparation uses ten-step histories and ten-step future actions, a per-task
90/5/5 demonstration split, and a pinned MiniLM model revision. Histories combine
actions at `[t-10,t)` with states at `[t-9,t+1)`. Future actions occupy `[t,t+10)`.
Normalization is fitted on training demonstrations by BPC.

The existing AWS input locations are recorded in `configs/aws_assets.json`.
`configs/reference_data.json` records their experiment identities and reward scales.
The existing campaign's annotation cache can be used directly. Reuse that cache
for exact text-vector identity. Raw-image feature extraction and VLM annotation
generation are documented in the parent repository's
[annotation guide](https://github.com/madang6/bpc-multi/blob/4852cbb9bdd19f409c56de0f3b948ef6d01aad59/docs/demonstration_annotation.md).

## Train

Training additionally requires a fitted pooled Fisher artifact associated with
the same bank and annotation identity. BPC validates that identity and computes
the Fisher-key cache when the cache path is new. `configs/train.json` contains
the 2,048-update, eight-task recipe. Its two reward scales were calibrated on
training-only Fisher queries from the original bank. Recalibrate those scales
when changing the dataset or Fisher model.

```bash
python -m bpc_gnn.train \
  --banks /data/r3m/banks \
  --annotations /data/annotation_embeddings.pt \
  --fisher /data/fisher.pt \
  --fisher-cache /data/fisher_keys.pt \
  --output outputs/seed20261105 \
  --device cuda:0
```

Use `--seed` to change the training seed while preserving evaluation seeds.
Use `--steps` to override the update budget. A JSON config can be passed with
`--config`. Model dimensions and balanced batch sizes are validated against the
loaded data. `configs/bpc.json` retains the physical-bank and continuation settings
used by the parent study.

`metrics.jsonl` contains each update and deterministic/sampled validation curves,
including training probes and per-task results. `checkpoint.pt` saves weights,
optimizer state and training random-number states. `identity.json` records data,
Fisher, configuration, source and runtime identities. `status.json` reports the
latest saved update. Add `--resume` with the same output path to continue.
A `STOP` file in the run directory requests a checkpoint at the next update
boundary. Remove it before resuming. A file lock prevents concurrent writers.

## Evaluate

```bash
python -m bpc_gnn.evaluate \
  --banks /data/r3m/banks \
  --annotations /data/annotation_embeddings.pt \
  --fisher /data/fisher.pt \
  --fisher-cache /data/fisher_keys.pt \
  --checkpoint outputs/seed20261105/checkpoint.pt \
  --output outputs/seed20261105/validation.json \
  --device cuda:0
```

Evaluation reports future-action MSE, physical-history reconstruction MSE,
support size and coefficient amplification, including sampled retrieval and
per-task means. `--queries-per-task` changes the cohort size and `--split test`
selects the held-out test demonstrations. Evaluation uses its own random-number
generator. The entry point accepts checkpoints written by this repository.
Model state dictionaries also match the parent's `bank_summary` architecture,
which enables direct parity tests with its saved checkpoints.

This extraction uses the original summary-node GNN. Candidate-to-candidate
refinement variants and the multi-GPU campaign scheduler live in the parent
repository. The evaluation here measures offline prediction and reconstruction.
Closed-loop task success remains a separate experiment.

## Check correctness

```bash
python -m unittest discover -s tests -v
wc -l bpc_gnn/*.py
```

The tests compare model scores, gradients, sampling and prediction metrics with
the pinned implementation. They check annotation boundaries, demonstration
exclusion, gradient replay, and equivalence of resumed and uninterrupted
training. Test fixtures and reference implementations come from the pinned BPC
checkout.

The extraction also passed a CPU comparison on all 1,608,027 real bank columns
using a trained summary-node checkpoint. Candidate embeddings matched exactly,
as did full-bank scores, adaptive supports and sampled neighborhoods for sixteen
queries spanning all eight tasks and both training and validation splits.
The receipt is in `validation/real-bank-parity.json`. This validates the extraction
on those inputs. The ongoing parent campaign measures model quality separately.
