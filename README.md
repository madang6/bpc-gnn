# BPC-GNN

Task-conditioned, future-informed retrieval for Behavior Predictive Control (BPC).

## Problem formulation

Consider task-conditioned MDPs $`\mathcal{M}_c=(\mathcal{X},\mathcal{U},P_c,r_c,\gamma)`$ with observations $`y^{(t)}=f_y(x^{(t)})`$ and expert demonstrations $`\mathcal{D}=\bigcup_c\mathcal{D}_c`$. Following [Adang et al., *Training-free Behavior Cloning*](https://trainingfreebehaviorcloning.github.io/), §III, a query and its trajectory-template library are

```math
z^{(t)}=\mathrm{col}\!\left(u^{(t-T_{\mathrm{ini}}:t-1)},y^{(t-T_{\mathrm{ini}}+1:t)}\right),
\qquad H_{\mathcal{D}}=\begin{bmatrix}H\\U\end{bmatrix}.
```

Each column pairs a demonstrated history $`h_i`$ with its future-action continuation $`u_i^+`$. The objective is to construct a task-conditioned policy $`\hat{\pi}_{\mathcal{D}}(z,c)`$ to estimate the expert's next action $`u^{(t),\star}`$.

## Retrieval and action continuation

A GNN passes messages between candidate histories and 32 query-conditioned summary nodes. Task instructions and recorded candidate-continuation captions supply semantic features. For eligible columns $`i\in\mathcal{I}`$, it adds learned corrections to pooled sparsemax-LDA scores, $`s_i=s_i^{\mathrm{Fisher}}+\Delta s_{\theta,i}`$. The sparsemax threshold $`\tau`$ determines the evaluation neighborhood

```math
\sum_{i\in\mathcal{I}}\max(s_i-\tau,0)=1,
\qquad \mathcal{N}_{\mathrm{det}}=\{i\in\mathcal{I}:s_i>\tau\}.
```

For the retrieved banks $`H_{\mathcal{N}},U_{\mathcal{N}}`$, the affine continuation prior is

```math
g^{\star}\in\arg\min_{\mathbf{1}^{\top} g=1}\frac{1}{2}\|H_{\mathcal{N}}g-z\|_2^2,
\qquad \hat{u}_f=U_{\mathcal{N}}g^{\star},
\qquad G(z,c)=A_{\mathcal{N}}g^{\star},
```

where $`A_{\mathcal{N}}=U_{\mathcal{N}}^{(1)}`$ contains the first future action. This configuration uses the paper's Eq. (6) with $`\lambda_g=0`$.

For exploration temperature $`T>0`$, training assigns independent Bernoulli inclusion probabilities $`q_i`$ to columns, then conditions on a nonempty neighborhood $`\varnothing\ne\mathcal{N}\subseteq\mathcal{I}`$,

```math
q_i=\frac{1}{1+\exp(-(s_i-\tau)/T)},
\qquad p_\theta(\mathcal{N}\mid z,c,\mathcal{D})=
\frac{\prod_{i\in\mathcal{N}}q_i\prod_{j\in\mathcal{I}\setminus\mathcal{N}}(1-q_j)}
{1-\prod_{j\in\mathcal{I}}(1-q_j)}.
```

Demonstrated continuations $`u_f^{\star}`$ define the reward and offline training objective

```math
\begin{aligned}
R(\mathcal{N})&=-\frac{\mathrm{MSE}(U_{\mathcal{N}}g^{\star},u_f^{\star})}{C_u}
                -\frac{\mathrm{MSE}(H_{\mathcal{N}}g^{\star},z)}{C_z},\\
\max_\theta J(\theta),\qquad J(\theta)&=
\mathbb{E}_{(z,u_f^{\star},c)\sim\mathcal{D},\;\mathcal{N}\sim p_\theta}[R(\mathcal{N})].
\end{aligned}
```

Here $`C_u,C_z`$ are the mean action and history-reconstruction errors of Fisher retrieval on training queries. For each query, draw $`M\ge2`$ independent neighborhoods $`\mathcal{N}_m\sim p_\theta`$ and write $`R_m=R(\mathcal{N}_m)`$. The policy-gradient estimate subtracts the mean reward of the other samples

```math
b_m=\frac{1}{M-1}\sum_{\ell\ne m}R_\ell,
\qquad \widehat{\nabla_\theta J}=\frac{1}{M}\sum_{m=1}^{M}
(R_m-b_m)\nabla_\theta\log p_\theta(\mathcal{N}_m\mid z,c,\mathcal{D}).
```

Batches contain equal numbers of queries from each task, and each training query's entire demonstration is excluded from $`\mathcal{I}`$. Evaluation reports deterministic and sampled action/reconstruction errors by task.

## Run

Inputs are cached R3M demonstrations, segment annotations, and a fitted pooled Fisher model. BPC is a pinned submodule. Existing AWS asset paths and the training recipe are provided in the configs. [Assets](configs/aws_assets.json) · [Training configuration](configs/train.json)

```bash
git clone --recurse-submodules https://github.com/madang6/bpc-gnn.git
cd bpc-gnn
python -m pip install -r requirements.txt

python -m bpc_gnn.prepare --banks /data/r3m/banks \
  --annotations /data/annotations --output /data/annotation_embeddings.pt

inputs=(--banks /data/r3m/banks --annotations /data/annotation_embeddings.pt
        --fisher /data/fisher.pt --fisher-cache /data/fisher_keys.pt)
python -m bpc_gnn.train "${inputs[@]}" --output outputs/run
python -m bpc_gnn.evaluate "${inputs[@]}" \
  --checkpoint outputs/run/checkpoint.pt --output outputs/run/validation.json

python -m unittest discover -s tests -v
```

Use Bash and a CUDA-enabled PyTorch installation. Training accepts `--config`, `--seed`, `--steps`, and `--resume`. The defaults use $`T_{\mathrm{ini}}=T_f=10`$. Recalibrate $`C_u,C_z`$ when changing the bank or Fisher model. Full-bank extraction parity is recorded in the validation receipt. [Parity check](validation/real-bank-parity.json)
