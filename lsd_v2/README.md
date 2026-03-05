# STAT 31310 Koopman Experiment Implementation

This repository implements the experiment from the proposal and implementation plan:

- **Model assumptions**: Llama 3.1 8B, bf16, all 32 layers
- **Prompt set**: 500 truthful/untruthful prompt pairs
- **Regimes**:
  - Deterministic (`temperature=0.0`)
  - Stochastic (`temperature=0.7`, 32 rollouts per prompt)
- **Formulations**:
  - `H_k` block evolution
  - `H_l` layer evolution (starting at layer 10)
- **EDMD variants**:
  - Hankel delays for `H_k`: `0,2,4`
  - Hankel delays for `H_l`: `0,3,8`
  - Basis maps:
    - `H_k`: top-k logits (`k=256`), Fourier (`10` and `32` low-frequency coefficients), PCA (`m=512`), autoencoder (`m=512`)
    - `H_l`: Fourier (`10` and `32`), PCA (`m=512`), autoencoder (`m=512`)
- **Linear probe**:
  - 80/20 split
  - Grid `C ∈ [1e-3, 1e-1, 1, 10, 1e3]`

## Data contract

Put extracted trajectory files in `data/` (parallel to `experiment/`) as `.npz` files.  
Each file should contain:

- `hk` (required): shape `[num_blocks, L, T0, d]`
- `label` (required): `{truthful/untruthful}` or `{1/0}`
- `prompt_id` (required): unique prompt identifier
- `temperature` (required): float
- `hl` (optional): shape `[L, T, d]` (if omitted, it is derived from `hk`)
- `logits` (optional): shape `[num_blocks, T0, vocab]` (required for top-k logits basis)
- `rollout_id` (optional): integer rollout id (used for stochastic grouping)

## Extract Data

`extract_data.py` reads `experiment/data_prompts.json`, runs the model, and writes `.npz` files into `data/`.

```bash
pip install -r requirements.txt
python extract_data.py --prompts-file experiment/data_prompts.json --output-dir data
```

Useful test run:

```bash
python extract_data.py --max-pairs 2 --stochastic-rollouts 2 --num-blocks 6
```

Notes:
- `topic + prompt` text is ingested, tokenized, then forced to exactly `T0` tokens (default `16`) via truncate/pad.
- The extractor writes both deterministic (`temperature=0.0`) and stochastic (`temperature=0.7`) trajectories by default.
- `logits` are stored as top-k values (`k=256` by default) to reduce disk usage.

## Run Experiment

```bash
pip install -r requirements.txt
python run_experiment.py --data-root data --output-dir outputs
```

Outputs are written to `outputs/experiment_results.json`.
