This repository implements the experiment from the proposal and implementation plan:

- **Model assumptions**: Qwen2.5-14B-Instruct, bf16, all 48 layers
- **Prompt set**: 500 truthful/untruthful prompt pairs
- **Temperatures**: selectable via CLI; default sweep is `0.0`, `0.3`, and `0.7`
- **Formulation**: token-level Hankel EDMD using prompt-tail + generated-token activations
- **Sequence construction**:
  - Last `5` prompt tokens from the prefill pass
  - First `32` generated tokens
  - Hankel window `W = 5`
- **Layer selection**: final hidden layer, middle hidden layer, or token-wise concatenation of all hidden layers
- **Observable bases**:
  - Coordinate basis (`D = 5120 × 5 = 25,600`)
  - PCA basis (`m = 256`, fit on a random sample of up to `10,000` training activations)
  - Fourier basis (retain the first `32` low-frequency coefficients by default)
- **Operator fitting**:
  - 80/20 prompt-level train/test split
  - Ridge grid `λ ∈ {10, 100, 1000, 5000, 10000}`

## Files

| File | Description |
|---|---|
| `extract_data.py` | Runs the model on prompts, extracts prompt-prefill and generated-token activations/logits/outputs, writes `.npz` files into temperature-specific `data_<temp>/` directories. |
| `inspect_outputs.py` | Reads all `.npz` files in a data directory and prints a diagnostic report (summary, per-file table, pair comparison, flagged issues). |
| `run_experiment.py` | Runs the token-level EDMD experiment on extracted data, comparing coordinate, PCA, and Fourier observables across selectable temperatures. |
| `analyze_multistep_dynamics.py` | Replays saved Koopman operators on held-out trajectories to measure multi-step rollout error for the selected output folders. |
| `eigenmode_analysis.py` | Compares dominant truthful vs untruthful Koopman modes and projects final-layer coordinate modes back into vocabulary space. |
| `set_affinity_gpu_polaris.sh` | MPI wrapper that pins each rank to a unique GPU on Polaris (ALCF). |
| `experiment/` | Package containing config, EDMD, basis maps, probes, data loading, and prompt expansion. |
| `data_0/`, `data_0.3/`, `data_0.7/` | Temperature-specific output directories for extracted `.npz` trajectory files (created by `extract_data.py`). |

## Data contract

Put extracted trajectory files in temperature-specific directories such as `data_0/`, `data_0.3/`, and `data_0.7/` (parallel to `experiment/`) as `.npz` files.  
Each file should contain:

- `hk` (required): shape `[num_blocks, L, T0, d]`
- `label` (required): `{1/0}` (truthful/untruthful)
- `prompt_id` (required): unique prompt identifier
- `temperature` (required): float
- `hl` (optional): shape `[L, T, d]` (if omitted, it is derived from `hk`)
- `logits` (optional): shape `[num_blocks, T0, topk]` (required for top-k logits basis)
- `generated_token_ids` (optional): shape `[num_blocks, T0]` — generated token ids
- `generated_text` (optional): decoded model output text
- `generated_text_raw` (optional): raw decoded output before stripping any reasoning tags
- `response_text` (optional): decoded output with any extracted reasoning span removed
- `input_token_ids` (optional): shape `[prompt_len]` — full prompt token ids used for prefill
- `input_tok_len_raw` (optional): int — true prompt token count before generation
- `category` (optional): prompt category string
- `topic` (optional): prompt topic string
- `prompt_kind` (optional): `"truthful"` or `"untruthful"`
- `prompt_hl` (optional): shape `[L, prompt_len, d]` — prompt-side hidden states from the prefill pass
- `prompt_logits` (optional): shape `[prompt_len, topk]` — top-k logits at each prompt token position during prefill
- `reasoning_trace_found` (optional): boolean indicating whether a tagged reasoning span was detected
- `reasoning_text` (optional): extracted reasoning content, if present
- `reasoning_token_ids` (optional): token ids belonging to the extracted reasoning span
- `reasoning_token_mask` (optional): shape `[num_blocks * T0]` — boolean mask over generated tokens for the reasoning span
- `kv_cache` (optional): shape `[num_layers, 2, n_kv_heads, total_seq_len, head_dim]` — final KV-cache state (only saved with `--save-kv-cache`)

## Extract Data

`extract_data.py` reads `experiment/data_prompts.json`, runs the model, and writes `.npz` files into temperature-specific directories such as `data_0/`, `data_0.3/`, and `data_0.7/`.

### Polaris (ALCF) — interactive job

```bash
qsub -I -l select=1 -l filesystems=home:eagle -l walltime=1:00:00 -q debug -A <YourProject>
module load cray-python
cd /eagle/GeoSolv/qinan/LSD/lsd_v2

# Single GPU
mpiexec -n 1 --ppn 1 ./set_affinity_gpu_polaris.sh python3 extract_data.py

# All 4 GPUs (work is automatically partitioned across ranks)
mpiexec -n 4 --ppn 4 --depth=8 --cpu-bind depth ./set_affinity_gpu_polaris.sh python3 extract_data.py
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--prompts-file` | `experiment/data_prompts.json` | Input prompt JSON file. |
| `--output-dir` | `data` | Output directory prefix. Files are written to sibling directories named `data_<temp>`. |
| `--model-name` | `Qwen/Qwen2.5-14B-Instruct` | HuggingFace model name or local path. |
| `--t0` | `20` | Generation block size. Total new tokens = `num_blocks * t0`. |
| `--num-blocks` | `4` | Generated blocks per sample. |
| `--max-pairs` | `None` | Cap on prompt pairs from JSON (useful for test runs). |
| `--logit-topk` | `256` | Store top-k logit values per step. |
| `--deterministic-temp` | `0.0` | Temperature for deterministic generation. |
| `--mid-temp` | `0.3` | Intermediate stochastic temperature. |
| `--stochastic-temp` | `0.7` | Higher stochastic temperature. |
| `--temperatures` | `0.0 0.3 0.7` | Explicit list of temperatures to extract. |
| `--device` | `auto` | `auto`, `cuda`, or `cpu`. |
| `--dtype` | `auto` | `auto`, `bf16`, or `fp32`. |
| `--include-truthful` | off | Only include truthful prompts. |
| `--include-untruthful` | off | Only include untruthful prompts. |
| `--save-kv-cache` | off | Save final KV-cache state (~18-36 MB extra per file). |

### Test run (50 pairs, 4 GPUs)

```bash
mpiexec -n 4 --ppn 4 --depth=8 --cpu-bind depth ./set_affinity_gpu_polaris.sh python3 extract_data.py --max-pairs 50
```

### Test run with KV cache

```bash
mpiexec -n 1 --ppn 1 ./set_affinity_gpu_polaris.sh python3 extract_data.py --max-pairs 5 --save-kv-cache
```

### One temperature only

```bash
mpiexec -n 1 --ppn 1 ./set_affinity_gpu_polaris.sh python3 extract_data.py --temperatures 0
```

### Notes

- `topic + prompt` text is tokenized once for the full prefill pass; `input_token_ids` and `prompt_hl` therefore cover the entire prompt sequence, not a truncated `T0`-token window.
- Generated-token activations are still stored in fixed-size blocks with `T0` tokens per block.
- By default, extraction runs at `temperature in {0.0, 0.3, 0.7}` and writes each temperature into its own `data_<temp>/` directory.
- Each distinct prompt is extracted once per requested temperature, so every `data_<temp>/` directory contains at most one `.npz` per prompt.
- If Qwen emits tagged reasoning spans such as `<think>...</think>`, the extractor stores the reasoning text and token alignment alongside the normal decoded response.
- `logits` are stored as top-k values (`k=256` by default) to reduce disk usage.
- Model weights are loaded directly onto the GPU via `torch.device()` context manager (no CPU staging).
- Work is automatically partitioned across MPI ranks using `PMI_RANK`/`PMI_SIZE`.
- The HF model is resolved to a local cache path (`HF_HUB_OFFLINE=1`) so compute nodes do not require internet access.

## Inspect Outputs

`inspect_outputs.py` reads all `.npz` files in a selected data directory and prints a diagnostic report.

```bash
python3 inspect_outputs.py
python3 inspect_outputs.py --data-dir data_0 --max-files 10 --max-text 120
```

Report sections:

1. **Summary** — file count, truthful/untruthful split, average input/output token lengths, padding/truncation counts, shape consistency, KV-cache storage.
2. **Per-file table** — one row per `.npz`: prompt ID, kind, topic, input token count (raw and padded), output token count, text preview, issues.
3. **Pair comparison** — side-by-side truthful vs untruthful generated text with token counts for the first 10 pairs.
4. **Flagged files** — files with `EMPTY_TEXT`, `VERY_SHORT`, `REPETITION(xN)`, `ALL_SAME_TOKEN`, `PADDED`, or `TRUNCATED` issues.

## Run Experiment

```bash
pip install -r requirements.txt
python run_experiment.py --data-root data_0 --output-dir outputs/token_level
```

`run_experiment.py` assumes `--data-root` points to the deterministic directory (for example `data_0`) and infers sibling directories for the other requested temperatures (for example `data_0.3` and `data_0.7`).
For each requested temperature, results are written to a temperature-specific output directory of the form `token_level_<temp>_<max_pairs>`, for example `outputs/token_level_0_10` or `outputs/token_level_0.7_all`.

### Experimental procedure

For each selected temperature:

1. Load the extracted trajectories from `data_<temp>/`.
2. For each prompt, build a token-level sequence consisting of:
   - the last `5` prompt tokens from `prompt_hl`
   - the first `32` generated token activations from `hk`
3. Select either the final layer, the middle layer, or concatenate all layers for each token.
4. Transform each token activation using one of the supported observables:
   - coordinate basis
   - PCA basis
   - Fourier basis
5. Build Hankel states with window size `W = 5`.
6. Split prompts into 80% train and 20% test.
7. Fit a ridge-regularized Koopman operator and choose the best `λ` by held-out one-step prediction MSE, or optionally hold `λ` fixed.
8. Save eigenvalue plots, operator artifacts, held-out MSE summaries, and truth-vs-lie spectral overlays.

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--data-root` | `data_0` | Deterministic temperature directory. Other requested temperatures are inferred as sibling directories. |
| `--output-dir` | `outputs/token_level` | Base output directory. Each selected temperature writes to a sibling directory named `<base>_<temp>_<layer>_<alpha_mode>_<max_pairs>`. |
| `--layer` | `final` | Hidden representation to analyze: `final`, `middle`, or `all` (concatenate all layers per token). |
| `--temperatures` | `0.0 0.3 0.7` | Temperatures to evaluate. |
| `--bases` | `coordinate pca fourier` | Observable bases to evaluate. |
| `--max-pairs` | `None` | Optional cap on prompt pairs. |
| `--fixed-alpha` | `None` | Optional fixed ridge penalty. When set, the experiment skips the search grid and fits only this `alpha`. |
| `--prompt-tail` | `5` | Number of prompt tokens kept before generation. |
| `--generation-steps` | `32` | Number of generated token activations used per prompt. |
| `--hankel-window` | `5` | Hankel window size. |
| `--pca-dim` | `256` | PCA latent dimension before Hankel concatenation. |
| `--pca-sample-size` | `10000` | Maximum number of training activations sampled to fit PCA. |
| `--fourier-low-freq-count` | `32` | Number of low-frequency Fourier coefficients retained per token activation. |
| `--log-file` | `None` | Optional path to mirror logs to a file. |
| `--log-level` | `INFO` | Logging verbosity. |

### Example runs

```bash
# Final layer, all default temperatures and bases
python run_experiment.py --data-root data_0 --output-dir outputs/token_level

# Middle layer, coordinate basis only, deterministic temperature only
python run_experiment.py --data-root data_0 --layer middle --temperatures 0.0 --bases coordinate

# Final layer, PCA / Fourier only, deterministic + high-temperature comparison
python run_experiment.py --data-root data_0 --temperatures 0.0 0.7 --bases pca fourier

# Fourier basis with a smaller retained spectrum
python run_experiment.py --data-root data_0 --bases fourier --fourier-low-freq-count 16

# Deterministic run on 10 prompt pairs -> outputs/token_level_0_final_grid_10
python run_experiment.py --data-root data_0_simple --temperatures 0.0 --bases pca coordinate fourier --max-pairs 10

# Force a single ridge penalty instead of searching the grid -> outputs/token_level_0_final_alpha1000_all
python run_experiment.py --data-root data_0_simple --temperatures 0.0 --bases coordinate --fixed-alpha 1000

# Concatenate all layers for each token before building Hankel states -> outputs/token_level_0_all_grid_10
python run_experiment.py --data-root data_0_simple --layer all --temperatures 0.0 --bases pca fourier --max-pairs 10
```

### Outputs

For each selected temperature, the experiment writes:

- `experiment_results.json` with per-temperature / per-label / per-basis metrics
- `heldout_mse_summary.png` comparing held-out 1-step prediction error
- `comparisons/overlay_tau_<temp>_<basis>.png` truth-vs-lie eigenvalue overlays
- `artifacts/...` with saved operator matrices and per-combo eigenvalue plots

Outputs are written under the temperature-specific directory derived from `--output-dir`.

## Analyze Multi-Step Dynamics

`analyze_multistep_dynamics.py` scans saved `run_experiment.py` outputs, rebuilds the held-out trajectories from the extracted activation data, and reports rollout error over multiple horizons.

By default it is restricted to:

- temperatures `0.0`, `0.3`, `0.7`
- non-fixed penalty runs (`alpha_mode = grid_search`)
- pair caps `100`, `150`, and `200`
- `final` and `middle` layers only (`layer=all` is skipped)
- output trees other than `outputs/tier1` and `outputs/last_token_pca`

Example:

```bash
python3.11 analyze_multistep_dynamics.py --outputs-root outputs --data-root data_0_simple
```

Useful flags:

| Flag | Default | Description |
|---|---|---|
| `--outputs-root` | `outputs` | Root containing `run_experiment.py` output directories. |
| `--data-root` | `data_0_simple` | Deterministic activation directory; sibling temperature directories are inferred from it. |
| `--temperatures` | `0.0 0.3 0.7` | Temperatures to analyze. |
| `--pair-caps` | `100 150 200` | Only analyze output folders whose parsed prompt-pair cap is in this list. |
| `--alpha-modes` | `grid_search` | Keep only selected penalty modes. |
| `--max-horizon` | `8` | Maximum rollout horizon scored on the held-out set. |
| `--coordinate-eval-dims` | `1024` | For coordinate basis, evaluate rollout MSE on a reproducible subset of Hankel coordinates. |
| `--log-file` | `None` | Optional path to mirror logs to a file. |
| `--log-level` | `INFO` | Logging verbosity. |

Outputs per matching folder:

- `multistep_rollout_results.json`
- `multistep_rollout_summary.png`
- `multistep_rollout_results_coordinate.json`
- `multistep_rollout_results_pca.json`
- `multistep_rollout_results_fourier.json`
- `multistep_rollout_summary_<basis>.png`

Global index:

- `outputs/multistep_rollout_index.json`

## Eigenmode Analysis

`eigenmode_analysis.py` compares truthful and untruthful dominant Koopman modes for each selected `(temperature, layer, basis, pair-cap)` run.

It reports:

- cosine similarity between the dominant `\lambda \approx 1` truthful and untruthful eigenvectors
- the dominant eigenvalue for each side
- for `basis=coordinate` and `layer=final`, a vocabulary projection of the dominant mode using the Qwen unembedding matrix

The vocabulary projection uses the selected token slot inside the Hankel window (default: the most recent token, `--projection-token-index -1`) and reports the top decoded tokens for the truthful and untruthful dominant modes.

Example:

```bash
python3.11 eigenmode_analysis.py --outputs-root outputs
```

Useful flags:

| Flag | Default | Description |
|---|---|---|
| `--outputs-root` | `outputs` | Root containing `run_experiment.py` output directories. |
| `--results-glob` | `**/experiment_results.json` | Glob used to discover result files under `--outputs-root`. |
| `--temperatures` | `0.0 0.3 0.7` | Temperatures to analyze. |
| `--pair-caps` | `100 150 200` | Only analyze output folders whose parsed prompt-pair cap is in this list. |
| `--alpha-modes` | `grid_search` | Keep only selected penalty modes. |
| `--model-name` | `Qwen/Qwen2.5-14B-Instruct` | Model snapshot used to load the tokenizer and unembedding matrix. |
| `--projection-top-k` | `5` | Number of top decoded tokens reported for final-layer coordinate dominant modes. |
| `--projection-token-index` | `-1` | Hankel token slot projected back into vocabulary space. |
| `--log-file` | `None` | Optional path to mirror logs to a file. |
| `--log-level` | `INFO` | Logging verbosity. |

Outputs:

- per matching folder: `eigenmode_analysis.json`
- global index: `outputs/eigenmode_analysis_index.json`
