# Reproduce the paper

All three tools run on Apple Silicon with `mlx-lm`. Point them at Hugging Face repo ids or local paths.

## The blindspot (canary on base vs instruct models)

```bash
# instruction-tuned models enumerate the SOP faithfully and invent nothing
python canary.py --model mlx-community/Qwen3.5-9B-4bit
python canary.py --model mlx-community/Mistral-7B-Instruct-v0.3-4bit

# a raw base model reproduces steps but fails the conditional branch
python canary.py --model mistralai/Mistral-7B-v0.3
```

Read `recall` first. A reliable agent has `recall >= 0.8` and `invented_x` near 0. A model with low recall did not do the task, so its `invented_x` is not meaningful. Reasoning models need `--think` and a larger `--max-tokens`.

## The mechanism (coherence probe)

```bash
python coherence_probe.py --model meta-llama/Llama-3.1-8B
python coherence_probe.py --model mistralai/Mistral-7B-v0.3
```

Full run computes one SVD spectrum per linear tensor, which takes a few minutes on a 7–8B model. Add `--max-tensors 60` for a fast sample. You are looking for the coherence gap: low-rank SVD above `coherent_fraction = 0.007` at every dose, pruning below it at every density. The `0.007` and `0.01` thresholds are calibrated on 7–8B dense models; recalibrate per family before treating the gate as a hard block.

## The screen (agent-safety gate)

```bash
# build a quantized model, then screen it against its source
python -m mlx_lm convert --hf-path meta-llama/Llama-3.1-8B -q --q-bits 4 --mlx-path llama-4bit
python agent_safety_gate.py --bf16 meta-llama/Llama-3.1-8B --quant llama-4bit

# aggressive low-bit trips the gate
python -m mlx_lm convert --hf-path meta-llama/Llama-3.1-8B -q --q-bits 2 --mlx-path llama-2bit
python agent_safety_gate.py --bf16 meta-llama/Llama-3.1-8B --quant llama-2bit --strict
```

Standard 4-bit quantization passes (its error is low-coherence). Aggressive 2-bit and any low-rank-factorized build get flagged. Wire the `--strict` form into a quantization pipeline right after the build step to fail agent-unsafe builds automatically.

## Wiring the gate into a pipeline

```python
from agent_safety_gate import compute_gate
g = compute_gate("path/to/bf16-source", "path/to/mlx-quant")
if g["verdict"] == "FLAG":
    raise SystemExit("build is agent-unsafe on the coherence axis; run canary.py to confirm")
```
