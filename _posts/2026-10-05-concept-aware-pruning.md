---
layout: post
title: "Concept-Aware Pruning: Keeping the Neurons That Matter"
date: 2026-10-05T08:00:00+00:00
author: Ahmed Samir Oraby
categories:
  - llm
tags: [LLM, Pruning, Model Compression, Interpretability, PyTorch]
excerpt: "Protect the neurons that carry specific capabilities, and prune the rest. A report on how concept-aware pruning works, how to apply it across architectures, and experimental results across 3 model families."
---

Protect the neurons that carry specific capabilities, and prune the rest. This report covers how the method works, how to apply it to other pruning tasks, and what the experiments show across three model families.

![Lambada perplexity after pruning 40% of MLP width on Llama-3.2-1B: magnitude 88.5, NB02 hybrid 47.5, concept-aware 24.0, dense 5.62]({{ '/assets/images/concept-pruning/overview.png' | prepend: site.baseurl }})

---

## 1. The question

Standard LLM pruning scores each neuron by weight magnitude or by how strongly it activates on some calibration text. Neither score knows what a neuron does. This project asks whether we get a better pruned model at the same size if we first find the neurons that matter for specific capabilities ("concepts") and protect them. It also asks whether that advantage survives held-out evaluation, a different architecture, and recovery fine-tuning.

The idea comes from arXiv 2601.03570 (Yao et al., *"How Do Large Language Models Learn Concepts During Continual Pre-Training?"*), which studies concept circuits during training. That paper never discusses pruning. Using concept circuits to guide pruning is this project's own extension. Baselines and evaluation protocols are adapted from Chapter 5 of Pere Martra's *Rearchitecting LLMs*.

---

## 2. How the architecture works

The method changes only one thing in a standard width-pruning pipeline: the score used to decide which neurons to keep. The removal mechanics, the number of neurons removed per layer, and therefore the final parameter count are identical to the baseline. Any difference in quality comes from *which* neurons are kept.

### Pipeline at a glance

1. **Define concepts**: clean / corrupted prompt pairs
2. **Attribute**: Integrated Gradients (IG) on `down_proj` input
3. **Aggregate**: mean over pairs, max over concepts
4. **Blend**: `minmax(base) + β · minmax(cov)`
5. **Prune**: keep top-k per layer
6. **Evaluate**: held-out probes + `lm_eval`

### 2.1 Where it hooks in: the GLU MLP

Llama, Qwen and Gemma MLP blocks are gated linear units. `gate_proj` and `up_proj` expand the hidden state (2048 wide in Llama-3.2-1B) to the intermediate width (8192). The two are combined elementwise, and `down_proj` projects back to 2048. Width pruning removes intermediate neurons. Removing neuron *i* deletes row *i* of `gate_proj`, row *i* of `up_proj` and column *i* of `down_proj`. 

The input to `down_proj` (called X<sub>d</sub>) has exactly one value per intermediate neuron. That makes it the natural place to measure importance, and it is where our attribution reads.

![Diagram of a GLU MLP block with a hook on the input to down_proj]({{ '/assets/images/concept-pruning/mlp_hook.svg' | prepend: site.baseurl }})

### 2.2 Concepts as contrastive pairs

A concept is a set of **clean / corrupted prompt pairs** plus a target token. The clean prompt makes the target correct, and the corrupted prompt changes only the detail the concept depends on. Both prompts must tokenize to the same length, because the attribution step swaps activations between them position by position.

| Concept | Clean prompt → target | Corrupted prompt |
|---|---|---|
| **IOI** | John and Mary went to the park. John gave a book to → ` Mary` | … Mary gave a book to |
| **SVA** | The keys in the cabinet → ` are` | The key in the cabinet |
| **Factual** | The capital of France is → ` Paris` | The capital of Japan is |

*Training and held-out pairs use disjoint pools of names, places, nouns and countries, so held-out accuracy is never measured on anything the attribution step saw.*

**More examples by use case:**

| Use case | Capability protected | Clean prompt → target | Corrupted prompt |
|---|---|---|---|
| Translation | Choosing target language | Translate to French: Thank you. → ` Merci` | Translate to Spanish: Thank you. |
| Translation | Word sense from context | He sat by the river bank. French: la → ` rive` | He went to the city bank. French: la |
| De-identification | Entity type (person vs place) | Sarah lives in Cairo. Cairo is a → ` city` | Sarah lives in Cairo. Sarah is a |
| Classification | Sentiment label | Review: The food was wonderful. Sentiment: → ` positive` | Review: The food was terrible. Sentiment: |
| RAG / document QA | Answering from context, not memory | Context: The meeting is on Tuesday. Question: When is the meeting? Answer: → ` Tuesday` | Context: The meeting is on Friday. Question: When is the meeting? Answer: |
| Code | Variable binding | `a, b = 3, 7; print(a) # prints` → ` 3` | `a, b = 3, 7; print(b) # prints` |
| Structured output | Closing the right bracket | `values = [1, 2, 3` → `]` | `values = (1, 2, 3` |
| Arabic | Factual recall in another language | عاصمة مصر هي → `القاهرة` | عاصمة فرنسا هي |

*These rows are illustrative. Only IOI, SVA, factual, greater-than and arithmetic were tested in §4.*

### 2.3 Attribution: integrated gradients on X<sub>d</sub>

For each pair, the model runs once on the corrupted prompt and once on the clean prompt, and the hook captures X<sub>d</sub> at every layer. It then runs *m* = 4 more forward and backward passes on the corrupted input. In each pass, every layer's X<sub>d</sub> is replaced by a point on the straight line from corrupted to clean, and the loss is the cross-entropy of the clean target token. The neuron's score is the average gradient times the activation difference, which estimates how much moving that neuron from "corrupted" to "clean" contributes to predicting the right answer:

```
x_α         = x_corr + α · (x_clean − x_corr),    α ∈ {¼, ½, ¾, 1}
grad_ℓ,t,j  = mean over α of ∂ CE(logits_last, target) / ∂ x_α[ℓ, t, j]
A_ℓ,j       = Σ_t | grad_ℓ,t,j · (x_clean − x_corr)[ℓ, t, j] |
cov_ℓ,j     = max over concepts ( mean over pairs A_ℓ,j )
```

Where ℓ is the layer, *t* the token position and *j* the intermediate neuron. Three design choices matter:

- **Contrast, not raw activation:** A neuron that fires the same on clean and corrupted inputs gets zero score, however large its activation. Only neurons that carry the difference the concept depends on are protected.
- **Max over concepts:** A neuron is protected if it matters to *any* chosen concept. Summing instead would favour neurons that are moderately useful to everything.
- **Neuron-level, not circuit-level:** The paper's own tool, EAP-IG, treats each MLP as one node over the 2048-wide residual stream and cannot score the 8192 intermediate neurons width pruning removes. This node-level attribution is the right granularity for this job.

Cost: 120 pairs (IOI and SVA, 60 each) took **47 seconds** on one L40S for Llama-3.2-1B.

### 2.4 Blending and pruning

Concept coverage alone is sparse. Most neurons get near-zero scores, and a neuron unused by the probes can still matter for general fluency. So coverage is blended with a base importance score, per layer, after scaling both to [0, 1]:

```
base_ℓ   = p2p(gate_proj rows) + p2p(up_proj rows)        p2p(w) = max(w) + |min(w)|
score_ℓ  = minmax(base_ℓ) + β · minmax(cov_ℓ)               β = 2 (default)
keep_ℓ   = top-k(score_ℓ),   k = width_ℓ − ⌊pp · width_ℓ⌋   (same k as the baseline)
```

**How the score is calculated, step by step:**

1. **Base score from the weights:** For neuron *j*, take row *j* of `gate_proj`, which holds the 2048 weights feeding that neuron, and add its largest weight to the absolute value of its smallest. Do the same for row *j* of `up_proj` and add the two. This is the magnitude baseline (uses no data).
2. **Coverage from the concepts:** `cov_j` is the neuron's attribution averaged over the pairs of one concept, then maximized across concepts.
3. **Put both on the same scale:** Min-max scale both metrics within the layer: `(v_j − min v) / (max v − min v)`.
4. **Add them with weight β:** `score_j = scaled base + β · scaled coverage`. With β = 2 the score runs from 0 to 3. A neuron whose scaled coverage is above 0.5 scores more than 1, so it outranks zero-coverage neurons regardless of magnitude.
5. **Keep the top k:** Sort the layer's neurons by score and keep `k = N − ⌊pp · N⌋`. For Llama-3.2-1B at 40%, that keeps 4,916 neurons per layer out of 8,192.

A toy layer with 5 neurons, `pp = 40%` and `β = 2`:

| Neuron | base | cov | scaled base | scaled cov | score | Magnitude only | Concept-aware |
|---|---|---|---|---|---|---|---|
| A | 1.20 | 0.000 | 1.00 | 0.00 | 1.00 | kept | kept |
| B | 0.90 | 0.002 | 0.70 | 0.05 | 0.80 | kept | kept |
| C | 0.70 | 0.000 | 0.50 | 0.00 | 0.50 | kept | pruned |
| D | 0.40 | 0.040 | 0.20 | 1.00 | **2.20** | pruned | **kept** |
| E | 0.20 | 0.012 | 0.00 | 0.30 | 0.60 | pruned | pruned |

D has small weights but carries the concept, so it outranks C and survives.

### 2.5 Package layout

- `model_utils.py`: Loads any Hugging Face causal LM, with a fallback for multimodal wrappers. `find_decoder_layers()` finds transformer blocks even when nested, and `get_layer_mlp_widths()` handles non-uniform widths.
- `concepts.py`: `ConceptPair` data structures and loaders from templates, in-memory lists, or JSONL files. Filters out length-mismatched prompts.
- `attribution.py`: `_AttributionHooks` (context manager that patches X<sub>d</sub> with forward-pre-hooks), `compute_concept_importance()`, and aggregation routines.
- `scoring.py`: Magnitude importance and `blend_scores()`.
- `pruning.py`: `prune_model(model, pp, coverage, beta)` removes gate/up/down neuron triples by score and updates module dimensions.
- `evaluation.py`: Held-out concept accuracy, `lm_eval` wrapper, and custom metric evaluators.
- `cli.py`: `concept-prune run` CLI for end-to-end pruning and benchmark comparison.

---

## 3. Using it for any pruning task

Every structured pruning method has the same three parts: a **unit** to remove (neuron, head, block, expert, weight), a **score** per unit, and a **removal step** that drops the lowest-scoring units under a budget. Concept-aware pruning only replaces the score. It plugs into any method where you can find an activation that belongs to exactly one unit, hook it, and patch it.

### 3.1 The general recipe

1. **Pick the unit and its hook point:** The hooked tensor must have one slice per prunable unit.
2. **Write concept pairs:** For capabilities you must keep (same token length, one target token, disjoint vocabulary held-out set).
3. **Run attribution at that hook point:** Reduce per-position scores to one number per unit.
4. **Aggregate:** Mean over pairs, then max over concepts.
5. **Blend with the baseline score:** After scaling both to [0, 1]. Verify `β = 0` reproduces the baseline exactly.
6. **Evaluate on three axes:** Held-out concept probes, general benchmarks at limit ≥ 200, and perplexity on uncalibrated text.

### 3.2 Mapping to other pruning types

| Pruning type | Unit and hook point | Status |
|---|---|---|
| **MLP width (GLU)** | Intermediate neuron. Pre-hook on `down_proj` input. | **Validated** |
| **Depth (whole blocks)** | Decoder block. Block delta (output − input), summed over hidden dim. | Tested, no gain |
| **Combined with data-driven score** | Same neuron. Blend coverage into activation-based score. | Additive blend fails |
| **Attention heads** | Head. Pre-hook on `o_proj` input, reshaped to `[heads, head_dim]`. | Untested |
| **MoE experts** | Expert. Hook each expert's output before router weighting. | Untested |
| **Unstructured / Wanda-style** | Weight. Use coverage of input channel as multiplier on \|W\|·‖X‖. | Untested |
| **Per-layer sparsity budget** | Layer. Use layer total coverage to give concept-heavy layers lower ratios. | Untested |

### 3.3 Where it fits and where it doesn't

The attribution asks: *how much does swapping this unit's activation from corrupted to clean change the answer?* That matches width pruning because removing one neuron out of 8,192 is a small, local perturbation. It does not match depth pruning, which deletes several whole blocks at once, requiring the residual stream to absorb macro changes.

The rule of thumb: **the method helps most when the removed units are many and small, and the budget is aggressive (40% or more).**

### 3.4 Adapter sketch for a new unit type

```python
# What you write for a new unit: hook point and per-unit reduction
def hook_module(layer):            # e.g. heads: layer.self_attn.o_proj
    return layer.self_attn.o_proj

def reduce_to_units(attr):         # attr: [seq, hidden] -> [n_units]
    return attr.abs().sum(0).view(n_heads, head_dim).sum(-1)

# Everything else is reused unchanged:
cov   = combine_concepts({c: attribute(model, pairs[c], hook_module, reduce_to_units) for c in pairs})
score = minmax(base_score) + beta * minmax(cov)
keep  = topk(score, k_same_as_baseline)
```

---

## 4. Results

Unless noted: `unsloth/Llama-3.2-1B` (fp16, β = 2), concept accuracy on held-out pairs (n = 60), `lm_eval` 0-shot at limit = 200. Baseline is magnitude pruning from Chapter 5 of *Rearchitecting LLMs*.

### 4.1 The core effect at 40% width pruning

| Metric | Dense | Magnitude | Concept-aware |
|---|---|---|---|
| **Lambada PPL** | 5.62 | 88.52 | **28.39** |
| **Lambada acc** | 0.635 | 0.300 | **0.445** |
| **IOI** | 0.750 | 0.200 | **0.983** |
| **SVA** | 0.600 | 0.450 | **0.533** |
| **Factual** | 0.817 | 0.000 | **0.567** |

Both pruned models have 913.8M of 1,235.8M parameters. All metrics are on held-out data unvisited during attribution.

### 4.2 Pruning ratio sweep

![Perplexity at every pruning ratio from 20% to 60%]({{ '/assets/images/concept-pruning/sweep.png' | prepend: site.baseurl }})

| Pruned | Magnitude PPL | β = 2 PPL | β = 4 PPL |
|---|---|---|---|
| **20%** | 30.78 | **9.90** | 10.10 |
| **30%** | 58.74 | **17.40** | 17.73 |
| **40%** | 88.52 | **28.09** | 34.19 |
| **50%** | 388.84 | **91.70** | 95.78 |
| **60%** | 2634.41 | 627.68 | **579.31** |

β = 2 beats magnitude on every metric at every ratio from 20% to 60%.

### 4.3 Second architecture: Qwen2.5-1.5B at 40%

| Metric | Dense | Magnitude | Concept-aware |
|---|---|---|---|
| **Lambada PPL** | 6.13 | 937.46 | **242.6** |
| **IOI** | 0.417 | 0.250 | **0.433** |
| **Greater-than** | 1.000 | 0.000 | **1.000** |
| **Factual** | 0.917 | 0.000 | **0.350** |
| **SVA** | 0.833 | 0.783 | 0.750 |

The retention advantage is even larger on Qwen2.5-1.5B than on Llama.

### 4.4 How many concepts to protect

| Protected | Lambada PPL | ARC-Easy | IOI | SVA | Factual |
|---|---|---|---|---|---|
| none (magnitude) | 88.52 | 0.400 | 0.133 | 0.500 | 0.000 |
| IOI | 24.19 | 0.440 | 1.000 | 0.700 | 0.233 |
| **IOI + SVA** | **24.03** | **0.440** | 1.000 | 0.733 | **0.733** |
| + factual | 29.65 | 0.415 | 1.000 | 0.833 | 0.717 |
| + arithmetic | 29.55 | 0.420 | 1.000 | 0.833 | 0.717 |

Protection spreads to concepts not explicitly probed: factual recall jumped when SVA was added. IOI + SVA is the recommended default.

### 4.5 After recovery fine-tuning (Qwen3-0.6B, 10% pruning)

| Metric | Magnitude pre → post | Concept-aware pre → post |
|---|---|---|
| **ARC-Easy (n ≈ 2376)** | 0.486 → 0.535 | **0.563 → 0.593** |
| **HellaSwag** | 0.420 → 0.430 | **0.450 → 0.480** |
| **Lambada acc** | 0.140 → 0.300 | **0.230 → 0.360** |
| **SVA** | 0.333 → 0.367 | **0.883 → 0.717** |
| **Factual** | 0.000 → 0.167 | **0.567 → 0.633** |

ARC-Easy 95% confidence intervals do not overlap before or after recovery. Generic recovery fine-tuning does not restore capabilities that pruning erased completely.

### 4.6 Against a stronger baseline: CH05 NB02

The second notebook of Chapter 5 in *Rearchitecting LLMs* introduces a stronger data-driven method (NB02) that multiplies a weight score by the L2 norm of activations over calibration texts. Concept-aware pruning roughly halves Lambada perplexity compared to NB02 while keeping far higher accuracy on the protected capabilities.

![NB02 against concept-aware on WikiText and Lambada perplexity]({{ '/assets/images/concept-pruning/nb02.png' | prepend: site.baseurl }})

| 40% pruning | WikiText ppl | SMS ppl | Lambada ppl | Lambada acc | PIQA | IOI | SVA | Factual |
|---|---|---|---|---|---|---|---|---|
| **Magnitude (NB01)** | 141.7 | 773.8 | 88.5 | 0.300 | 0.625 | 0.25 | 0.38 | 0.00 |
| **NB02, WikiText calib** | **96.2** | 547.3 | 47.5 | 0.315 | 0.605 | 0.65 | 0.75 | 0.42 |
| **NB02, SMS calib** | 124.1 | **493.0** | 52.4 | 0.325 | 0.600 | 0.20 | 0.58 | 0.40 |
| **Concept-aware** | 116.4 | 621.0 | **24.0** | **0.460** | **0.670** | **1.00** | 0.77 | **0.80** |
| **NB02 (wiki) + concept** | 117.6 | 609.7 | 45.1 | 0.355 | 0.580 | 0.95 | **0.82** | 0.35 |

WikiText and SMS perplexities are on held-out text that no method calibrated on. Concept-aware achieves half the Lambada perplexity of NB02 (24.0 vs 47.5).

---

## 5. The package

`concept_prune` packages the width-pruning method for any Hugging Face causal LM with GLU MLPs. The main repository is [github.com/oraby8/concept-aware-pruning](https://github.com/oraby8/concept-aware-pruning).

### CLI Usage

```bash
concept-prune run \
  --model unsloth/Llama-3.2-1B \
  --concepts examples/concepts_ioi_sva.yaml \
  --holdout-concepts examples/concepts_ioi_sva_holdout.yaml \
  --compression 0.4 --beta 2.0 \
  --benchmarks lambada_openai,arc_easy,piqa --limit 200 \
  --output results.json
```

### Python API

```python
from concept_prune import load_model, find_decoder_layers
from concept_prune.concepts import concepts_from_templates
from concept_prune.examples_templates import generate_ioi, generate_sva
from concept_prune.attribution import compute_concept_importance, combine_concepts
from concept_prune.pruning import prune_model

model, tok = load_model("unsloth/Llama-3.2-1B")
layers     = find_decoder_layers(model)
concepts   = concepts_from_templates(tok, {"ioi": generate_ioi, "sva": generate_sva}, n_per_concept=60)
coverage   = combine_concepts({n: compute_concept_importance(model, p, layers=layers) for n, p in concepts.items()})
pruned     = prune_model(model, 0.4, coverage_by_layer=coverage, beta=2.0)
```
