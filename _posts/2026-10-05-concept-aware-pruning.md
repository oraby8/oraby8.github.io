---
layout: post
title: "Concept-Aware Pruning: Keeping the Neurons That Matter"
date: 2026-10-05T08:00:00+00:00
author: Ahmed Samir Oraby
categories:
  - llm
tags: [LLM, Pruning, Model Compression, Interpretability, PyTorch]
excerpt: "Standard LLM pruning drops neurons by weight magnitude or activation size without knowing what a neuron actually does. I tested scoring neurons by whether they carry specific capabilities instead — protecting concept circuits before pruning."
---

![Lambada perplexity after pruning 40% of MLP width on Llama-3.2-1B: magnitude 88.5, NB02 hybrid 47.5, concept-aware 24.0, dense 5.62]({{ '/assets/images/concept-pruning/overview.png' | prepend: site.baseurl }})

## The question

Width pruning removes neurons from a model's MLP blocks to make it smaller and faster. The usual question is *which* neurons to remove, and the usual answers don't know anything about what a neuron does. Magnitude pruning keeps the neurons with the largest weights. Data-driven pruning keeps the ones that fire hardest on some calibration text.

I wanted to know whether we get a better pruned model at the same size if we first find the neurons that matter for specific capabilities ("concepts") and protect them. I wanted to check whether that advantage holds on data the scoring never saw, on a different architecture, after recovery fine-tuning, and against stronger data-driven baselines.

The idea comes from [How Do Large Language Models Learn Concepts During Continual Pre-Training?](https://arxiv.org/abs/2601.03570) (Yao et al.), which studies "concept circuits" inside LLMs during training. That paper doesn't discuss pruning — using concept circuits to guide structured pruning is my own extension. The baselines and evaluation protocol come from Chapter 5 of Pere Martra's *Rearchitecting LLMs*.

## How it works

The method changes only one thing in a standard width-pruning pipeline: the score used to decide which neurons to keep. The removal mechanics, the number of neurons removed per layer, and therefore the final parameter count are identical to the baseline. Any difference in quality comes from *which* neurons survive.

### Where it hooks in: the GLU MLP

Llama, Qwen, and Gemma use gated (GLU) MLP blocks. `gate_proj` and `up_proj` expand the hidden state (2048 wide in Llama-3.2-1B) to the intermediate width (8192). The two are combined elementwise, and `down_proj` projects back to 2048. Removing intermediate neuron *i* means deleting row *i* of `gate_proj`, row *i* of `up_proj`, and column *i* of `down_proj`.

![Diagram of a GLU MLP block with a hook on the input to down_proj, which has one value per intermediate neuron]({{ '/assets/images/concept-pruning/mlp_hook.svg' | prepend: site.baseurl }})

The input to `down_proj` (call it X<sub>d</sub>) has exactly one value per intermediate neuron, which makes it the natural place to measure importance. A forward-pre-hook there can either record the activation or replace it with a different tensor.

### Concepts as contrastive pairs

A concept is a set of clean and corrupted prompt pairs plus a target token. The clean prompt makes the target correct. The corrupted prompt changes only the detail the concept depends on. Both must tokenize to the same length, because the attribution step swaps activations between them position by position.

| Concept | Clean prompt → target | Corrupted prompt |
|---|---|---|
| **Indirect object (IOI)** | John and Mary went to the park. John gave a book to → ` Mary` | … Mary gave a book to |
| **Subject-verb agreement (SVA)** | The keys in the cabinet → ` are` | The key in the cabinet |
| **Factual recall** | The capital of France is → ` Paris` | The capital of Japan is |

Training pairs and held-out pairs come from disjoint pools of names, places, nouns, and countries, so held-out accuracy never reuses anything the scoring step saw.

### Attribution: Integrated Gradients on X<sub>d</sub>

For each pair, the model runs once on the corrupted prompt and once on the clean one, and the hook records X<sub>d</sub> at every layer. Then it runs four more forward and backward passes on the corrupted prompt. In each pass, every layer's X<sub>d</sub> is replaced by a point on the straight line from the corrupted activation to the clean one. The loss is the cross-entropy of the clean target token.

```
x_α         = x_corr + α · (x_clean − x_corr),    α ∈ {¼, ½, ¾, 1}
grad_ℓ,t,j  = mean over α of ∂loss / ∂x_α[ℓ, t, j]
score_ℓ,j   = Σ_t | grad_ℓ,t,j · (x_clean − x_corr)[ℓ, t, j] |
coverage    = max over concepts (mean over pairs of score)
```

Here ℓ is the layer, *t* the token position, and *j* the neuron. This is integrated gradients applied at the neuron level:

- **It measures contrast, not activation size.** A neuron that fires the same way on clean and corrupted prompts scores zero, however strongly it fires. Only neurons that carry the difference the concept depends on get protected.
- **Max over concepts.** A neuron is protected if it matters to *any* of the chosen concepts. Summing instead would favor neurons that are moderately useful for everything.
- **Neuron level, not circuit level.** I first tried the paper's own tool, EAP-IG. Its reference implementation treats each MLP as a single node over the 2048-wide residual stream, so it can't score the 8192 intermediate neurons that width pruning removes. The node-level version above is the right granularity for this job.

It's cheap: 120 pairs (60 each for IOI and SVA) took **47 seconds** on one L40S for Llama-3.2-1B.

### Blending and pruning

Concept coverage alone is sparse. Most neurons score near zero, and a neuron the probes never used can still matter for general fluency. So I blend it with the usual magnitude score, per layer, after scaling both to [0, 1]:

```
base_ℓ   = p2p(gate_proj rows) + p2p(up_proj rows)        p2p(w) = max(w) + |min(w)|
score_ℓ  = minmax(base_ℓ) + β · minmax(cov_ℓ),              β = 2 (default)
keep_ℓ   = top-k(score_ℓ),   k = width_ℓ − ⌊pp · width_ℓ⌋   (same k as the baseline)
```

With β = 0 this reproduces magnitude pruning exactly, which I use as a sanity check in every run.

## Results

Unless noted otherwise: Llama-3.2-1B, fp16, β = 2, concept accuracy on 60 held-out pairs, and `lm_eval` 0-shot at 200 examples per task.

### The core effect at 40% pruning

| Metric | Dense | Magnitude | Concept-aware |
|---|---|---|---|
| **Lambada perplexity** | 5.62 | 88.52 | **28.39** |
| **Lambada accuracy** | 0.635 | 0.300 | **0.445** |
| **IOI (held-out)** | 0.750 | 0.200 | **0.983** |
| **SVA (held-out)** | 0.600 | 0.450 | **0.533** |
| **Factual (held-out)** | 0.817 | 0.000 | **0.567** |

Both pruned models have 913.8M of the original 1,235.8M parameters. All metrics are on held-out data unvisited during attribution.

### Across pruning ratios

![Lambada perplexity by share of MLP neurons removed from 20% to 60%]({{ '/assets/images/concept-pruning/sweep.png' | prepend: site.baseurl }})

| Pruned | Magnitude PPL | β = 2 PPL | β = 4 PPL |
|---|---|---|---|
| **20%** | 30.78 | **9.90** | 10.10 |
| **30%** | 58.74 | **17.40** | 17.73 |
| **40%** | 88.52 | **28.09** | 34.19 |
| **50%** | 388.84 | **91.70** | 95.78 |
| **60%** | 2634.41 | 627.68 | **579.31** |

β = 2 beats magnitude pruning on every metric at every ratio from 20% to 60%.

### A second architecture: Qwen2.5-1.5B

On Qwen2.5-1.5B at 40%, magnitude pruning is far more destructive than on Llama (Lambada perplexity 937 vs dense 6.1). Concept-aware cuts that to 243 and keeps IOI, greater-than, and factual recall well above the baseline. The effect is even larger here than on Llama.

| Metric | Dense | Magnitude | Concept-aware |
|---|---|---|---|
| **Lambada PPL** | 6.13 | 937.46 | **242.6** |
| **IOI** | 0.417 | 0.250 | **0.433** |
| **Greater-than** | 1.000 | 0.000 | **1.000** |
| **Factual** | 0.917 | 0.000 | **0.350** |
| **SVA** | 0.833 | 0.783 | 0.750 |

### How many concepts to protect

| Protected | Lambada PPL | ARC-Easy | IOI | SVA | Factual |
|---|---|---|---|---|---|
| **none (magnitude)** | 88.52 | 0.400 | 0.133 | 0.500 | 0.000 |
| **IOI** | 24.19 | 0.440 | 1.000 | 0.700 | 0.233 |
| **IOI + SVA** | **24.03** | **0.440** | 1.000 | 0.733 | **0.733** |
| **+ factual** | 29.65 | 0.415 | 1.000 | 0.833 | 0.717 |
| **+ arithmetic** | 29.55 | 0.420 | 1.000 | 0.833 | 0.717 |

Protection spreads to concepts you didn't probe: factual recall jumped when I added SVA, not when I added factual itself. Adding a third concept cost about 23% in perplexity. Two concepts (IOI + SVA) was the best trade-off here.

### After recovery fine-tuning

Pruned models are usually fine-tuned afterwards to recover. I used the distillation recipe from *Rearchitecting LLMs* on Qwen3-0.6B at 10% pruning:

| Metric | Magnitude pre → post | Concept-aware pre → post |
|---|---|---|
| **ARC-Easy (n ≈ 2376)** | 0.486 → 0.535 | **0.563 → 0.593** |
| **HellaSwag** | 0.420 → 0.430 | **0.450 → 0.480** |
| **Lambada acc** | 0.140 → 0.300 | **0.230 → 0.360** |
| **SVA** | 0.333 → 0.367 | **0.883 → 0.717** |
| **Factual** | 0.000 → 0.167 | **0.567 → 0.633** |

On ARC-Easy at full size (about 2,400 questions), concept-aware won before and after recovery with non-overlapping 95% confidence intervals. Recovery narrows the general-benchmark gap, but the concept gap survives: generic recovery data doesn't rebuild a capability that pruning removed entirely.

### Against a stronger baseline: CH05 NB02

The second notebook of Chapter 5 in *Rearchitecting LLMs* introduces a stronger data-driven method (NB02) that multiplies a weight score by the L2 norm of activations over calibration texts (WikiText-2 or SMS Spam). It reads the same hook point as my attribution, making it the natural competitor:

![At 40% pruning, NB02 calibrated on WikiText has the lowest held-out WikiText perplexity, while concept-aware has the lowest Lambada perplexity and highest accuracy]({{ '/assets/images/concept-pruning/nb02.png' | prepend: site.baseurl }})

| 40% pruning | WikiText ppl | SMS ppl | Lambada ppl | Lambada acc | PIQA | IOI | SVA | Factual |
|---|---|---|---|---|---|---|---|---|
| **Magnitude (NB01)** | 141.7 | 773.8 | 88.5 | 0.300 | 0.625 | 0.25 | 0.38 | 0.00 |
| **NB02, WikiText calib** | **96.2** | 547.3 | 47.5 | 0.315 | 0.605 | 0.65 | 0.75 | 0.42 |
| **NB02, SMS calib** | 124.1 | **493.0** | 52.4 | 0.325 | 0.600 | 0.20 | 0.58 | 0.40 |
| **Concept-aware** | 116.4 | 621.0 | **24.0** | **0.460** | **0.670** | **1.00** | 0.77 | **0.80** |
| **NB02 (wiki) + concept** | 117.6 | 609.7 | 45.1 | 0.355 | 0.580 | 0.95 | **0.82** | 0.35 |

What this shows:
- **NB02 is a much stronger baseline than magnitude pruning.** It roughly halves Lambada perplexity at both 20% and 40%.
- **NB02's domain specialization replicates.** Each NB02 model has the lowest perplexity on its own calibration domain on held-out text. Concept-aware doesn't beat it there.
- **Concept-aware still wins overall.** Lambada perplexity is half of NB02's (24.0 vs 47.5), Lambada accuracy is 14 points higher, and it keeps far more of the protected capabilities.

## Where it doesn't help: depth pruning

Chapter 4 of the book removes whole decoder blocks, ranked by Block Importance (how little a block changes its input). I tried concept-aware block selection on Qwen3-0.6B, removing 3 of 28 blocks.

Scoring what each block adds (output minus input) produced a genuinely different signal, but it only tied Block Importance: slightly better on WikiText and SMS perplexity, worse on Lambada, and worse on its own concept probes.

The reason is what the attribution measures: it asks how much swapping one activation from corrupted to clean changes the answer. That matches removing one neuron out of 8,192. It doesn't match deleting three whole blocks and leaving the residual stream to absorb the change.

## What this doesn't prove

I want to be upfront about the limits here, because it's easy for clean numbers to overstate certainty:

- **Small models only.** Everything tested here is 0.6B to 1.5B parameters.
- **Single run, no significance testing.** The only formal confidence intervals are from the ARC-Easy full-set check.
- **Templated probes.** The concepts are short next-token tasks. Two others I tried (greater-than and simple arithmetic) were too easy on Llama-3.2-1B to tell any method apart.
- **One recovery recipe.** It transfers poorly to Llama-3.2-1B, so the recovery conclusions are strongest on Qwen3-0.6B.

What I think this *does* show: scoring neurons by capability rather than raw magnitude or general activation prevents catastrophic capability collapse under aggressive pruning budgets.

## What's next

- A better way to combine NB02 and concept scores, such as blending ranks or multiplying NB02's score by `(1 + β · coverage)`, to get NB02's domain perplexity and concept-aware's capability retention in one model.
- Evaluating on the full benchmark suite from Chapter 5, including IFEval to test whether width pruning can improve instruction following.
- Speed and energy throughput measurements for the pruned models on edge hardware.

---

*Code and reproduction scripts are available in the repository at [github.com/oraby8/concept-aware-pruning](https://github.com/oraby8/concept-aware-pruning).*
