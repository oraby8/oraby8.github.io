---
layout: post
title: "Concept-Aware Pruning: Keeping the Neurons That Matter"
date: 2026-10-05T08:00:00+00:00
author: Ahmed Samir Oraby
categories:
  - llm
tags: [LLM, Pruning, Model Compression, Interpretability, PyTorch]
excerpt: "Width pruning usually decides which MLP neurons to drop by weight magnitude or activation size. I tried scoring neurons by whether they carry specific capabilities instead, and compared it with both pruning methods from Chapter 5 of Rearchitecting LLMs at identical parameter counts."
---

![Lambada perplexity after pruning 40% of MLP width on Llama-3.2-1B: magnitude 88.5, NB02 hybrid 47.5, concept-aware 24.0, dense 5.62]({{ '/assets/images/concept-pruning/overview.png' | prepend: site.baseurl }})

## The question

Width pruning removes neurons from a model's MLP blocks to make it smaller and faster. The usual question is *which* neurons to remove, and the usual answers don't know anything about what a neuron does. Magnitude pruning keeps the neurons with the largest weights. Data-driven pruning keeps the ones that fire hardest on some calibration text.

I wanted to know whether we get a better pruned model at the same size if we first find the neurons that carry specific capabilities and protect them. I wanted to check that the advantage holds on data the scoring never saw, on a different architecture, after recovery fine-tuning, and against a stronger data-driven baseline.

The idea comes from [How Do Large Language Models Learn Concepts During Continual Pre-Training?](https://arxiv.org/abs/2601.03570) (Yao et al.), which studies "concept circuits" inside LLMs during training. That paper doesn't discuss pruning. Using concept circuits to guide pruning is my own extension. The baselines and evaluation protocol come from Chapter 5 of Pere Martra's [Rearchitecting LLMs](https://github.com/peremartra/Rearchitecting-LLMs).

## How it works

The method changes one thing in a standard width-pruning pipeline: the score used to decide which neurons to keep. The removal step, the number of neurons removed per layer, and therefore the final parameter count are identical to the baseline. Any difference in quality comes from *which* neurons survive.

### Where it hooks in

Llama, Qwen and Gemma use gated (GLU) MLP blocks. `gate_proj` and `up_proj` expand the hidden state from 2048 to 8192 dimensions in Llama-3.2-1B, the two are multiplied elementwise, and `down_proj` projects back to 2048. Removing intermediate neuron *i* means deleting row *i* of `gate_proj`, row *i* of `up_proj`, and column *i* of `down_proj`.

![Diagram of a GLU MLP block with a hook on the input to down_proj, which has one value per intermediate neuron]({{ '/assets/images/concept-pruning/mlp_hook.svg' | prepend: site.baseurl }})

The input to `down_proj` (call it X<sub>d</sub>) has exactly one value per intermediate neuron, which makes it the right place to measure importance. A forward-pre-hook there can either record the activation or replace it with a different tensor. The book's data-driven method (CH05 NB02) reads its activation statistics from the same point.

### Concepts as contrastive pairs

A concept is a set of clean and corrupted prompt pairs plus a target token. The clean prompt makes the target correct. The corrupted prompt changes only the detail the concept depends on. Both must tokenize to the same length, because the attribution step swaps activations between them position by position.

| Concept | Clean prompt → target | Corrupted prompt |
|---|---|---|
| Indirect object (IOI) | John and Mary went to the park. John gave a book to → ` Mary` | … Mary gave a book to |
| Subject-verb agreement (SVA) | The keys in the cabinet → ` are` | The key in the cabinet |
| Factual recall | The capital of France is → ` Paris` | The capital of Japan is |

Training pairs and held-out pairs come from disjoint pools of names, places, nouns and countries, so held-out accuracy never reuses anything the scoring step saw.

### Attribution

For each pair, the model runs once on the corrupted prompt and once on the clean one, and the hook records X<sub>d</sub> at every layer. Then it runs four more forward and backward passes on the corrupted prompt. In each pass, every layer's X<sub>d</sub> is replaced by a point on the straight line from the corrupted activation to the clean one. The loss is the cross-entropy of the clean target token.

```
x_α          = x_corr + α · (x_clean − x_corr)          α ∈ {¼, ½, ¾, 1}
grad[l,t,j]  = mean over α of ∂loss / ∂x_α[l,t,j]
score[l,j]   = Σ_t | grad[l,t,j] · (x_clean − x_corr)[l,t,j] |
coverage     = max over concepts of (mean over pairs of score)
```

Here *l* is the layer, *t* the token position, and *j* the neuron. This is integrated gradients, applied at the neuron level. A few choices matter:

- **It measures contrast, not activation size.** A neuron that fires the same way on clean and corrupted prompts scores zero, however strongly it fires. Only neurons that carry the difference the concept depends on get protected.
- **Max over concepts.** A neuron is protected if it matters to *any* of the chosen concepts. Summing instead would favour neurons that are moderately useful for everything.
- **Neuron level, not circuit level.** I first tried the paper's own tool, EAP-IG. Its reference implementation treats each MLP as a single node over the 2048-wide residual stream, so it can't score the 8192 intermediate neurons that width pruning removes. The node-level version above is the right granularity for this job.

It's cheap. 120 pairs (60 each for IOI and SVA) took 47 seconds on one L40S for Llama-3.2-1B. The book's NB02 activation calibration took 13 seconds.

### Blending and pruning

Concept coverage alone is sparse. Most neurons score near zero, and a neuron the probes never used can still matter for general fluency. So I blend it with the usual magnitude score, per layer, after scaling both to [0, 1]:

```
magnitude[l] = peak-to-peak(gate_proj rows) + peak-to-peak(up_proj rows)
score[l]     = minmax(magnitude[l]) + β · minmax(coverage[l])        β = 2
keep[l]      = top-k(score[l]),   k = same as the baseline
```

With β = 0 this reproduces magnitude pruning exactly, which I use as a sanity check in every run.

## Results

Unless I say otherwise: Llama-3.2-1B, fp16, β = 2, concept accuracy on 60 held-out pairs, and `lm_eval` 0-shot at 200 examples per task.

### The core effect at 40% pruning

| Metric | Dense | Magnitude | Concept-aware |
|---|---|---|---|
| Lambada perplexity | 5.62 | 88.52 | **28.39** |
| Lambada accuracy | 0.635 | 0.300 | **0.445** |
| IOI (held-out) | 0.750 | 0.200 | **0.983** |
| SVA (held-out) | 0.600 | 0.450 | **0.533** |
| Factual (held-out) | 0.817 | 0.000 | **0.567** |

Both pruned models have 913.8M of the original 1,235.8M parameters.

### Across pruning ratios

![Lambada perplexity by share of MLP neurons removed: concept-aware stays roughly 3 to 4 times lower than magnitude pruning from 20% to 60%]({{ '/assets/images/concept-pruning/sweep.png' | prepend: site.baseurl }})

β = 2 beats magnitude pruning on every metric at every ratio from 20% to 60%. β = 4 keeps slightly more concept accuracy at 50% and above, at some cost in perplexity, so it's a trade-off rather than an upgrade.

### A second architecture

On Qwen2.5-1.5B at 40%, magnitude pruning is far more destructive than on Llama (Lambada perplexity 937 vs dense 6.1). Concept-aware cuts that to 243 and keeps IOI, greater-than and factual recall well above the baseline. The effect is larger here than on Llama.

### How many concepts to protect

| Protected | Lambada perplexity | IOI | SVA | Factual |
|---|---|---|---|---|
| none (magnitude) | 88.52 | 0.133 | 0.500 | 0.000 |
| IOI | 24.19 | 1.000 | 0.700 | 0.233 |
| IOI + SVA | **24.03** | 1.000 | 0.733 | 0.733 |
| + factual | 29.65 | 1.000 | 0.833 | 0.717 |

Protection spreads to concepts you didn't probe. Factual recall jumped when I added SVA, not when I added factual itself. Adding a third concept cost about 23% in perplexity. Two concepts was the best trade-off here.

### After recovery fine-tuning

Pruned models are usually fine-tuned afterwards to recover. I used the book's own distillation recipe on Qwen3-0.6B at 10% pruning, the setup it was built for. Recovery narrows the general-benchmark gap, but the concept gap survives: SVA 0.72 vs 0.37 and factual recall 0.63 vs 0.17 after recovery. On ARC-Easy at full size (about 2,400 questions), concept-aware won before and after recovery with non-overlapping 95% confidence intervals. Generic recovery data doesn't rebuild a capability that pruning removed entirely.

## Against a stronger baseline: CH05 NB02

Everything above compares against magnitude pruning, the method from the book's first Chapter 5 notebook. The second notebook (NB02) introduces a stronger method that uses data. It multiplies a weight score from all three projections by the L2 norm of X<sub>d</sub> over 100 calibration texts, from either WikiText-2 or SMS Spam. It reads the same hook point as my attribution, so it's the natural competitor.

![At 40% pruning, NB02 calibrated on WikiText has the lowest held-out WikiText perplexity (96.2 vs 116.4 for concept-aware), while concept-aware has the lowest Lambada perplexity (24.0 vs 47.5)]({{ '/assets/images/concept-pruning/nb02.png' | prepend: site.baseurl }})

| 40% pruning | WikiText ppl | SMS ppl | Lambada ppl | Lambada acc | PIQA | IOI | SVA | Factual |
|---|---|---|---|---|---|---|---|---|
| Magnitude (NB01) | 141.7 | 773.8 | 88.5 | 0.300 | 0.625 | 0.25 | 0.38 | 0.00 |
| NB02, WikiText calibration | **96.2** | 547.3 | 47.5 | 0.315 | 0.605 | 0.65 | 0.75 | 0.42 |
| NB02, SMS calibration | 124.1 | **493.0** | 52.4 | 0.325 | 0.600 | 0.20 | 0.58 | 0.40 |
| Concept-aware | 116.4 | 621.0 | **24.0** | **0.460** | **0.670** | **1.00** | 0.77 | **0.80** |
| NB02 (wiki) + concept | 117.6 | 609.7 | 45.1 | 0.355 | 0.580 | 0.95 | **0.82** | 0.35 |

WikiText and SMS perplexities are on held-out text that no method calibrated on. HellaSwag, WinoGrande and ARC-Easy were within ±3.5 points of each other, which is the noise level at 200 examples.

What this shows:

- **NB02 is a much stronger baseline than magnitude pruning.** It roughly halves Lambada perplexity at both 20% and 40%.
- **NB02's domain specialization replicates.** Each NB02 model has the lowest perplexity on its own calibration domain, on held-out text too. Concept-aware doesn't beat it there.
- **Concept-aware still wins overall.** Lambada perplexity is half of NB02's (24.0 vs 47.5), Lambada accuracy is 14 points higher, and it keeps far more of the protected capabilities.
- **Adding concept coverage to NB02's score doesn't work.** It brings IOI and SVA back but loses most of the Lambada gain. NB02's score is a product of weight and activation terms, and plain addition after min-max scaling mixes two very differently shaped distributions. I've only tried that one blend.

At 20% the pattern is the same with smaller gaps: Lambada perplexity 10.2 for concept-aware vs 12.0 for NB02 and 30.8 for magnitude.

## Where it doesn't help: depth pruning

Chapter 4 of the book removes whole decoder blocks, ranked by Block Importance (how little a block changes its input). I tried concept-aware block selection on Qwen3-0.6B, removing 3 of 28 blocks.

My first version scored each block's output, which carries the whole residual stream from earlier blocks, so every block looked about equally important. Scoring only what each block *adds* (output minus input) fixed that and produced a genuinely different signal. But it only tied Block Importance: slightly better on WikiText and SMS perplexity, worse on Lambada, and worse on its own concept probes.

I think the reason is what the attribution measures. It asks how much swapping one activation from corrupted to clean changes the answer. That matches removing one neuron out of 8,192. It doesn't match deleting three whole blocks and leaving the residual stream to absorb the change.

## Using it for other pruning tasks

Every structured pruning method has three parts: a unit to remove, a score per unit, and a step that removes the lowest-scoring units under a budget. Concept-aware pruning only replaces the score. It works wherever you can hook an activation with one slice per unit:

1. Pick the unit and a hook point with one slice per unit.
2. Write clean/corrupted pairs for the capabilities you need to keep, plus a held-out set from different vocabulary.
3. Run the attribution at that hook and reduce it to one number per unit.
4. Average over pairs, take the max over concepts.
5. Blend with the method's existing score after scaling both to [0, 1]. Keep the budget and removal step unchanged, and check that β = 0 reproduces the baseline.
6. Evaluate on held-out concept probes, general benchmarks at 200+ examples, and perplexity on text no step calibrated on.

| Pruning type | Unit and hook point | Status |
|---|---|---|
| MLP width | Intermediate neuron, input to `down_proj` | Works |
| Depth | Whole block, block output minus input | Ties Block Importance |
| Combined with NB02 | Same neuron, added to NB02's score | Additive blend fails |
| Attention heads | Head, input to `o_proj` split per head | Untested |
| MoE experts | Expert, each expert's output | Untested |
| Unstructured (Wanda-style) | Weight, coverage of its input channel | Untested |

The method helps most when the removed units are many and small, and the pruning budget is aggressive.

## What this doesn't prove

- **Small models only.** Everything here is 0.6B to 1.5B parameters.
- **One seed.** The only formal confidence intervals are from the ARC-Easy check.
- **Templated probes.** The concepts are short next-token tasks. Two others I tried (greater-than and simple arithmetic) were too easy on Llama-3.2-1B to tell any method apart.
- **One recovery recipe.** It transfers poorly to Llama-3.2-1B, so the recovery conclusions are strongest on Qwen3-0.6B.
- **Small evaluation samples mislead.** At 100 examples per task, close results flipped in both directions when I reran them at full size. I now treat any close call below a few hundred examples as provisional.

## What's next

- A better way to combine NB02 and concept scores, such as blending ranks or multiplying NB02's score by (1 + β · coverage), to get NB02's domain perplexity and concept-aware's retention in one model.
- The full benchmark table from the book's first Chapter 5 notebook, including IFEval, MMLU, ARC-Challenge, BoolQ and GSM8K. IFEval tests the chapter's claim that width pruning can improve instruction following.
- Speed and energy measurements for the pruned models, matching NB02's.
