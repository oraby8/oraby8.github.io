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

## The question

Width pruning removes neurons from a model's MLP blocks to make it smaller and faster. The usual question is *which* neurons to remove, and the usual answers don't know anything about what a neuron does. Magnitude pruning keeps the neurons with the largest weights. Data-driven pruning keeps the ones that fire hardest on some calibration text.

I wanted to know whether we get a better pruned model at the same size if we first find the neurons that matter for specific capabilities ("concepts") and protect them. I wanted to check whether that advantage holds on data the scoring never saw, on a different architecture, after recovery fine-tuning, and against stronger data-driven baselines.

The idea comes from [How Do Large Language Models Learn Concepts During Continual Pre-Training?](https://arxiv.org/abs/2601.03570) (Yao et al.), which studies "concept circuits" inside LLMs during training. That paper doesn't discuss pruning — using concept circuits to guide structured pruning is my own extension. The baselines and evaluation protocol come from Chapter 5 of Pere Martra's *Rearchitecting LLMs*.

## How it works

The method changes only one thing in a standard width-pruning pipeline: the score used to decide which neurons to keep. The removal mechanics, the number of neurons removed per layer, and therefore the final parameter count are identical to the baseline. Any difference in quality comes from *which* neurons survive.

### Where it hooks in: the GLU MLP

Llama, Qwen, and Gemma use gated (GLU) MLP blocks. `gate_proj` and `up_proj` expand the hidden state (2048 wide in Llama-3.2-1B) to the intermediate width (8192). The two are combined elementwise, and `down_proj` projects back to 2048. Removing intermediate neuron *i* means deleting row *i* of `gate_proj`, row *i* of `up_proj`, and column *i* of `down_proj`.

![Diagram of a GLU MLP block with a hook on the input to down_proj, which has one value per intermediate neuron]({{ '/assets/images/concept-pruning/mlp_hook.svg' | prepend: site.baseurl }})

The input to `down_proj` (call it Xd) has exactly one value per intermediate neuron, which makes it the natural place to measure importance. A forward-pre-hook there can either record the activation or replace it with a different tensor.

### Concepts as contrastive pairs

A concept is a set of **clean / corrupted prompt pairs** plus a target token. The clean prompt makes the target correct, and the corrupted prompt changes only the detail the concept depends on. Both prompts must tokenize to the same length, because the attribution step swaps activations between them position by position.


| Concept | Clean prompt → target                                        | Corrupted prompt        |
| ------- | ------------------------------------------------------------ | ----------------------- |
| IOI     | John and Mary went to the park. John gave a book to → `Mary` | … Mary gave a book to   |
| SVA     | The keys in the cabinet → `are`                              | The key in the cabinet  |
| Factual | The capital of France is → `Paris`                           | The capital of Japan is |


Training and held-out pairs use disjoint pools of names, places, nouns and countries, so held-out accuracy is never measured on anything the attribution step saw.

**More examples by use case.** The pattern is the same in every row. Change the one detail the capability depends on, keep everything else fixed, and the right answer changes. Pick the capability your pruned model must keep, then write pairs around it.


| Use case          | Capability protected                        | Clean prompt → target                                                                  | Corrupted prompt                                                          |
| ----------------- | ------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Translation       | Choosing the target language                | Translate to French: Thank you. → `Merci`                                              | Translate to Spanish: Thank you.                                          |
| Translation       | Word sense from context                     | He sat by the river bank. French: la → `rive`                                          | He went to the city bank. French: la                                      |
| De-identification | Entity type (person vs place)               | Sarah lives in Cairo. Cairo is a → `city`                                              | Sarah lives in Cairo. Sarah is a                                          |
| Classification    | Sentiment label                             | Review: The food was wonderful. Sentiment: → `positive`                                | Review: The food was terrible. Sentiment:                                 |
| RAG / document QA | Answering from the context, not from memory | Context: The meeting is on Tuesday. Question: When is the meeting? Answer: → `Tuesday` | Context: The meeting is on Friday. Question: When is the meeting? Answer: |
| Code              | Variable binding                            | `a, b = 3, 7; print(a) # prints` → `3`                                                 | `a, b = 3, 7; print(b) # prints`                                          |
| Structured output | Closing the right bracket                   | `values = [1, 2, 3` → `]`                                                              | `values = (1, 2, 3`                                                       |
| Arabic            | Factual recall in another language          | عاصمة مصر هي → `القاهرة`                                                               | عاصمة فرنسا هي                                                            |


These rows are illustrative and have not been run. Only IOI, SVA, factual, greater-than and arithmetic were tested (§4). Token lengths were not checked against a tokenizer, so verify each pair on your model: the loader drops pairs whose clean and corrupted prompts differ in length. If an answer spans several tokens, as the Arabic one probably does, use its first token as the target. One example per row is shown; a real concept needs dozens of pairs (60 were used here) plus a held-out set.

### Attribution: Integrated Gradients on Xd

For each pair, the model runs once on the corrupted prompt and once on the clean one, and the hook records Xd at every layer. Then it runs four more forward and backward passes on the corrupted prompt. In each pass, every layer's Xd is replaced by a point on the straight line from the corrupted activation to the clean one. The loss is the cross-entropy of the clean target token.

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


| Metric                 | Dense | Magnitude | Concept-aware |
| ---------------------- | ----- | --------- | ------------- |
| **Lambada perplexity** | 5.62  | 88.52     | **28.39**     |
| **Lambada accuracy**   | 0.635 | 0.300     | **0.445**     |
| **IOI (held-out)**     | 0.750 | 0.200     | **0.983**     |
| **SVA (held-out)**     | 0.600 | 0.450     | **0.533**     |
| **Factual (held-out)** | 0.817 | 0.000     | **0.567**     |


Both pruned models have 913.8M of the original 1,235.8M parameters. All metrics are on held-out data unvisited during attribution.

### Across pruning ratios

![Lambada perplexity by share of MLP neurons removed from 20% to 60%]({{ '/assets/images/concept-pruning/sweep.png' | prepend: site.baseurl }})


| Pruned  | Magnitude PPL | β = 2 PPL | β = 4 PPL  |
| ------- | ------------- | --------- | ---------- |
| **20%** | 30.78         | **9.90**  | 10.10      |
| **30%** | 58.74         | **17.40** | 17.73      |
| **40%** | 88.52         | **28.09** | 34.19      |
| **50%** | 388.84        | **91.70** | 95.78      |
| **60%** | 2634.41       | 627.68    | **579.31** |


β = 2 beats magnitude pruning on every metric at every ratio from 20% to 60%.

### A second architecture: Qwen2.5-1.5B

On Qwen2.5-1.5B at 40%, magnitude pruning is far more destructive than on Llama (Lambada perplexity 937 vs dense 6.1). Concept-aware cuts that to 243 and keeps IOI, greater-than, and factual recall well above the baseline. The effect is even larger here than on Llama.


| Metric           | Dense | Magnitude | Concept-aware |
| ---------------- | ----- | --------- | ------------- |
| **Lambada PPL**  | 6.13  | 937.46    | **242.6**     |
| **IOI**          | 0.417 | 0.250     | **0.433**     |
| **Greater-than** | 1.000 | 0.000     | **1.000**     |
| **Factual**      | 0.917 | 0.000     | **0.350**     |
| **SVA**          | 0.833 | 0.783     | 0.750         |




### How many concepts to protect


| Protected            | Lambada PPL | ARC-Easy  | IOI   | SVA   | Factual   |
| -------------------- | ----------- | --------- | ----- | ----- | --------- |
| **none (magnitude)** | 88.52       | 0.400     | 0.133 | 0.500 | 0.000     |
| **IOI**              | 24.19       | 0.440     | 1.000 | 0.700 | 0.233     |
| **IOI + SVA**        | **24.03**   | **0.440** | 1.000 | 0.733 | **0.733** |
| **+ factual**        | 29.65       | 0.415     | 1.000 | 0.833 | 0.717     |
| **+ arithmetic**     | 29.55       | 0.420     | 1.000 | 0.833 | 0.717     |


Protection spreads to concepts you didn't probe: factual recall jumped when I added SVA, not when I added factual itself. Adding a third concept cost about 23% in perplexity. Two concepts (IOI + SVA) was the best trade-off here.

### After recovery fine-tuning

Pruned models are usually fine-tuned afterwards to recover. I used the distillation recipe from *Rearchitecting LLMs* on Qwen3-0.6B at 10% pruning:


| Metric                  | Magnitude pre → post | Concept-aware pre → post |
| ----------------------- | -------------------- | ------------------------ |
| **ARC-Easy (n ≈ 2376)** | 0.486 → 0.535        | **0.563 → 0.593**        |
| **HellaSwag**           | 0.420 → 0.430        | **0.450 → 0.480**        |
| **Lambada acc**         | 0.140 → 0.300        | **0.230 → 0.360**        |
| **SVA**                 | 0.333 → 0.367        | **0.883 → 0.717**        |
| **Factual**             | 0.000 → 0.167        | **0.567 → 0.633**        |


On ARC-Easy at full size (about 2,400 questions), concept-aware won before and after recovery with non-overlapping 95% confidence intervals. Recovery narrows the general-benchmark gap, but the concept gap survives: generic recovery data doesn't rebuild a capability that pruning removed entirely.

## The package

*Code and reproduction scripts are available in the repository at [github.com/oraby8/concept-aware-pruning](https://github.com/oraby8/concept-aware-pruning).*

```
concept-prune run \
  --model unsloth/Llama-3.2-1B \
  --concepts examples/concepts_ioi_sva.yaml \
  --holdout-concepts examples/concepts_ioi_sva_holdout.yaml \
  --compression 0.4 --beta 2.0 \
  --benchmarks lambada_openai,arc_easy,piqa --limit 200 \
  --output results.json
```

```
# Python API
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

