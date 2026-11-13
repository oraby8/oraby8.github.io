---
layout: post
title: "Which Open LLM Already Speaks Egyptian, Saudi and Gulf Arabic? A Fair Test of 21 Models"
date: 2026-11-13T12:00:00+00:00
author: Ahmed Samir Oraby
categories:
  - llm
  - nlp
tags: [ArabicNLP, NLP, LLM, ArabicDialects, Machine Learning, AI, Research]
excerpt: "Which open model is the best starting point for Egyptian, Saudi and Gulf Arabic? I evaluated 21 open LLMs from 1B to 40B using bits per character (BPC), token fertility, and multiple-choice benchmarks across translated text, web text, and spontaneous speech."
---

My last post compared the **tokenizers** of 24 LLMs on MSA and Arabic dialects: how many tokens it takes to say the same thing.
👉 [https://github.com/oraby8/arabic-tokenizer-eval](https://github.com/oraby8/arabic-tokenizer-eval)

A cheap tokenizer is only half the story. Before you spend GPUs on continued pretraining or fine-tuning for a dialect, you want to start from the model that **already knows that dialect best**. So the next question was:

**Which open model is the best starting point for Egyptian, Saudi and Gulf Arabic?**

I tested 21 models, from 1B to 40B:

- **Global:** Llama 3.1/3.2, Qwen 2.5/3/3.5, Gemma 3 and 4, Falcon-H1, Aya Expanse, Command-R, gpt-oss.
- **Arabic-centric:** Jais, ALLaM, Fanar 1 and 2, Karnak, Command-R7B-Arabic, Atlas-Chat, Nile-Chat.

Every dialect is ranked **on its own**: a model that is great at Egyptian isn't necessarily great at Gulf.

Code and per-document results are open source:
👉 [https://github.com/oraby8/arabic-base-model-eval](https://github.com/oraby8/arabic-base-model-eval)

## The metric: bits per character

The main score is **bits per character (BPC)**: how surprised the model is by real dialect text, measured per character. Lower is better. It is the most direct measure of what a model knows about a language, and it is exactly what continued pretraining improves.

Dividing by characters instead of tokens matters. A model whose tokenizer cuts Arabic into many small pieces doesn't look better or worse because of it. Every model reads the same text, so the scores are directly comparable.

I also report:

- **Fertility** (tokens per word): your training and inference bill.
- **Multiple-choice accuracy:** Belebele for Egyptian and Najdi, and the new **Alyah** benchmark for Emirati.
- **Paired bootstrap confidence intervals** on every gap between models, so "better" means statistically better.

## The test data

| Dialect | Text |
|---|---|
| **Egyptian** | FLORES+ (translated news), real Egyptian web text from FineWeb-2, and transcribed spontaneous speech from 15 speakers (Meta's Omnilingual ASR corpus) |
| **Saudi** | FLORES+ Najdi, Najdi web text, and Najdi and Hijazi speech |
| **Gulf** | Gulf speech (Omnilingual), Emirati TV and podcast transcripts, and Emirati radio |
| **MSA and English** | References |

Web text was filtered with ALDi, a model that rates how dialectal a sentence is, so the "dialect" text really is dialectal. That gives 4,151 documents and 4.2M characters in total, and no model ever truncates a document.

## Getting the measurement right

The usual quick script for this (`labels=input_ids`, divide by the decoded length, truncate at 2,048 tokens) is not comparable across models. I made three fixes:

1. **Every character is scored.** The first token of a text is never predicted, so a model with long first tokens gets free characters.
2. **The denominator is the original text,** not a decoded copy that some tokenizers normalize.
3. **No truncation.** With a fixed token limit, a model with a less efficient tokenizer is scored on less text.

The surprise was the **start token**: what the model sees before the first word. It changes the score far more than I expected:

| Model | Bad start token | Good start token |
|---|---|---|
| Gemma-3-12B | 2.85 BPC (end-of-text) | **1.32** (its BOS) |
| Llama-3.2-1B | 3.95 (end-of-text) | **2.03** (its BOS) |
| Qwen3-8B-Base | 1.98 (`<|endoftext|>`) | **1.62** (a newline) |

Any single fixed rule would favour some models over others. So each model tries four start tokens (its default, BOS, EOS, a newline) on separate calibration text and keeps its best one. My first full run used a fixed rule, and Qwen3 and Karnak looked much worse than they really are.

**Lesson:** if you compare models by perplexity or BPC, check the start token. It can change the score by a factor of two.

## 1. One model wins every dialect: Gemma-4-31B

![BPC vs model size for Egyptian, Saudi and Gulf](https://raw.githubusercontent.com/oraby8/arabic-base-model-eval/main/results/bpc_vs_size.png)

| | Egyptian | Saudi | Gulf / Emirati |
|---|---|---|---|
| 1 | **Gemma-4-31B** (1.294) | **Gemma-4-31B** (1.277) | **Gemma-4-31B** (1.502) |
| 2 | Command-R-32B (1.323) | Command-R-32B (1.326) | Command-R-32B (1.569) |
| 3 | Fanar-2-27B (1.350) | Fanar-2-27B (1.343) | Fanar-2-27B (1.591) |
| 4 | Gemma-3-12B (1.396) | Gemma-3-12B (1.386) | Gemma-3-12B (1.635) |
| 5 | ALLaM-7B (1.416) | ALLaM-7B (1.410) | ALLaM-7B (1.684) |

Gemma-4-31B is first in all three dialects and in MSA. Every other model is significantly behind: no confidence interval of a gap touches zero. Its lead is widest on **Gulf**, the least-resourced dialect, and it also has the best Emirati multiple-choice score (Alyah, 50.6%).

The next two have a catch for continued pretraining:

- **Command-R-32B** only has a public chat checkpoint, and Cohere releases it for non-commercial use.
- **Fanar-2-27B** is also chat-only (it is built on Gemma-3-27B). It has the best Belebele scores (52.4% Egyptian, 49.9% Najdi), but that lead is within the ±3-point noise.

**Lesson:** at around 30B, Gemma-4-31B is the clear starting point for all three dialects, and its base checkpoint is public.

## 2. At 7–12B: Gemma-3-12B, or ALLaM-7B for cheaper tokens

- **Gemma-3-12B** is the best *base* checkpoint below 30B in every dialect.
- **ALLaM-7B** is right behind it (+0.02 to +0.05 BPC), with 40% fewer parameters.
- **ALLaM's tokenizer** needs **1.43 tokens per Egyptian word against 1.88 for Gemma**, about 25% fewer tokens for the same text. That saving applies to every training step and every inference call.
- **Fanar-1-9B** (base) is the next option.

**Lesson:** at mid size, choose between the best base model (Gemma-3-12B) and the cheapest tokens (ALLaM-7B). The quality gap between them is small.

## 3. Arabic-centric beats global, but not every Arabic model

At the same size, the best Arabic-centric models (ALLaM-7B, Fanar-1-9B) beat Qwen, Llama and Falcon-H1 by **0.15–0.35 BPC on dialects**. That is a big gap: going from Gemma-3-12B up to Gemma-4-31B gains only 0.10.

But the "Arabic" label isn't a guarantee:

- **Karnak-40B** is the largest model tested, but ranks 13th on Egyptian, behind 7–12B models. Its Arabic tokens were added on top of Qwen3's vocabulary.
- **Karnak-6B** and **Atlas-Chat-9B** (built for Moroccan Darija) are near the bottom for all three dialects.
- **Qwen3 is not better than Qwen2.5 on dialects,** and Qwen3.5 is only slightly better.

**Lesson:** check, don't trust the label. Size and an "Arabic" name predict dialect knowledge less well than how much real Arabic a model was trained on.

## 4. Translated benchmarks flatter some models

FLORES+ dialect text is professionally translated news. Compare it with real web text and real speech:

| Egyptian BPC | FLORES (translated) | Web | Speech |
|---|---|---|---|
| Gemma-4-31B | 1.300 | 1.306 | 1.275 |
| Gemma-3-12B | 1.391 | 1.423 | 1.374 |
| ALLaM-7B | **1.332** | 1.446 | 1.469 |
| Nile-Chat-12B | 1.378 | 1.509 | 1.426 |

On FLORES, ALLaM looks clearly better than Gemma-3-12B. On real Egyptian web text and speech, it's the other way round. Saudi shows the same pattern, and more strongly: on FLORES "Najdi", every model scores almost the same as on MSA (ratio about 1.02). The translation is close to MSA, so it barely tests the dialect at all.

**Lesson:** test on real dialect text, ideally spontaneous speech. Translated benchmarks reward formal-Arabic knowledge and hide the dialect gap.

## 5. Specialised models: real gains, narrow ones

**Nile-Chat-12B**, built for Egyptian, has the **smallest Egyptian-vs-MSA gap** of all 21 models on the same sentences: 1.085, against about 1.19 for Gemma. It clearly learned Egyptian specifically, but its MSA and Gulf scores are much weaker, and it ranks 6th on Egyptian overall.

**Lesson:** a dialect-specialised model learns the dialect, but a strong general Arabic base model still predicts real Egyptian text better.

## My recommendation

| If you can train… | Egyptian | Saudi | Gulf / Emirati |
|---|---|---|---|
| **About 30B** | Gemma-4-31B | Gemma-4-31B | Gemma-4-31B |
| **About 7–12B** | Gemma-3-12B, or ALLaM-7B | Gemma-3-12B, or ALLaM-7B | Gemma-3-12B, or ALLaM-7B |
| **About 1B** | Jais-3-0.9B | Jais-3-0.9B | Jais-3-0.9B |

## Caveats

- **Chat checkpoints** (Command-R, Fanar-2, ALLaM, Nile-Chat and others) are scored on raw text. Instruction tuning usually makes raw-text scores somewhat worse, so their base models would likely rank a little higher.
- **Gulf data is small** (550K characters, all speech), so its ranking is clear but its absolute numbers aren't comparable with Egyptian or Saudi.
- **Contamination:** FLORES+, Belebele and FineWeb-2 are public and may be in some models' training data. The speech transcripts are the least likely to have been seen.
- **Not tested:** Llama-4-Scout, DeepSeek-V3 and Command-A were too large for my server's disk.

## What We Learned

1. **Gemma-4-31B is the best starting point for Egyptian, Saudi and Gulf Arabic,** with a public base checkpoint.
2. **At 7–12B, Gemma-3-12B and ALLaM-7B** are the options. ALLaM's tokenizer is about 25% cheaper.
3. **The start token can double a model's score.** Calibrate it per model before comparing perplexities.
4. **Translated benchmarks hide the dialect gap.** Real web text and speech change the ranking.
5. **An "Arabic" label and size don't guarantee dialect knowledge.** Measure it.

## Open Questions

- How much continued pretraining on real dialect text does it take to close the gap between ALLaM-7B and Gemma-4-31B?
- Does the cheaper ALLaM tokenizer pay off once you count total training cost, not just the starting quality?
- Which models would rank differently with base checkpoints instead of chat checkpoints?

Code, data recipes and per-document results:
👉 [https://github.com/oraby8/arabic-base-model-eval](https://github.com/oraby8/arabic-base-model-eval)
