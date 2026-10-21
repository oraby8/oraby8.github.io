---
layout: post
title: "From Tashkeel-50M to Tashkeel-v4: How We Took an Arabic Diacritizer from 15.3% to 1.55% Error"
date: 2026-10-21T08:00:00+00:00
author: Ahmed Samir Oraby
categories:
  - nlp
  - speech
tags: [Arabic, Diacritization, Tashkeel, NLP, MARBERT, CRF, Machine Learning]
excerpt: "Arabic is normally written without short vowels. We built four generations of an Arabic diacritization model, dropping error from 15.3% to 1.55%. Here is what we changed in each generation and which changes actually moved the numbers."
---

*Four generations of an Arabic diacritization model: what we changed in each one, and which changes actually moved the numbers.*

---

## TL;DR

Arabic is normally written without short vowels. **Tashkeel** (diacritization) is the task of putting them back: `كتب` → `كَتَبَ`. It matters for TTS, search, learner tools, and any NLP that needs to disambiguate words.

We built four generations of the model. Each one added something specific:

| Model | What we added | Sadeed DER / WER | CATT DER / WER |
|---|---|---|---|
| **Tashkeel-50M** | A 50M causal language model that generates the diacritized sentence | 15.29 / – | – |
| **Tashkeel-v2** | Switch to a **character tagger**: MARBERTv2 encoder, morphology embeddings, separate case-ending head, BiLSTM, **CRF + Viterbi**, large multi-corpus training | 2.857 / 7.92 | 6.373 / 17.40 |
| **Tashkeel-v3** | **Clean, complete data** (Tashkeela classical + MSA), **self-training on news**, continued training with **weight averaging**, an `إ` fix, and a fixed **measurement** protocol | 2.261 / 6.63 | 4.983 / 14.04 |
| **Tashkeel-v4** | **LLM-teacher labels** on hard news sentences, more self-training rounds, hard classical text selection | **1.552 / 5.25** | **2.952 / 10.37** |

*(DER = diacritic error rate, WER = word error rate, both from the benchmark authors' official scorer. Lower is better.)*

The surprise is that most of the later gains did **not** come from architecture. Tweaking the architecture gave nothing once we controlled for noise. The gains came from measuring honestly, cleaning data, and adding the right kind of new data.

---

## 1. The task and the benchmarks

Two benchmarks drove every decision:

- **SadeedDiac-25** (`Misraj/SadeedDiac-25`): 1,200 sentences, classical-leaning, in-domain for our training data.
- **CATT benchmark** (`Bisher/CATT_benchmark`): 742 modern news sentences, out-of-domain for us.

Having one in-domain and one out-of-domain benchmark was valuable. Several "improvements" helped one and hurt the other, and you only see that if you look at both.

---

## 2. Tashkeel-50M: the generative baseline

The first model, **Tashkeel-50M**, was a 50M-parameter **causal language model**: give it the undiacritized sentence and let it generate the diacritized one.

It scored **15.29 DER on Sadeed**, and that understates the problem: **86% of its outputs dropped or invented characters**, so the output could not even be aligned with the input.

**Lesson: diacritization should be tagging, not generation.** If the output must be exactly the input letters plus marks, make the model predict **one label per input character**. Then it cannot change the text.

---

## 3. Tashkeel-v2: a character tagger with a CRF (Sadeed 2.857, CATT 6.373)

v2 changed the problem formulation, then stacked several components, each of which helped:

| Addition | Why it helps | Sadeed DER after |
|---|---|---|
| **Character tagger** (one label per character, CAMeL-BERT encoder, 1M sentences) | Output can never change the text | 5.05 (from 15.29) |
| **MARBERTv2 encoder + separate case-ending (iʿrāb) head**, case-ending loss weighted 2.5x, word-position embeddings | Case endings depend on syntax, internal vowels on morphology, so each gets its own head and loss | 3.75 |
| **Multi-corpus training** (Sadeed + Shamela + Quran, 1.41M) | More diverse text | 3.31 |
| **Prefix / stem / suffix embeddings** | Gives the model the clitic structure of the word | 3.18 |
| **5M deduplicated corpus, including poetry** | More data, with duplicates removed so the benchmarks stay clean | 2.92 |
| **Fine-tuning on the Sadeed Tashkeela training set** | Matches the target convention | 2.855 |
| **Linear-chain CRF with Viterbi decoding** | Keeps predicted marks consistent with each other; best CATT score | 2.857 (CATT 6.37) |

### The v2 architecture

- **MARBERTv2** encoder (768d) for sentence context.
- Character, word-position, and morpheme embeddings concatenated with it.
- A 2-layer BiLSTM with a residual projection.
- Separate heads for internal vowels (stem) and word-final case endings.
- A **linear-chain CRF** with Viterbi decoding over 15 diacritic classes.

### Things that did not help at this stage (we stopped doing them)

- A bigger encoder (370M AraBERT-Large) was *worse* than MARBERTv2.
- Extra part-of-speech and active/passive heads, and a two-pass refiner, gave no gain.
- A morphological-dictionary signal: replacing it with random noise gave the same accuracy, so the "gain" was never from the dictionary.
- A word-pattern (wazan) auxiliary task had no effect, or hurt.

---

## 4. Before v3: fix the measurement

Before improving the model, we audited how we measured it. Four problems turned up, and fixing them changed every later decision:

**4.1 Two scorers were in use.** A rewritten copy of the metric script had drifted from the original, and it treated "fully diacritized" differently. Re-scoring the *same* checkpoint gave 6.37 with one and 6.19 with the other. Several reported "improvements" were just a scorer change.

> **Rule:** use the benchmark authors' original scorer, hash it (MD5), and never edit it.

**4.2 Error analyses that did not match the metric.** One analysis said 39% of CATT errors were "no mark in the reference, model predicts fatha". Under the benchmark's setting those positions cost nothing, so we had been chasing error mass the metric ignores.

**4.3 Checkpoints selected on the test set.** Choosing the best epoch by benchmark score leaks the test set. We switched to a rule fixed before each run: **final checkpoint of each seed, then average two seeds.**

**4.4 No error bars.** Differences of 0.02 DER are common between seeds. We added **paired bootstrap confidence intervals** to every comparison, and counted a result only if the 95% interval excluded zero. That one habit killed about a dozen "improvements".

> **Rule:** always compare against a *matched control* (same compute, same replay data, same seeds) and report a paired CI.

---

## 5. Tashkeel-v3: clean data, self-training, averaging (Sadeed 2.261, CATT 4.983)

Everything in v3 is about data and training procedure. The architecture is unchanged from v2.

### 5.1 Continued training and weight averaging

Continued training from the best checkpoint, then **averaging the weights** of several runs that share one initialization. Averaging is nearly free and was the most reliable tool in the project. One caveat: the seeds must start from the *same* head initialization, otherwise averaging degrades the model.

### 5.2 Complete diacritization, not more data

We had big external corpora (Tashkeela 0.3 classical text, Shamela). At first, adding them made the model **worse**. The cause was not domain mismatch. Many sentences were only partly diacritized: 5 to 22% of final letters had no mark. A model trained on that learns that "no mark" is often correct, and starts leaving letters unmarked.

The fix was a filter applied to every dataset:

1. **Completeness:** all final letters marked, at most 2% of interior letters unmarked.
2. **Wasl normalization:** the connecting alef (همزة الوصل) is written in different conventions across corpora, so normalize it.
3. **Canonical mark order:** shadda+vowel can be stored either way, so pick one.
4. **10-gram dedup** against the training set *and* the benchmarks and dev set, so nothing leaks.

With that filter, the same corpora **helped**. Lesson: audit data quality before adding more data, and always check for benchmark overlap.

### 5.3 Modern Standard Arabic for CATT

CATT is modern news, and our data was mostly classical. Two MSA sources closed part of the gap:

- **Filtered Wikipedia (29k sentences):** we kept only sentences where the reference diacritics were within 3% DER of the model's own output, which drops bad references.
- **Self-training on news (SANAD, 579k sentences):** the model labels unlabeled sentences, and we keep only those where the labels agree with an *independent older model* on at least 97% of words. Agreement is a cheap confidence filter. Two rounds (the second relabeling all 579k with the improved model, 153k kept) gave the first big CATT jump.

Each round is: label → filter → train a treatment against a control with the same replay data → average two seeds → compare with paired CIs.

### 5.4 The hamza-below fix

External data had taught the model to leave `إ` unmarked. A tiny post-processing rule fixes it:

```python
import re
text = re.sub(r"إ(?![ً-ْ])", "إِ", text)
```

**Result of v3:** Sadeed 2.857 → 2.261 and CATT 6.373 → 4.983. On the CATT paper's own scoring protocol, v3 beat all 13 published systems when case endings are counted.

---

## 6. Tashkeel-v4: a stronger teacher and hard examples (Sadeed 1.552, CATT 2.952)

By the end of v3, error analysis showed that most remaining mistakes were inside rare words that the model had seen at most 50 times. More of the same data was not going to fix that. We needed *new labeled text on exactly the hard cases*.

### 6.1 Distillation on unlabeled news

We distilled the model on unlabeled news. That moved Sadeed from 2.337 to 1.801 and CATT from 4.992 to 3.650.

### 6.2 Free self-training rounds

With the teacher data in hand, the improved model relabeled the news corpus again. Two more rounds gave small but significant gains (Sadeed 1.801 → 1.630, CATT 3.650 → 3.100), with clear diminishing returns. Interestingly, much of the first round's gain came from simply making *more passes over the teacher data*.

### 6.3 Hard classical text, with the free references

On classical text, the paid teacher labels were **no better** than the free human references. So for classical text we used the references, and used the model to select the **hard** chunks (where its output disagreed with the reference). That brought the final numbers to Sadeed **1.552 / WER 5.25** and CATT **2.952 / WER 10.37**. A second round added only about 0.02 to 0.03, so we stopped.

On the CATT paper's own protocol, v4 scored 6.562 / 27.463 with case endings and 5.915 / 25.283 without, the best of all systems on all four columns.

---

## 7. Model releases

- **[Tashkeel-v3](https://huggingface.co/ahmedsamirtarjama/Tashkeel-v3)** on Hugging Face Hub
- **[Tashkeel-v4](https://huggingface.co/ahmedsamirtarjama/Tashkeel-v4)** on Hugging Face Hub *(gated, research-only license)*

---

*All numbers are from the benchmark authors' official scorer. Thresholds for data selection were fixed before each run.*
