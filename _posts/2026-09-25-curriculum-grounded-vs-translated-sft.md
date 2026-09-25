---
layout: post
title: "Does Culture Need to Come From the Curriculum? Testing a Claim About Arabic LLM Data"
date: 2026-09-25T12:00:00+02:00
author: Ahmed Samir Oraby
categories:
  - llm
  - nlp
tags: [LLM, Arabic, Fine-Tuning, Evaluation, NLP, SFT, CuCu]
excerpt: "A recent paper argues you can't teach an LLM culture by translating English instruction data — you have to start from an authoritative local source like a national curriculum. I ran a same-model, same-method A/B test to see if that holds up for Saudi Arabic."
---

## The claim

A recent paper — *"From National Curricula to Cultural Awareness: Constructing Open-Ended Culture-Specific Question Answering Dataset"* — makes a pointed argument about how to teach large language models about culture:

> Don't try to teach an LLM culture just by translating English instruction data. Instead, start from authoritative local knowledge — such as a national curriculum — and systematically turn it into culturally grounded SFT data.

It's a clean, falsifiable claim, and it's the foundation of **CuCu**, a project I'm working on that adapts this exact methodology to Saudi Arabic: extracting learning outcomes from OCR'd Saudi textbooks, generating colloquial Najdi/Hijazi queries from those outcomes, generating multi-level Arabic answers, and gating everything through a sensitivity rubric before it becomes training data.

I wanted to know: does this actually work better than the thing everyone already does — take an English instruction dataset and translate it?

So I built both, fine-tuned the same base model on each with identical hyperparameters, and evaluated both on held-out data. This post walks through the pipeline and the result.

## Why a naive baseline doesn't test the claim

My first instinct was to translate the curriculum-grounded Arabic data to English and back to Arabic, and compare that to the original. That's the wrong experiment — round-tripping the *same* content through translation only measures translation noise, not whether the underlying content's *source* matters. Both arms would still be curriculum-sourced.

To actually test the paper's claim, the content itself has to come from somewhere else: a generic instruction/QA dataset that was never touched by the Saudi curriculum, translated into Arabic. That's the real-world practice the paper is arguing against.

## Building the translated-instruction baseline

I went looking for an English-language QA dataset that was topic-matched to CuCu's Saudi dataset (which turned out to be ~74% Qur'an/tajweed/Islamic studies content) but *not* derived from any curriculum. I settled on [**QCRI/IslamicFaithQA**](https://huggingface.co/datasets/QCRI/IslamicFaithQA) — 3,808 English, open-ended QA pairs sourced from general Islamic knowledge bases (IslamTrust, QIAS2025, PalmX), not from any textbook.

I translated it to Arabic with `google/gemma-3-12b-it`, running on a rented H100, with an explicit instruction to render Arabic-origin technical terms (*mudarabah*, *wakalah*, etc.) in proper Arabic script rather than leaving them transliterated in Latin letters. That still slipped through on about 6–7% of rows — mostly specialist Islamic-finance jargon — so I ran an automated Latin-script scan and excluded flagged rows from the training set.

Two data-quality issues surfaced along the way, both worth naming honestly:

- **11 rows in the source dataset itself** were malformed (JSON parse errors baked into QCRI's own generation pipeline, with no question/answer content at all). Skipped, not translated.
- **At least one row was straight-up contaminated** — an English "Islamic QA" row whose actual content was a video-game trivia question about *Sid Meier's Railroads*. The translation model translated it faithfully; the bug was upstream in the source data. Caught by the same script-purity scan and excluded.

After translation and filtering: **3,639 clean Arabic QA pairs** — call this **Model T**'s training data (T for *translated*).

## The curriculum-grounded side

For the other arm, I used CuCu's own pipeline output — but restricted to the same Islamic-studies subject slice (تجويد, القرآن الكريم والدراسات الإسلامية, القرآن الكريم وتفسيره, الحديث) so the topic domain matches Model T as closely as possible. That's **4,172 training examples** — call this **Model C** (C for *curriculum*).

Both datasets went through the same leakage-safe split: grouped by underlying outcome/source ID and stratified by subject, so paraphrases or variants of the same question can't leak across train/test.

## Fine-tuning: same model, same method, only the data differs

Both models are LoRA fine-tunes of the *same* base model (`google/gemma-3-12b-it`, r=16, α=32, identical learning rate, batch size, and 3 epochs), trained on a single rented H100. The only thing that changes between Model C and Model T is which dataset they saw.

| | Model C (curriculum) | Model T (translated) |
|---|---|---|
| Train examples | 4,172 | 3,274 |
| Training loss: start → end | ~1.02 → ~0.53 | ~2.08 → ~0.33 |

(Model T's lower final loss doesn't mean it "learned better" — its data is more uniform/formulaic academic Q&A, which is inherently easier to fit than Model C's paraphrase- and dialect-varied colloquial data. Loss magnitude isn't comparable across datasets with different intrinsic entropy.)

## Evaluation: cross-domain, not just in-domain

Fine-tuning is not enough by itself — I evaluated **both models on both held-out test sets**, not just each model on its own test set. This is the part that actually isolates the claim:

- Model C, evaluated on curriculum-style (colloquial Saudi) queries — *in-domain*
- Model C, evaluated on translated-style (formal academic) queries — *out-of-domain*
- Model T, evaluated on translated-style queries — *in-domain*
- Model T, evaluated on curriculum-style queries — *out-of-domain*

1,658 generations total, scored with CuCu's own sensitivity rubric — a Likert judge (`openai/gpt-oss-120b`) plus four hard content gates (`religious_ok`, `governance_ok`, `regional_naming_ok`, `historical_framing_ok`). A row passes only if it clears all four gates *and* scores ≥7 on both cultural appropriateness and language use.

## Results

![Pass rate comparison: Model C scores 87.5% on curriculum-style queries vs Model T's 55.4% on the same test set; Model T scores 90.4% on translated-style queries vs Model C's 70.1%]({{ '/assets/images/cucu/pass_rate.png' | prepend: site.baseurl }})

The comparison that actually tests the paper's claim is the two bars on the left: **both models, evaluated on the same curriculum-style test set.** Model C — trained on curriculum-grounded data — passes **87.5%** of the time. Model T — trained on translated instruction data — passes **55.4%**. Same base model, same fine-tuning method, same evaluation. The only variable is where the training data came from.

![Two panel chart: cultural appropriateness scores are close between models, but language/register fit drops sharply for Model T on curriculum-style queries (6.81 vs 8.14)]({{ '/assets/images/cucu/quality_metrics.png' | prepend: site.baseurl }})

Breaking the pass/fail down by metric shows *why*. Cultural appropriateness is close across the board (7.7–9.4 out of 10 in every cell) — both models mostly avoid saying something outright wrong or inappropriate. The real gap is in **language and register fit**: Model T's score collapses to **6.81** when it has to answer a colloquial Najdi/Hijazi query, versus 8.14–8.42 everywhere else. It answers correctly in substance but in stiff, formal MSA — it reads like a translation, because in a sense, its training data was one.

There's also an asymmetry in how much each model degrades leaving its home turf: Model C drops 17.4 points moving out-of-domain (87.5% → 70.1%); Model T drops 35 points (90.4% → 55.4%). Model T is *more* brittle outside its narrow training distribution — curriculum grounding looks like it buys more than just a good fit to its own test set.

## What this doesn't prove

I want to be upfront about the limits here, because it's easy for a clean bar chart to overstate certainty:

- **Single run, no significance testing.** No multiple seeds, no confidence intervals. The gap is large (32 points) but I haven't quantified how much of that is noise.
- **~365–464 examples per cell.** Not huge.
- **The judge itself is imperfect.** Earlier validation of this same sensitivity-rubric judge against an independent human rater landed at Cohen's κ = 0.43 on cultural appropriateness — below the project's own 0.61 bar for "production-ready." That doesn't invalidate the comparison (the judge is applied identically to both models, so it's not obviously biased toward one), but it means the exact numbers deserve a human-rated spot check before anyone treats this as final.
- **One base model, one fine-tuning method, one language pair.** This is a single data point, not a sweep.

What I think this *does* show: the paper's claim survives a same-model, same-method, cross-domain A/B test on a real, if small, evaluation. The mechanism is legible too — it's not vague "cultural vibes," it's a concrete, measurable register mismatch that shows up specifically when a translation-trained model has to handle the way people actually talk.

## What's next

- Independent human rating on a sample of the disagreement cases, especially Model T's low-register-fit rows, to sanity-check the judge.
- A non-Islamic-subject slice of the curriculum data (social studies, life skills) to see if the gap holds outside religious content.
- A second translated baseline built from a *generic* instruction set (not topic-matched at all) to see how much of Model T's shortfall is "translated" versus just "topic-mismatched."

---

*This work is part of a Saudi Arabic adaptation of the CuCu methodology — the culture-specific QA dataset construction pipeline first published against Korean curricula as KCaQA. The Saudi pipeline (learning-outcome extraction, query generation, multi-level response generation, and a sensitivity rubric) runs on Saudi OCR textbook corpora with a hard content gate before anything ships as training data.*
