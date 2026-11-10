---
layout: post
title: "How Much Does Arabic Cost? A Fair Comparison of 24 LLM Tokenizers on MSA and Dialects"
date: 2026-11-10T12:00:00+00:00
author: Ahmed Samir Oraby
categories:
  - llm
  - nlp
tags: [ArabicNLP, NLP, LLM, Tokenization, Machine Learning, AI, Research]
excerpt: "How efficient are today's popular LLM tokenizers on MSA and Arabic dialects when compared fairly? I evaluated 24 models across MSA and up to 13 dialects on parallel corpora, and trained controlled reference tokenizers to separate vocabulary size from training data."
---

In my last article, I reviewed five recent papers on Arabic tokenization and ended with an open question:
[https://www.linkedin.com/pulse/why-arabic-hard-tokenizers-three-core-challenges-ahmed-oraby-vidde?trk=public_post_feed-article-content](https://www.linkedin.com/pulse/why-arabic-hard-tokenizers-three-core-challenges-ahmed-oraby-vidde?trk=public_post_feed-article-content)

**How efficient are today's popular LLM tokenizers on MSA and dialects, compared fairly?**

So I measured it. I took 24 models (Llama 3 and 4, Qwen 2.5/3/3.5, Gemma 3 and 4, DeepSeek-V3, Aya/Command, gpt-oss, Jais, ALLaM, Fanar, Falcon-H1, Karnak, Atlas-Chat, Nile-Chat) and ran them on MSA and up to 13 Arabic dialects. I also trained my own reference tokenizers, to separate what comes from the vocabulary size from what comes from the training data.

Everything is open source: code, results, and the trained tokenizers.
👉 [https://github.com/oraby8/arabic-tokenizer-eval](https://github.com/oraby8/arabic-tokenizer-eval)

## What "fair" means here

Tokenizer comparisons are easy to get wrong:

- **Different texts:** if each model is tested on different text, the content gets measured, not the tokenizer.
- **Different word definitions:** "tokens per word" depends on how a word is defined.
- **Special tokens:** BOS/EOS markers and chat templates inflate counts.
- **Dataset labels:** a set labeled "dialect" isn't necessarily dialect.

So I used:

- **Parallel corpora:** the same meaning in every variety (FLORES+ and PADIC).
- **One preprocessing pipeline and one word splitter** for every tokenizer.
- **Plain token counts:** no special tokens and no chat templates.
- **Dialectness checked, not assumed:** every document was scored with ALDi, a model that rates how dialectal a sentence is.
- **Confidence intervals and paired tests** on every comparison.

This is a cost and efficiency study: how many tokens it takes to say the same thing. It is not a model-quality study.

## 1. Arabic still costs more than English, for most models

On the same FLORES+ sentences, this is how many Arabic (MSA) tokens each tokenizer needs per English token:

![Arabic vs English token cost for 15 tokenizers](https://raw.githubusercontent.com/oraby8/arabic-tokenizer-eval/main/results/blog_figures/1_arabic_vs_english.png)

In practice, a 128K context holds about **104K Arabic words with ALLaM but only 66K with Llama-3**. Llama-3, DeepSeek-V3 and Qwen2.5/3 are no better than raw UTF-8 bytes relative to English.

**Lesson:** for Arabic, the tokenizer can decide whether you get 1.6× more context and cost per word.

## 2. Vocabulary size doesn't matter; Arabic tokens do

Does a bigger vocabulary help? Barely:

- **Vocabulary size:** almost no correlation with efficiency (ρ = −0.28).
- **Number of Arabic tokens in the vocabulary:** almost perfect correlation (ρ = 0.96).

![Number of Arabic tokens in the vocabulary vs MSA characters per token](https://raw.githubusercontent.com/oraby8/arabic-tokenizer-eval/main/results/blog_figures/2_arabic_tokens_vs_efficiency.png)

For example:

- **Llama-4** has a 201K vocabulary, but only 2.2% of it is Arabic. It is barely better than Llama-3.
- **Jais-13B** (85K) beats almost every 250K vocabulary.

**Lesson:** "large multilingual vocabulary" says nothing about Arabic. Count the Arabic tokens.

## 3. Many "Arabic models" don't have an Arabic tokenizer

This surprised me. Several Arabic-focused models reuse a general tokenizer **unchanged**:

| Model | Tokenizer |
|---|---|
| Fanar-2, Nile-Chat (Egyptian) | Gemma-3 |
| Atlas-Chat (Moroccan) | Gemma-2 |
| Command-R7B-Arabic | the same as Command-R / Aya |

So the Egyptian and Moroccan models get **no tokenizer advantage on the dialect they target**.

The positive exception is **Karnak**. It extends Qwen3's vocabulary with Arabic tokens, which makes Arabic **39% more compact** than with plain Qwen3.

**Lesson:** "Arabic model" and "Arabic tokenizer" are separate claims. Check the tokenizer.

## 4. Every tokenizer pays a dialect penalty, and the Maghreb pays the most

Dialect text always costs more tokens per character than MSA:

- **Moroccan is the worst dialect for every tokenizer** on FLORES+.
- **Algiers and Annaba are the worst** in the parallel conversational data (PADIC).
- **The MSA-focused Arabic tokenizers (ALLaM, Jais) lose the most on dialects.** They are still the most efficient on dialect text in absolute terms.

**Lesson:** a tokenizer that is excellent on MSA can be noticeably worse on dialects. Evaluate both.

## 5. The controlled experiment: data beats vocabulary size

To find the cause, I trained my own BPE and Unigram tokenizers. Everything is identical (1B training characters, same algorithm, same vocabulary size) except **whether half the training text is dialect**:

![Dialect penalty for tokenizers trained on MSA only vs MSA plus dialect](https://raw.githubusercontent.com/oraby8/arabic-tokenizer-eval/main/results/blog_figures/3_dialect_training_experiment.png)

The dialect penalty **drops by about half** (BPE: +31% → +17%), and MSA efficiency drops by only 2.6–4.2%. The improvement also **carries over to dialects that were never in the training data** (Levantine, Mesopotamian, Tunisian).

Making the vocabulary bigger doesn't do this. From 32K to 128K, MSA efficiency rises a lot, but the dialect penalty stays the same (+16%).

**Lesson:** if you want a tokenizer that handles dialects, put dialect text in the tokenizer's training data. A bigger vocabulary won't fix it.

## 6. Normalization: AraToken was right, but only halfway

AraToken reported that normalization (unifying Alif variants, removing tatweel, mapping digits) reduces fertility. I tested two settings:

- **Normalizing only the input to an existing tokenizer:** tokens usually go **up**. Light normalization adds up to +2.6% on MSA, and aggressive normalization (including ة→ه) adds up to +12%.

  The vocabularies learned "إلى" and "مدرسة" as single tokens, so "الى" and "مدرسه" split into more pieces.
- **Training the tokenizer on normalized text too:** tokens go **down**, but only by **1.3–2.7%**.

Aggressive normalization saves less than 1 point more than light normalization, while erasing real distinctions such as ة/ه and the hamza seats.

**Lesson:** normalize at training time or not at all, and prefer light over aggressive normalization. The efficiency gain is small; the information loss is real.

## 7. Two practical surprises: tweets and diacritics

**Tweets:**

- **URLs alone take 12–18% of the tokens.**
- Removing URLs, mentions, hashtag marks and emoji saves **21–34%**.
- Elongation (جمييييل) appears in 20% of tweets but costs only about 1% of tokens.

**Diacritics multiply the token count by 2–5×** for the models tested, and up to 8× for tokenizers trained without diacritics:

![Extra tokens for fully diacritized text, per tokenizer](https://raw.githubusercontent.com/oraby8/arabic-tokenizer-eval/main/results/blog_figures/4_diacritics_cost.png)

- **Qwen3.5** handles them best: +108%, more than twice as good as Qwen2.5/3 (+239%).
- **Jais-adapted (+377%) and Karnak (+344%)** are among the worst.
- Unicode NFC reorders shadda and fatha, which costs some tokenizers up to 6 more points.

**Lesson:** if your use case involves diacritized text (Quran, poetry, education, TTS), test diacritized text specifically. MSA results won't tell you.

## 8. Warnings about the data

These cost me the most time, and they may save you time:

- **FineWeb-2's "Moroccan" subset (`ary_Arab`) is about 98% MSA news.** Its dialect labels can't be trusted without checking them.
- **The FLORES+ Najdi set reads as MSA** by the dialectness score.
- **Egyptian and Moroccan Wikipedia are about 50% bot-generated templates.** One tokenizer had memorized them, which made Egyptian look *cheaper* than MSA.
- **The Hugging Face repo named "MADAR"** (`IQSeC-Lab/MADAR`) is a malware dataset, not the Arabic dialect corpus.

**Lesson:** check dialectness and duplication before measuring anything.

## What We Learned

1. **The Arabic share of the vocabulary predicts efficiency.** Total vocabulary size doesn't.
2. **Check the tokenizer, not the model's name.** Many Arabic models reuse a general tokenizer.
3. **Dialect coverage comes from the training data**, not from adding vocabulary entries.
4. **Normalization is a training decision,** not a preprocessing trick, and its gain is small.
5. **Verify your data.** Dataset labels, translations and Wikipedias can all mislead.

## Open Questions

- Does lower token cost translate into better models? I didn't measure model quality. Bits per character and benchmarks like Belebele are the next step.
- How do closed models (Claude, Gemini, newer ChatGPT models) compare? Their tokenizers can only be measured through their APIs.
- Would a morphology-aware tokenizer (MorphBPE, SPLINTER) keep its morphology benefits on dialects?

Code, results and the trained tokenizers: [https://github.com/oraby8/arabic-tokenizer-eval](https://github.com/oraby8/arabic-tokenizer-eval)

If you work on Arabic tokenization or have dialect data, I'd love to hear from you.
