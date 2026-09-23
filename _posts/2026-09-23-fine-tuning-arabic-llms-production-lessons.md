---
layout: post
title: "Fine-Tuning Arabic LLMs: Practical Engineering Lessons from Production Pipelines"
date: 2026-09-23T12:00:00+02:00
author: Ahmed Samir Oraby
categories:
  - llm
  - nlp
tags: [Arabic, LLM, Fine-Tuning, LoRA, NLP, PyTorch, vLLM]
cover: "/assets/header_image.jpg"
---

Training and fine-tuning Large Language Models for Arabic presents unique challenges that off-the-shelf English-centric recipes often fail to address. From the linguistic intricacies of Arabic script and morphology to tokenization fertility and domain alignment, succeeding on Arabic benchmarks and in production workflows requires tailored engineering decisions.

In this article, I share key technical takeaways from engineering end-to-end Arabic LLM pipelines, scaling bilingual translation models, and optimizing production deployments.

---

## 1. The Tokenization Bottleneck & Vocabulary Fertility

One of the most consequential yet frequently underestimated decisions in non-English LLM engineering is tokenization.

Standard multilingual tokenizers (such as those in early LLaMA models or general open models) often have limited representation of Arabic script. This manifests in two ways:
1. **High Token Fertility**: An Arabic word may be split into 4–6 subwords or raw UTF-8 byte tokens, compared to 1–2 tokens for its English translation.
2. **Context Compression Loss**: Because context windows are measured in tokens, a high fertility rate drastically shrinks the effective context length for Arabic documents and increases inference latency.

### Practical Fixes:
- **Vocabulary Extension**: When adapting an open base model to Arabic, evaluate whether extending the tokenizer vocabulary with top Arabic unigrams/morphemes is necessary. If extending vocabulary, ensure embedding layers and output heads are carefully initialized (e.g., mean-initializing new tokens based on semantically closest existing tokens).
- **Diacritization (Tashkeel) Policy**: Raw Arabic training corpora often contain inconsistent vowel markings (Tashkeel). Unless explicitly training a diacritizer or TTS front-end, normalize or selectively strip non-essential diacritics to avoid vocabulary fragmentation during tokenization.

```python
import unicodedata
import re

def normalize_arabic(text: str) -> str:
    # Strip Tashkeel (diacritics) for standardized LLM pre-processing
    tashkeel_pattern = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(tashkeel_pattern, '', text)
    # Normalize Alifs and Yaa/Alef Maksura
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'ء', text)
    text = re.sub(r'ئ', 'ء', text)
    text = re.sub(r'ة', 'ه', text)
    return unicodedata.normalize('NFKC', text)
```

---

## 2. Supervised Fine-Tuning (SFT): Quality vs. Quantity

When fine-tuning for Arabic downstream capabilities (such as bilingual translation, summarization, and task orchestration), dataset cleanliness consistently outperforms raw token volume.

### Dataset Curation Principles:
- **Dialectal vs. Modern Standard Arabic (MSA)**: Identify whether your target task requires formal Modern Standard Arabic (news, legal, formal documentation) or regional dialects (Egyptian, Saudi/Najdi, Levantine). Mixing colloquial conversational data with formal regulatory text without task-specific prefixes or system instructions degrades generation coherence.
- **Deduplication and Heuristic Filtering**: MinHash deduplication combined with language identification filtering (e.g., fastText with confidence thresholding) eliminates corrupted web crawls and mixed-encoding artifacts.
- **Bilingual Parallel Alignment**: For translation and cross-lingual QA, ensuring symmetry in the SFT prompt template prevents the model from developing stylistic degeneration or "language mixing" hallucinations.

---

## 3. PEFT Strategy: Optimizing LoRA & QLoRA for Arabic

Full-parameter fine-tuning is resource-intensive and often unnecessary for domain specialization or instruction tuning. Parameter-Efficient Fine-Tuning (PEFT) with LoRA or 4-bit QLoRA provides rapid iteration speed, provided the rank and target modules are selected properly.

### Target Module Selection:
Many tutorials target only the query and value projections (`q_proj`, `v_proj`). For non-English script adaptation, this is insufficient. You should target all linear layers in the transformer block:

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

Targeting the MLP layers (`gate_proj`, `up_proj`, `down_proj`) is essential for cross-lingual knowledge grounding and grammatical restructuring in Arabic sentences.

---

## 4. Evaluation Beyond Perplexity

Evaluating Arabic models requires a multi-dimensional benchmarking setup:
- **Leaderboard Benchmarks**: Evaluating on standardized Arabic benchmarks (such as Arabic MMLU, ACVA, and Arabic cultural reasoning tasks).
- **Fluency and Hallucination**: Using automated LLM-as-a-judge frameworks alongside human evaluation with native Arabic linguists to grade grammatical agreement (gender, dual/plural morphology, case endings).
- **Bilingual BLEU / COMET**: For translation models, utilizing reference-free and reference-based metrics like COMET alongside BLEU prevents penalizing valid paraphrastic Arabic outputs.

---

## 5. Production Inference with vLLM

Deploying fine-tuned models to production requires predictable latency and maximum throughput:
- **PagedAttention & Continuous Batching**: Using vLLM or TensorRT-LLM ensures memory is allocated dynamically per request, drastically reducing VRAM waste on variable-length Arabic prompts.
- **Quantization**: AWQ and FP8 quantization retain benchmark performance within 0.5% margin of FP16 models while halving the GPU footprint.
- **Prefix Caching**: For agent workflows where common system prompts and Arabic instructions are passed across multi-turn interactions, enabling automatic prefix caching saves significant prefill computation.

---

## Summary

Building competitive Arabic language models requires respecting the unique structural properties of the language across every stage of the pipeline — from tokenizer inspection to linear adapter target selection and domain-aware evaluation.

If you are working on Arabic NLP, speech synthesis, or agentic architectures, feel free to reach out via [GitHub](https://github.com/oraby8) or [LinkedIn](https://www.linkedin.com/in/ahmed-oraby-7b076881/).
