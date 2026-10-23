---
layout: post
title: "DialectNorm: One Small Model to Normalize the Spelling of 12 Arabic Dialects"
date: 2026-10-21T12:00:00+00:00
author: Ahmed Samir Oraby
categories:
  - nlp
tags: [Arabic, NLP, Dialects, Spelling Correction, Open Models, CODA]
excerpt: "Arabic dialects have no official orthography, causing words to be typed in dozens of inconsistent ways. DialectNorm is a 143M-parameter model that normalizes raw dialectal Arabic into CODA spelling across 12 dialects, matching the best published 3-model ensemble on MADAR CODA with a single small model."
---

Arabic is written in two very different ways. Modern Standard Arabic has a fixed spelling taught in school. The dialects people actually speak (Egyptian, Levantine, Gulf, Maghrebi and many more) have no official orthography at all. When people type dialect on their phones, the same word comes out spelled five different ways: `هيك` or `هيج`, `اكتر` or `اكثر`, `الشجره` or `الشجرة`, with or without a hamza, letters stretched for emphasis (`ميرووووه`), and words split or joined at random.

That inconsistency hurts every system built on top of dialectal text: search, machine translation, speech synthesis, sentiment analysis and LLM training data. Linguists answered this with **CODA**, the *Conventional Orthography for Dialectal Arabic*, a set of spelling conventions that give every dialect a consistent written form while keeping its own words and grammar.

Today I'm releasing **DialectNorm**, a 143M-parameter model that rewrites raw dialectal Arabic into CODA spelling for **12 dialects**. On the standard public benchmark for this task, it matches the best published system (a 3-model ensemble) using a single small model.

👉 **Model:** [huggingface.co/ahmedsamirtarjama/dialectnorm](https://huggingface.co/ahmedsamirtarjama/dialectnorm) (free for research; access requests are approved automatically)

---

## What it does

You give DialectNorm a sentence and tell it which dialect it is written in. It returns the same sentence with normalized spelling. It does **not** translate into Modern Standard Arabic, and it does not change the dialect's words or grammar. It only fixes how they are written.

| Dialect | Input | DialectNorm output | Main fixes |
|---|---|---|---|
| Iraqi | اذا هو شايل هلكد حمل المرور الي بلشارع شنو شغله حته ما وكفه | اذا هو شايل هالقد حمل المرور الي بالشارع شنو شغله حتى ما وقفه | هلكد → هالقد · بلشارع → بالشارع · حته → حتى · وكفه → وقفه |
| Egyptian | تمام . إمشي علطول و أحود يمين عند تالت ناصية . حتلاقيه عاليمين . | تمام. امشي على طول واحود يمين عند ثالث ناصية. حتلاقيه عاليمين. | علطول → على طول · و أحود → واحود · تالت → ثالث |
| Syrian | ياما شربان معو متة ؟ كنت صاحبو لهداك الشب ) وبتنقال لما الشب التاني بتعلا مراكزو ( | ياما شربان معه متة ؟ كنت صاحبه لهداك الشب ) وبتنقال لما الشب التاني بتعلا مراكزه ( | معو → معه · صاحبو → صاحبه · مراكزو → مراكزه |
| Palestinian | على طول اسألها : - من بدو يجي عنا ? | على طول اسألها : - من بده يجي عندنا ? | بدو → بده · عنا → عندنا |

Every output above is identical to the human-annotated reference. The "Main fixes" column lists the harder edits (words split or joined, letters added, removed or changed); hamza, ta marbuta and punctuation fixes are not listed. The Egyptian row comes from the MADAR CODA test set, the Syrian and Palestinian rows from the held-out test splits of their corpora, and the Iraqi row from the annotated Lisan corpus.

**Supported dialects:** Egyptian, Lebanese, Syrian, Palestinian, Qatari, Emirati, Iraqi, Yemeni, Sudanese, Libyan, Tunisian and Moroccan.

---

## Results

### The benchmark: MADAR CODA

The standard public benchmark for this task is **MADAR CODA**: 1,500 test sentences in five city dialects (Beirut, Cairo, Doha, Rabat and Tunis), scored with the M2 scorer's **F0.5**. This metric counts word-level edits and weights precision twice as much as recall, because a wrong "correction" is worse than a missed one.

| System | Type | F0.5 |
|---|---|---|
| GPT-4o, 5-shot prompting | LLM | 55.9 |
| AraT5 with dialect control (Alhafni et al., 2024) | single model | 86.3 |
| Best published single model (Alhafni & Habash, 2025) | single model | 86.5 |
| Best published ensemble, 3 models (Alhafni & Habash, 2025) | ensemble | 88.9 |
| **DialectNorm** | **single model, 143M** | **88.9** |

The comparison is like-for-like. The published authors released their system outputs, and when I re-scored them with my evaluation pipeline I got exactly the numbers they report, down to the second decimal. DialectNorm matches the best published ensemble (88.89 vs 88.86) and beats the best published single model by 2.4 points. That model uses the same base encoder, so the gain comes from the method and the data, not from a bigger network.

A note on large language models: few-shot GPT-4o scores 55.9. General-purpose LLMs "know" Arabic dialects, but they don't reliably follow a precise orthographic convention, and they tend to rewrite more than they should.

### All 12 dialects

Every number below comes from a held-out, human-annotated test set that was never used for training or tuning. All settings were chosen on validation data, and each test set was evaluated exactly once.

| Dialect | F0.5 | Precision | Recall |
|---|---|---|---|
| Qatari | **90.8** | 94.9 | 77.5 |
| Lebanese | **90.2** | 92.9 | 80.6 |
| Tunisian | **88.3** | 91.7 | 76.9 |
| Egyptian | **88.2** | 89.6 | 83.0 |
| Moroccan | **87.9** | 92.1 | 74.4 |
| Syrian | **87.0** | 91.5 | 72.9 |
| Emirati | **84.1** | 86.1 | 77.0 |
| Yemeni | **82.2** | 89.9 | 61.3 |
| Palestinian | **79.7** | 86.2 | 61.1 |
| Iraqi | **74.2** | 83.0 | 52.2 |
| Sudanese | **71.3** | 81.4 | 47.6 |
| Libyan | **69.7** | 81.2 | 44.6 |

Precision is above 80% for every dialect, so when DialectNorm changes something it is almost always right. The weaker dialects lose on recall instead: the model leaves some errors uncorrected. These are the dialects with the least training data (about 2,000 sentences each), and that points directly at what to improve next.

---

## How it works

### Why a model and not rules?

Some CODA edits are mechanical, and a rule would do them perfectly: dropping the hamza on a word-initial alef (`أنا → انا`), removing stretched letters (`ميرووووه → ميروه`) or attaching punctuation. Most edits are not like that, because the right spelling depends on what the word *is*, which the letters alone don't show:

- **A final `و` is sometimes a pronoun and sometimes a plural.** `معو → معه` and `صاحبو → صاحبه` ("with him", "his friend"), but `بيلعبو → بيلعبوا` and `اسألو → اسألوا` (plural verbs). A rule like "final و → ه" breaks half of them.
- **A final `ه` is sometimes a ta marbuta and sometimes "his".** `الدوله → الدولة` changes, but `شغله` ("his job") must stay.
- **Letter substitutions apply only to some words.** `تالت → ثالث` and `تاني → ثاني`, because these words historically had a ث, but `تاكسي` and `تعبان` stay. In Gulf Arabic `ويها → وجهها` and `للميلس → للمجلس` (ي that was originally ج), but most ي are just ي.
- **Splitting clitics needs to know every word.** `قالي → قال لي` and `ابعتلي → ابعث لي` split, but `عالي` ("high") and `حالي` must not.
- **Every dialect has its own conventions.** A rule system would need 12 rule sets and a full dictionary for each dialect. Good dictionaries exist for a few dialects, barely for Libyan, Sudanese or Yemeni, and social-media text keeps inventing words that no dictionary has.

A learned model avoids all of this. The annotators' decisions are already in 55,000 corrected sentences, and the model learns them from the words around each edit, instead of a linguist writing every case as a rule. Adding a dialect means annotating a few thousand sentences, not writing a new rule set.

There is a trade-off: a model can occasionally make an edit no rule would make, such as adding a wrong letter. The tagger design and the 90% confidence threshold keep that rare, and precision is above 80% for every dialect.

### A tagger, not a generator

Most recent work on Arabic error correction uses sequence-to-sequence models or LLMs that *generate* the corrected sentence. DialectNorm instead *tags* the input: for every character it predicts one edit:

- **keep** the character;
- **delete** it;
- **replace** it with another string;
- optionally **insert** something after it (a space, a letter).

The correction is produced by applying those edits to the input. This has three practical advantages:

1. **It can't hallucinate.** A generative model can drop a clause, swap a word for a synonym or "fix" the dialect into standard Arabic. A tagger can only make local character edits.
2. **It's fast and small.** One forward pass, no autoregressive decoding, 143M parameters: small enough to run without a GPU.
3. **It's controllable.** Each edit has a confidence score. An edit is applied only when the model is at least 90% sure (50% for punctuation). One number trades precision against recall.

### Architecture

```
dialect prefix + raw text
        │
        ▼
Arabic BERT-base encoder (12 layers, 768-d)  ──► subword states copied to each character
        │
        ⊕  64-d character embedding
        ▼
2-layer BiLSTM (384 per direction) + residual projection + LayerNorm
        │
        ▼
softmax over 250 character-edit labels  ──►  confidence threshold  ──►  apply edits
```

The encoder gives each character the context of the whole sentence. The character embedding and the BiLSTM let the model reason at the level of individual letters, which is where spelling lives. Mixing word-level and character-level information is what lets one model fix both a single wrong letter (`ه → ة`) and a missing space between two words.

### Telling the model which dialect it's reading

The same raw spelling should be normalized differently depending on the dialect. DialectNorm handles this with a simple trick: the dialect name, in Arabic, is prepended to the input as an extra word (for example `عراقية` for Iraqi). The model learns to condition its edits on it, and the prefix is removed from the output.

I first tried city names (Beirut, Cairo…), then switched to country-level dialect names. On the original five dialects, the switch alone changed the score by less than 0.2 points. Its real value is that it makes adding a new dialect clean: one new label, no change to the architecture.

---

## The training data

DialectNorm was trained on about **121,000 sentence pairs** (raw text → CODA spelling) across the 12 dialects:

- **About 55,000 human-annotated pairs.** Native-speaker linguists corrected each sentence by hand to CODA.
- **About 66,000 automatically labelled pairs.** These widen vocabulary coverage for the dialects with the most data, and only labels that passed an agreement filter were kept.

The text is deliberately varied:
- short everyday and travel conversation;
- social-media posts and comments;
- web-forum fiction and novels;
- informal discussions of local life, news and religion.

Every sentence is undiacritized and carries a country-level dialect label.

### Lesson 1: conventions have to be harmonised

Merging annotated data from different sources surfaced an unexpected problem. Annotation teams don't all apply CODA the same way. The clearest case was hamza on a word-initial alef: one part of the data consistently *removed* it (`أنا → انا`), while another consistently *added* it (`انا → أنا`). Train on both as-is and the model learns two opposite rules for the same input.

Before training, I profiled the most frequent edit types in every part of the data, found these conflicts and converted everything to one convention. It's unglamorous work, but skipping it would have made the model contradict itself.

### Lesson 2: data beats model size

For a while I tried to push the original five dialects higher with bigger models and more automatically labelled data. It stopped paying off: adding more machine-labelled sentences produced gains within noise, or small losses. Fine-tuned generative models of several billion parameters gained only about one point over the small tagger while being 15–80× bigger.

What *did* move the needle was **real annotated data for new dialects**. Before training on them, the model barely transferred to unseen dialects, even with the closest available dialect label:

| Dialect | Before (no data for it) | After |
|---|---|---|
| Syrian | 52.2 | **87.0** |
| Emirati | 37.6 | **84.1** |
| Palestinian | 44.2 | **79.7** |

The other new dialects (Iraqi, Yemeni, Sudanese, Libyan) rose from about 45–57 to 70–82. Adding them also *improved* the original benchmark slightly (88.78 → 88.89), and the Gulf dialect that had been the weakest (Qatari) gained about 2 points on validation data, helped by its Emirati neighbour.

Spelling conventions are learned per dialect. A model doesn't infer how Iraqis should write from how Lebanese write, so the way to cover a new dialect is a few thousand annotated sentences of it.

---

## Try it

```bash
pip install torch transformers safetensors huggingface_hub
huggingface-cli login   # after requesting access on the model page
```

```python
from huggingface_hub import snapshot_download
import sys

path = snapshot_download("ahmedsamirtarjama/dialectnorm")
sys.path.insert(0, path)
from dialectnorm import DialectNorm

normalizer = DialectNorm.from_pretrained(path)   # GPU if available, otherwise CPU

normalizer.normalize("الشجره الخبيثه اصلها خايس وفرعها في المجاري", dialect="Iraqi")
# 'الشجرة الخبيثة اصلها خايس وفرعها في المجاري'

normalizer.normalize(["اي اكتر شي بتستخدموا لتروحوا على المدرسة"], dialect="Palestinian")
# ['اي اكثر شي بتستخدموا لتروحوا على المدرسة']
```

`normalize()` accepts a string or a list. Its `threshold` argument (default 0.9) controls how bold the model is: lower values correct more, at some cost in precision.

---

## License

DialectNorm is free for **non-commercial research**. Request access on the [model page](https://huggingface.co/ahmedsamirtarjama/dialectnorm); requests are approved automatically. For commercial use, please get in touch.

If you use DialectNorm in your research, please cite it as **DialectNorm** with a link to the model page.
