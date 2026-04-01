# RAG Evaluation Report

**Dataset:** `Golden_QA_dataset.json`  
**Total questions:** 10  
**Average latency:** 89.79s per question

---

## Summary

| Metric | Score |
| --- | --- |
| Irrelevance Handling (refuse rate) | 0/0 (N/A) |
| Pinpoint Retrieval Accuracy (top-1 hit) | 10/10 (100%) |
| Avg. Correctness Score (0-2) | 0.20 |
| Avg. Faithfulness Score (0-2) | 2.00 |

---

## Pinpoint Retrieval & Generation Quality

| ID | Question | Expected Video | Top-1 Hit | Pinpoint? | C | F | Latency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `q1_attention_contextual_disambiguation` | According to the video's explanation of a transformer's | `—` | `kn6S7X98p9Q` | ✅ | 0 | 2 | 114.3s |
| `q2_attention_vs_mlp_contrast` | In the context of data flowing through a transformer, h | `—` | `kn6S7X98p9Q` | ✅ | 0 | 2 | 106.1s |
| `q3_autoregressive_generation_loop` | How does a text-generating transformer use single-word  | `—` | `kn6S7X98p9Q` | ✅ | 0 | 2 | 88.0s |
| `q4_system_prompt_missing_evidence` | How do you turn a basic word-predicting transformer int | `—` | `kn6S7X98p9Q` | ✅ | 0 | 2 | 94.2s |
| `q5_gpt3_parameter_structure` | How are the 175 billion parameters in GPT-3 specificall | `—` | `kn6S7X98p9Q` | ✅ | 1 | 2 | 68.0s |
| `q1_attention_block_model_disambiguation` | According to the video's explanation of a transformer's | `—` | `kn6S7X98p9Q` | ✅ | 0 | 2 | 80.9s |
| `q2_attention_vs_mlp` | In the context of data flowing through a transformer, h | `—` | `kn6S7X98p9Q` | ✅ | 0 | 2 | 105.1s |
| `q3_autoregressive_text_generation` | How does a text-generating transformer use single-word  | `—` | `kn6S7X98p9Q` | ✅ | 0 | 2 | 78.8s |
| `q4_system_prompt_unverifiable` | How do you turn a basic word-predicting transformer int | `—` | `kn6S7X98p9Q` | ✅ | 0 | 2 | 86.8s |
| `q5_gpt3_parameter_structure` | How are the 175 billion parameters in GPT-3 specificall | `—` | `kn6S7X98p9Q` | ✅ | 1 | 2 | 75.6s |

---

## Per-Question Details

### [RELEVANT] `q1_attention_contextual_disambiguation` — PASS
**Question:** According to the video's explanation of a transformer's architecture, how does the "attention block" specifically differentiate the word "model" in the phrases "machine learning model" and "fashion model"?

**Expected video:** `any`  **Timestamp hint:** `video_2 @ ~03:03`
**Top-1 retrieved:** video=`kn6S7X98p9Q` | timestamp=`03:03` | rerank=0.9909
**Pinpoint hit:** True

**Expected answer:** Before the attention block, each token's vector is a fixed lookup from the embedding matrix — it only encodes the isolated meaning of that one word. The attention block then allows the sequence [...]
**Generated answer:** The 'attention block' in a transformer allows vectors to focus on relevant parts of the input sequence during processing. In the case of the phrase 'machine learning model', the attention [...]
**Judge:** Correctness: Low bigram overlap (2%). Faithfulness: High grounding (3/3 sentences supported).

---

### [RELEVANT] `q2_attention_vs_mlp_contrast` — PASS
**Question:** In the context of data flowing through a transformer, how does the operation of a Multilayer Perceptron (MLP) block contrast with that of an Attention block?

**Expected video:** `any`  **Timestamp hint:** `video_2 @ ~03:03`
**Top-1 retrieved:** video=`kn6S7X98p9Q` | timestamp=`03:03` | rerank=0.9780
**Pinpoint hit:** True

**Expected answer:** In an attention block, vectors in the sequence 'talk to each other' — passing contextual information between tokens to update their values. In a Multilayer Perceptron block, vectors do not [...]
**Generated answer:** In the context of data flowing through a transformer, the operation of a Multilayer Perceptron (MLP) block contrasts with that of an Attention block primarily due to their distinct roles and [...]
**Judge:** Correctness: Low bigram overlap (13%). Faithfulness: High grounding (5/5 sentences supported).

---

### [RELEVANT] `q3_autoregressive_generation_loop` — PASS
**Question:** How does a text-generating transformer use single-word prediction to output a longer cohesive passage of text, according to the video's step-by-step breakdown?

**Expected video:** `any`  **Timestamp hint:** `video_2 @ ~01:44`
**Top-1 retrieved:** video=`kn6S7X98p9Q` | timestamp=`01:44` | rerank=0.9914
**Pinpoint hit:** True

**Expected answer:** The model takes an initial snippet of text, produces a probability distribution for the next token, takes a random sample from that distribution, appends the sampled word to the original text, [...]
**Generated answer:** A text-generating transformer uses single-word prediction to output a longer cohesive passage by repeatedly predicting the next word in a sequence while incorporating previously generated parts [...]
**Judge:** Correctness: Low bigram overlap (7%). Faithfulness: High grounding (3/3 sentences supported).

---

### [RELEVANT] `q4_system_prompt_missing_evidence` — PASS
**Question:** How do you turn a basic word-predicting transformer into a chatbot, using what the video describes as a "system prompt"?

**Expected video:** `any`  **Timestamp hint:** `video_2 @ ~06:06 (missing)`
**Top-1 retrieved:** video=`kn6S7X98p9Q` | timestamp=`00:00` | rerank=0.6246
**Pinpoint hit:** True

**Expected answer:** UNVERIFIABLE — No supporting transcript chunk was provided for this QA pair. The claimed timestamp does not correspond to any supplied chunk discussing system prompts or chatbot framing. This [...]
**Generated answer:** To turn a basic word-predicting transformer into a chatbot, use a system prompt that specifies the desired behavior. This involves setting up the model to generate responses rather than just [...]
**Judge:** Correctness: Low bigram overlap (0%). Faithfulness: High grounding (3/4 sentences supported).

---

### [RELEVANT] `q5_gpt3_parameter_structure` — PASS
**Question:** How are the 175 billion parameters in GPT-3 specifically organized to process data, as broken down in the video?

**Expected video:** `any`  **Timestamp hint:** `video_2 @ ~08:53`
**Top-1 retrieved:** video=`kn6S7X98p9Q` | timestamp=`08:53` | rerank=0.9987
**Pinpoint hit:** True

**Expected answer:** GPT-3's 175 billion weights are organized into just under 28,000 distinct matrices. These matrices fall into eight functional categories: Embedding, Key, Query, Value, Output, Up-projection, [...]
**Generated answer:** GPT-3's 175 billion parameters are organized into approximately 28,000 distinct matrices categorized into eight types: Embedding, Key, Query, Value, Output, Up-projection, Down-projection, and [...]
**Judge:** Correctness: Partial bigram overlap (33%). Faithfulness: High grounding (2/2 sentences supported).

---

### [RELEVANT] `q1_attention_block_model_disambiguation` — PASS
**Question:** According to the video's explanation of a transformer's architecture, how does the "attention block" specifically differentiate the word "model" in the phrases "machine learning model" and "fashion model"?

**Expected video:** `any`  **Timestamp hint:** `But what is a GPT? Visual intro to transformers | 03:03`
**Top-1 retrieved:** video=`kn6S7X98p9Q` | timestamp=`03:03` | rerank=0.9909
**Pinpoint hit:** True

**Expected answer:** Before the attention block, each token's vector is a fixed lookup from the embedding matrix — it only encodes the isolated meaning of that one word. The attention block then allows the sequence [...]
**Generated answer:** The 'attention block' in a transformer allows vectors to focus on relevant parts of the input sequence during processing. In the case of the phrase 'machine learning model', the attention [...]
**Judge:** Correctness: Low bigram overlap (2%). Faithfulness: High grounding (3/3 sentences supported).

---

### [RELEVANT] `q2_attention_vs_mlp` — PASS
**Question:** In the context of data flowing through a transformer, how does the operation of a Multilayer Perceptron (MLP) block contrast with that of an Attention block?

**Expected video:** `any`  **Timestamp hint:** `But what is a GPT? Visual intro to transformers | 03:03`
**Top-1 retrieved:** video=`kn6S7X98p9Q` | timestamp=`03:03` | rerank=0.9780
**Pinpoint hit:** True

**Expected answer:** In an attention block, vectors in the sequence 'talk to each other' — passing contextual information between tokens to update their values. In a Multilayer Perceptron block, vectors do not [...]
**Generated answer:** In the context of data flowing through a transformer, the operation of a Multilayer Perceptron (MLP) block contrasts with that of an Attention block primarily due to their distinct roles and [...]
**Judge:** Correctness: Low bigram overlap (13%). Faithfulness: High grounding (5/5 sentences supported).

---

### [RELEVANT] `q3_autoregressive_text_generation` — PASS
**Question:** How does a text-generating transformer use single-word prediction to output a longer cohesive passage of text, according to the video's step-by-step breakdown?

**Expected video:** `any`  **Timestamp hint:** `But what is a GPT? Visual intro to transformers | 01:44`
**Top-1 retrieved:** video=`kn6S7X98p9Q` | timestamp=`01:44` | rerank=0.9914
**Pinpoint hit:** True

**Expected answer:** The model takes an initial snippet of text, produces a probability distribution for the next token, takes a random sample from that distribution, appends the sampled word to the original text, [...]
**Generated answer:** A text-generating transformer uses single-word prediction to output a longer cohesive passage by repeatedly predicting the next word in a sequence while incorporating previously generated parts [...]
**Judge:** Correctness: Low bigram overlap (6%). Faithfulness: High grounding (3/3 sentences supported).

---

### [RELEVANT] `q4_system_prompt_unverifiable` — PASS
**Question:** How do you turn a basic word-predicting transformer into a chatbot, using what the video describes as a "system prompt"?

**Expected video:** `any`  **Timestamp hint:** `But what is a GPT? Visual intro to transformers | 06:06 (missing)`
**Top-1 retrieved:** video=`kn6S7X98p9Q` | timestamp=`00:00` | rerank=0.6246
**Pinpoint hit:** True

**Expected answer:** UNVERIFIABLE — No supporting transcript chunk was provided for this QA pair. The claimed SOURCE timestamp does not correspond to any supplied chunk, and no chunk discusses system prompts or [...]
**Generated answer:** To turn a basic word-predicting transformer into a chatbot, use a system prompt that specifies the desired behavior. This involves setting up the model to generate responses rather than just [...]
**Judge:** Correctness: Low bigram overlap (0%). Faithfulness: High grounding (3/4 sentences supported).

---

### [RELEVANT] `q5_gpt3_parameter_structure` — PASS
**Question:** How are the 175 billion parameters in GPT-3 specifically organized to process data, as broken down in the video?

**Expected video:** `any`  **Timestamp hint:** `But what is a GPT? Visual intro to transformers | 08:53`
**Top-1 retrieved:** video=`kn6S7X98p9Q` | timestamp=`08:53` | rerank=0.9987
**Pinpoint hit:** True

**Expected answer:** GPT-3's 175 billion weights are organized into just under 28,000 distinct matrices. These matrices fall into eight functional categories: Embedding, Key, Query, Value, Output, Up-projection, [...]
**Generated answer:** GPT-3's 175 billion parameters are organized into approximately 28,000 distinct matrices categorized into eight types: Embedding, Key, Query, Value, Output, Up-projection, Down-projection, and [...]
**Judge:** Correctness: Partial bigram overlap (24%). Faithfulness: High grounding (2/2 sentences supported).

---
