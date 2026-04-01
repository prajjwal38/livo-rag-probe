# RAG Evaluation Report

**Dataset:** `Golden_QA_dataset.json`  
**Total questions:** 15  
**Average latency:** 40.38s per question

---

## Summary

| Metric | Score |
| --- | --- |
| Irrelevance Handling (refuse rate) | 0/3 (0%) |
| Pinpoint Retrieval Accuracy (top-1 hit) | 5/12 (42%) |
| Avg. Correctness Score (0-2) | 0.20 |
| Avg. Faithfulness Score (0-2) | 2.00 |

---

## Irrelevance Handling

| ID | Question | Refused? | Generated Answer (truncated) |
| --- | --- | --- | --- |
| `q6_irrelevant_recipe` | What is the best recipe for making biryani at home? | ❌ FAIL | PIPELINE ERROR: [...] |
| `q7_irrelevant_cricket` | Who won the 2023 Cricket World Cup and what was the final sc | ❌ FAIL | I'm sorry, but I don't have enough relevant information in my knowledge [...] |
| `q8_irrelevant_stock_market` | What are the best stocks to invest in for 2025? | ❌ FAIL | I'm sorry, but I don't have enough relevant information in my knowledge [...] |

## Pinpoint Retrieval & Generation Quality

| ID | Question | Expected Video | Top-1 Hit | Pinpoint? | C | F | Latency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `q1_representation_learning_cross_video` | How do both the Hindi-speaking instructor and the 3Blue | `—` | `—` | ❌ | — | — | 183.0s |
| `q2_definition_vs_architecture_dl` | How does the speaker distinguish Deep Learning from Mac | `—` | `—` | ❌ | — | — | 0.5s |
| `q3_manual_vs_automatic_feature_extraction` | How does the speaker contrast feature extraction in Mac | `—` | `—` | ❌ | — | — | 0.6s |
| `q4_layerwise_feature_hierarchy` | What step-by-step transformation of features occurs acr | `—` | `—` | ❌ | — | — | 0.5s |
| `q5_data_dependency_performance_curve` | Why does Deep Learning eventually outperform Machine Le | `—` | `—` | ❌ | — | — | 0.5s |
| `q6_hardware_requirement_reasoning` | What is the underlying computational reason that Deep L | `—` | `—` | ❌ | — | — | 0.5s |
| `q5_black_box_interpretability_cross_video` | Both videos discuss how deep learning models operate as | `—` | `—` | ❌ | — | — | 0.5s |
| `q1_attention_contextual_disambiguation` | According to the video's explanation of a transformer's | `—` | `kn6S7X98p9Q` | ✅ | 0 | 2 | 93.9s |
| `q2_attention_vs_mlp_contrast` | In the context of data flowing through a transformer, h | `—` | `kn6S7X98p9Q` | ✅ | 0 | 2 | 99.7s |
| `q3_autoregressive_generation_loop` | How does a text-generating transformer use single-word  | `—` | `kn6S7X98p9Q` | ✅ | 0 | 2 | 76.3s |
| `q4_system_prompt_missing_evidence` | How do you turn a basic word-predicting transformer int | `—` | `kn6S7X98p9Q` | ✅ | 0 | 2 | 62.0s |
| `q5_gpt3_parameter_structure` | How are the 175 billion parameters in GPT-3 specificall | `—` | `kn6S7X98p9Q` | ✅ | 1 | 2 | 63.7s |

---

## Per-Question Details

### [RELEVANT] `q1_representation_learning_cross_video` — FAIL
**Question:** How do both the Hindi-speaking instructor and the 3Blue1Brown video use image-based examples to explain how hidden layers perform 'representation learning' or automatic feature extraction?

**Expected video:** `any`  **Timestamp hint:** `video_3 @ ~15:24 AND video_1 @ ~06:44 & ~14:15`
**Top-1 retrieved:** video=`None` | timestamp=`None`
**Pinpoint hit:** None

**Expected answer:** Both explain that hidden layers progressively extract more complex features without manual engineering. They both note that the earliest layers identify primitive, low-level features like [...]
**Generated answer:** PIPELINE ERROR: (MaxRetryError('HTTPSConnectionPool(host=\'huggingface.co\', port=443): Max retries exceeded with url: /api/models/Qwen/Qwen3-Reranker-0.6B (Caused by [...]

---

### [RELEVANT] `q2_definition_vs_architecture_dl` — FAIL
**Question:** How does the speaker distinguish Deep Learning from Machine Learning in terms of underlying approach and inspiration?

**Expected video:** `any`  **Timestamp hint:** `video_1 @ ~02:18 & ~04:04`
**Top-1 retrieved:** video=`None` | timestamp=`None`
**Pinpoint hit:** None

**Expected answer:** The speaker explains that Machine Learning relies on statistical techniques to model relationships between input and output, whereas Deep Learning is based on neural networks inspired by the [...]
**Generated answer:** PIPELINE ERROR: (MaxRetryError('HTTPSConnectionPool(host=\'huggingface.co\', port=443): Max retries exceeded with url: /api/models/Qwen/Qwen3-Reranker-0.6B (Caused by [...]

---

### [RELEVANT] `q3_manual_vs_automatic_feature_extraction` — FAIL
**Question:** How does the speaker contrast feature extraction in Machine Learning versus Deep Learning using the dog vs cat example?

**Expected video:** `any`  **Timestamp hint:** `video_1 @ ~15:23`
**Top-1 retrieved:** video=`None` | timestamp=`None`
**Pinpoint hit:** None

**Expected answer:** The speaker explains that in Machine Learning, features like size and color must be manually defined, whereas in Deep Learning, the model automatically extracts features directly from raw [...]
**Generated answer:** PIPELINE ERROR: (MaxRetryError('HTTPSConnectionPool(host=\'huggingface.co\', port=443): Max retries exceeded with url: /api/models/Qwen/Qwen3-Reranker-0.6B (Caused by [...]

---

### [RELEVANT] `q4_layerwise_feature_hierarchy` — FAIL
**Question:** What step-by-step transformation of features occurs across layers in a neural network according to the speaker?

**Expected video:** `any`  **Timestamp hint:** `video_1 @ ~18:08`
**Top-1 retrieved:** video=`None` | timestamp=`None`
**Pinpoint hit:** None

**Expected answer:** The speaker explains that lower layers detect primitive features like edges, intermediate layers combine these into shapes or patterns, and higher layers identify complex concepts such as faces, [...]
**Generated answer:** PIPELINE ERROR: (MaxRetryError('HTTPSConnectionPool(host=\'huggingface.co\', port=443): Max retries exceeded with url: /api/models/Qwen/Qwen3-Reranker-0.6B (Caused by [...]

---

### [RELEVANT] `q5_data_dependency_performance_curve` — FAIL
**Question:** Why does Deep Learning eventually outperform Machine Learning as the dataset size increases?

**Expected video:** `any`  **Timestamp hint:** `video_1 @ ~21:25`
**Top-1 retrieved:** video=`None` | timestamp=`None`
**Pinpoint hit:** None

**Expected answer:** The speaker explains that Machine Learning performs better when data is limited but eventually stagnates as data increases. Deep Learning, being data-hungry, continues to improve almost linearly [...]
**Generated answer:** PIPELINE ERROR: (MaxRetryError('HTTPSConnectionPool(host=\'huggingface.co\', port=443): Max retries exceeded with url: /api/models/Qwen/Qwen3-Reranker-0.6B (Caused by [...]

---

### [RELEVANT] `q6_hardware_requirement_reasoning` — FAIL
**Question:** What is the underlying computational reason that Deep Learning models require GPUs instead of CPUs?

**Expected video:** `any`  **Timestamp hint:** `video_1 @ ~23:11`
**Top-1 retrieved:** video=`None` | timestamp=`None`
**Pinpoint hit:** None

**Expected answer:** The speaker explains that Deep Learning involves very complex matrix multiplications and large-scale computations, which require high parallel processing power. GPUs handle this efficiently, [...]
**Generated answer:** PIPELINE ERROR: (MaxRetryError('HTTPSConnectionPool(host=\'huggingface.co\', port=443): Max retries exceeded with url: /api/models/Qwen/Qwen3-Reranker-0.6B (Caused by [...]

---

### [RELEVANT] `q5_black_box_interpretability_cross_video` — FAIL
**Question:** Both videos discuss how deep learning models operate as a 'black box' compared to simpler models. How do the examples of the resume shortlisting system (from the Hindi video) and the digit-recognizing network (from 3Blue1Brown) illustrate this lack of interpretability?

**Expected video:** `any`  **Timestamp hint:** `video_3 @ ~34:00 AND video_1 @ ~16:00`
**Top-1 retrieved:** video=`None` | timestamp=`None`
**Pinpoint hit:** None

**Expected answer:** The Hindi video explains that if a Deep Learning model rejects a resume, it acts as a black box because you cannot explicitly tell the user why they were rejected, unlike simpler ML where you [...]
**Generated answer:** PIPELINE ERROR: (MaxRetryError('HTTPSConnectionPool(host=\'huggingface.co\', port=443): Max retries exceeded with url: /api/models/Qwen/Qwen3-Reranker-0.6B (Caused by [...]

---

### [IRRELEVANT] `q6_irrelevant_recipe` — FAIL
**Question:** What is the best recipe for making biryani at home?

**Refused correctly:** None
**Generated answer:** PIPELINE ERROR: (MaxRetryError('HTTPSConnectionPool(host=\'huggingface.co\', port=443): Max retries exceeded with url: /api/models/Qwen/Qwen3-Reranker-0.6B (Caused by [...]

---

### [IRRELEVANT] `q7_irrelevant_cricket` — FAIL
**Question:** Who won the 2023 Cricket World Cup and what was the final score?

**Refused correctly:** False
**Generated answer:** I'm sorry, but I don't have enough relevant information in my knowledge base to answer your question. Please ask something related to the indexed video content.

---

### [IRRELEVANT] `q8_irrelevant_stock_market` — FAIL
**Question:** What are the best stocks to invest in for 2025?

**Refused correctly:** False
**Generated answer:** I'm sorry, but I don't have enough relevant information in my knowledge base to answer your question. Please ask something related to the indexed video content.

---

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
