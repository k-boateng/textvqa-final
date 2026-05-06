# Visual Understanding with TextVQA

## 1. Introduction

Visual question answering over text-rich images requires a model to both understand the visual scene and read text embedded in the image. This project evaluates that capability using the TextVQA dataset, where questions often require recognizing words, numbers, brands, labels, signs, and other textual elements in natural images. TextVQA is difficult because the model must not only perform OCR, but also ground the question to the correct region of the image, align visual text with the question intent, and output an answer in the short format expected by the dataset.

Recent vision-language models such as BLIP-2, LLaVA, and Qwen-VL have shown strong performance on multimodal understanding tasks, but text-rich visual question answering remains challenging because success depends on reading, grounding, reasoning, and producing dataset-aligned short answers. The goal of this project is to evaluate and improve the performance of an open-source vision-language model on TextVQA. I use Qwen2.5-VL-3B-Instruct as the primary model and compare three levels of adaptation: zero-shot inference, prompt engineering with and without OCR tokens, and LoRA fine-tuning on different training subset sizes. The final system is evaluated on a fixed validation subset of 2,000 examples using accuracy as the primary metric, along with token F1, ROUGE-L, and BLEU as secondary metrics.

The main finding is that both prompt engineering and LoRA fine-tuning substantially improve TextVQA performance. The naive zero-shot baseline achieved 72.20% accuracy. The best prompt-only method achieved 80.90% accuracy, and the best LoRA model, fine-tuned on 10,000 examples, achieved 87.15% accuracy.

## 2. Dataset

This project uses the TextVQA dataset, which is designed to test whether models can answer questions about text appearing in natural images. The dataset contains image-question-answer examples where the correct answer often depends on reading visual text. Examples include identifying words on signs, brands on products, numbers on labels, text on screens, and other real-world textual content.

The dataset has predefined train, validation, and test splits. For controlled experiments, I used the training split for LoRA fine-tuning and a fixed 2,000-example subset of the validation split for all evaluations. Using the same validation subset across all methods ensures that differences in performance reflect changes in prompting or fine-tuning rather than changes in the evaluation data.

During preprocessing, each example was converted into a standard format containing the image, question, list of ground-truth answers, OCR tokens when available, and image ID. To improve inference stability, images with a largest side greater than 1024 pixels were resized while preserving aspect ratio. This was necessary because very large images can produce excessive visual tokens and slow down or destabilize VLM inference.

## 3. Methodology

### 3.1 Model

The primary model used in this project is Qwen2.5-VL-3B-Instruct. This model was selected because it is an open-source vision-language model with a relatively small parameter count compared with larger VLMs, making it feasible to run on limited compute while still providing strong multimodal reasoning and visual text understanding.

The model was evaluated in three settings:

1. Zero-shot inference without additional training.
2. Prompt-engineered inference with different prompt formats and OCR-token usage.
3. LoRA fine-tuning on TextVQA training subsets.

### 3.2 Prompting Strategies

I evaluated three prompt styles:

**Naive prompt.** The model is simply asked to answer the question based on the image and return a short answer.

**OCR-aware prompt.** The model is given the detected OCR tokens and instructed to use them only when relevant.

**Constrained prompt.** The model is explicitly told that the answer is usually a short word, number, brand, name, or phrase visible in the image. It is also instructed not to explain and to return only the final answer.

The constrained prompt was designed to reduce verbose answers, explanations, and formatting mismatches. Since TextVQA uses short answer matching, constraining the output format is important for improving exact-match accuracy.

### 3.3 OCR Use

OCR tokens from the dataset were tested as an explicit input feature. To isolate the effect of OCR tokens, I compared the naive prompt with and without OCR and also compared the constrained prompt with and without OCR.

The results show that simply appending OCR tokens does not help under the naive prompt. However, OCR tokens provide a small additional gain when paired with the constrained prompt. This suggests that OCR tokens are most useful when the prompt clearly tells the model how to use them.

### 3.4 LoRA Fine-tuning

For fine-tuning, I used LoRA adapters rather than full model fine-tuning. LoRA updates only a small number of trainable low-rank parameters while keeping the base model frozen, making the method more computationally efficient.

I trained LoRA adapters using three training subset sizes:

- 1,000 examples
- 5,000 examples
- 10,000 examples

All LoRA runs used the constrained prompt with OCR tokens, since this was the best prompt-only configuration by accuracy. Training subsets were selected deterministically from the shuffled TextVQA training split using a fixed random seed, so that the 1k, 5k, and 10k subsets were nested and comparable.

The LoRA models were trained on an NVIDIA A100 GPU using batch size 2 and gradient accumulation 4, giving an effective batch size of 8. Evaluation was performed on the same fixed 2,000-example validation subset used for the prompt experiments.

**Training details and hyperparameters.** To make the setup reproducible, Table 1 lists the main LoRA and optimization settings used for the fine-tuning runs.

| Component | Setting |
|---|---|
| Base model | Qwen/Qwen2.5-VL-3B-Instruct |
| Prompt during LoRA training | Constrained prompt with OCR tokens |
| Training sizes | 1,000; 5,000; 10,000 examples |
| Epochs | 1 |
| Per-device batch size | 2 |
| Gradient accumulation | 4 |
| Effective batch size | 8 |
| Learning rate | 2e-4 |
| Optimizer | AdamW |
| Precision | bfloat16 on A100 |
| LoRA rank | r = 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Warmup ratio | 0.03 |
| Max generated tokens at evaluation | 32 |
| Decoding | greedy decoding, temperature = 0, do_sample = false |
| Image preprocessing | largest side capped at 1024 px with aspect ratio preserved |

LoRA was applied to both attention projection layers and feed-forward projection layers. This choice adapts the language model portions most directly involved in mapping multimodal representations to answer tokens while avoiding full-parameter fine-tuning. The learning rate and effective batch size were kept fixed across all three subset sizes so that the scaling comparison reflected training data size rather than optimizer changes.

### 3.5 Evaluation Metrics

The primary metric is exact-match accuracy, computed by normalizing the model prediction and checking whether it matches any ground-truth answer. I also report token F1, ROUGE-L, and BLEU as secondary metrics to capture partial overlap and semantic similarity between predictions and ground-truth answers.

Accuracy is emphasized most heavily because TextVQA evaluation is based on whether the model returns the correct short answer. Token F1, ROUGE-L, and BLEU are useful for diagnosing near misses, formatting issues, and partial text overlap.

## 4. Experimental Design

The experiments were designed to isolate the contribution of each intervention. Specifically, they test the hypothesis that performance gains come from three complementary mechanisms: improved output alignment through prompting, explicit text grounding through OCR tokens, and learned task-specific mappings through LoRA fine-tuning.

First, I evaluated the base Qwen2.5-VL-3B-Instruct model with a naive zero-shot prompt and no OCR tokens. This served as the baseline.

Second, I evaluated prompt-engineering variants while keeping the model weights fixed. This allowed me to measure how much improvement came from better prompting alone.

Third, I performed OCR ablations by comparing otherwise similar prompts with and without OCR tokens. This tested whether explicit OCR tokens improved performance beyond the model's own visual text-reading ability.

Finally, I fine-tuned LoRA adapters on 1k, 5k, and 10k training examples. This tested whether parameter-efficient fine-tuning improved performance beyond prompt engineering, and whether performance scaled with more training data.

All methods were evaluated on the same 2,000 validation examples. This fixed evaluation set makes the comparisons controlled and reproducible.

## 5. Results

Table 2 summarizes the main results across zero-shot prompting, prompt engineering, OCR ablation, and LoRA fine-tuning.

**Table 2. Main TextVQA Results**

| Method | Train Size | OCR | Prompt | Accuracy | Token F1 | ROUGE-L | BLEU | N |
|---|---:|---|---|---:|---:|---:|---:|---:|
| Zero-shot naive | 0 | No | naive | 0.7220 | 0.7936 | 0.8786 | 0.8128 | 2000 |
| Zero-shot naive + OCR | 0 | Yes | naive | 0.7220 | 0.7936 | 0.8786 | 0.8128 | 2000 |
| OCR-aware prompt | 0 | Yes | ocr_aware | 0.7920 | 0.8342 | 0.8535 | 0.8304 | 2000 |
| Constrained prompt | 0 | No | constrained | 0.8000 | 0.8451 | 0.8663 | 0.8413 | 2000 |
| Constrained prompt + OCR | 0 | Yes | constrained | 0.8090 | 0.8414 | 0.8586 | 0.8386 | 2000 |
| LoRA | 1000 | Yes | constrained | 0.8385 | 0.8681 | 0.8816 | 0.8684 | 2000 |
| LoRA | 5000 | Yes | constrained | 0.8650 | 0.8883 | 0.9022 | 0.8899 | 2000 |
| LoRA | 10000 | Yes | constrained | 0.8715 | 0.8930 | 0.9009 | 0.8940 | 2000 |

![Accuracy by method](../outputs/plots/final_accuracy_bar.png)

![Token F1 by method](../outputs/plots/final_token_f1_bar.png)

![BLEU by method](../outputs/plots/final_bleu_bar.png)

### 5.1 Prompt Engineering

The naive zero-shot baseline achieved 72.20% accuracy. Adding OCR tokens to the naive prompt did not change the result, producing the same 72.20% accuracy. This suggests that OCR tokens alone are not useful unless the prompt gives the model a clear instruction for how to use them.

The OCR-aware prompt improved accuracy to 79.20%, and the constrained prompt with OCR achieved 80.90%. This is an 8.70 percentage point improvement over the naive baseline. The constrained prompt without OCR also performed strongly at 80.00%, showing that much of the gain came from better output control rather than OCR alone. This pattern suggests that Qwen2.5-VL already has substantial visual text-reading ability, but the naive prompt does not reliably force the model to answer in the annotation style used by TextVQA. The constrained prompt likely helps by reducing explanations, discouraging full-sentence outputs, and biasing the model toward short text spans, numbers, brands, and names.

![OCR ablation](../outputs/plots/ocr_ablation_accuracy.png)

### 5.2 LoRA Fine-tuning

LoRA fine-tuning further improved performance beyond prompt engineering. With only 1,000 training examples, LoRA reached 83.85% accuracy, already outperforming the best prompt-only method by 2.95 percentage points. Increasing the training set to 5,000 examples improved accuracy to 86.50%, and 10,000 examples achieved the best accuracy of 87.15%.

![LoRA accuracy vs training size](../outputs/plots/lora_train_size_accuracy.png)

The largest LoRA gain came from increasing the training set from 1k to 5k examples. Accuracy improved from 83.85% to 86.50%, a gain of 2.65 percentage points. Increasing from 5k to 10k examples gave a smaller gain of 0.65 percentage points, suggesting diminishing returns after 5k examples under this setup. A likely explanation is that the model learns the most common TextVQA answer patterns, output style, and question-to-text alignment behavior within the first few thousand examples. Additional data still helps, but it may mostly reinforce patterns the model has already learned rather than exposing entirely new capabilities. The remaining errors also include inherently hard visual text cases and ambiguous annotations, so additional supervised examples alone may not remove all failures.

Overall, the best model was the LoRA-10k model with OCR and the constrained prompt. It achieved 87.15% accuracy, which is 14.95 percentage points higher than the naive baseline and 6.25 percentage points higher than the best prompt-only method.

## 6. Qualitative Analysis

To better understand the improvements, I compared predictions from three representative systems: the naive zero-shot baseline, the constrained prompt with OCR, and the LoRA-10k model. The comparison produced the following categories:

| Category | Count |
|---|---:|
| All methods correct | 1313 |
| All methods wrong | 175 |
| Prompt improves over naive | 271 |
| LoRA improves over prompt | 190 |
| LoRA regresses from prompt | 65 |

These results show that the aggregate improvements are reflected in individual examples. Prompt engineering fixed 271 examples that the naive baseline missed. LoRA fixed an additional 190 examples that the constrained prompt missed. LoRA also introduced 65 regressions relative to the constrained prompt, but the number of improvements was much larger than the number of regressions.

Qualitatively, the constrained prompt often helped by making the model return a concise answer instead of a full sentence or explanation. LoRA fine-tuning improved cases where the model needed to map the question to the correct piece of visible text or choose a short answer matching the TextVQA annotation style. Remaining failures often involved hard-to-read text, ambiguous questions, or cases where the model produced a plausible but non-matching answer.

Table 3 shows representative inspected examples from the LoRA-10k failure set. These examples were used to connect the aggregate error categories to concrete model behavior. The corresponding images were exported to `outputs/examples/error_analysis_images/`.

**Table 3. Representative qualitative examples from LoRA-10k failure analysis**

| Image | Question | Ground Truth | Prediction | Error Type | Explanation |
|---|---|---|---|---|---|
| ![](../outputs/examples/error_analysis_images/idx_9.jpg) | What brand of cigarettes? | sampoerna / nampoerna | classico | OCR/text-reading | The model selected a different visible token instead of the cigarette brand text. |
| ![](../outputs/examples/error_analysis_images/idx_30.jpg) | How many easton brand bags are shown here? | 2 | "answering does not require reading text in the image" | Hallucination | The model produced a meta/prompt-like response instead of answering the counting question. This is a failure of answer generation and constraint-following rather than a normal OCR error. |
| ![](../outputs/examples/error_analysis_images/idx_53.jpg) | What does the key on the bottom right corner of the keyboard say? | enter | control | OCR/text-reading | The model identified the wrong keyboard key label. |
| ![](../outputs/examples/error_analysis_images/idx_67.jpg) | What is written on the burger box? | big mac | mac | Normalization/partial answer | The prediction is partially correct but misses the full expected phrase. |
| ![](../outputs/examples/error_analysis_images/idx_72.jpg) | What is the name of the business? | paddy reillys | paddy reilly's | Normalization issue | The answer differs mainly by punctuation, but exact matching counts it as wrong. |
| ![](../outputs/examples/error_analysis_images/idx_92.jpg) | What brand are these toys from? | soft farm | farm animals | Reasoning error | The model answered the object category rather than the brand requested by the question. |
| ![](../outputs/examples/error_analysis_images/idx_122.jpg) | What time is on the clock? | 8:29 / 8:30 / 9:30 | 10:10 | Reasoning/visual reading | The model misread the clock hands, showing that non-text visual reasoning remains difficult. |
| ![](../outputs/examples/error_analysis_images/idx_140.jpg) | How much does this cost? | NT$31,888 / $31,888 | NT$40,888 | Ambiguous | Multiple prices are visible, so the referent of “this” is unclear. |

These examples show that the final model's errors are not dominated by one behavior. Some are true OCR failures, some are visual reasoning failures, and others are close answers penalized by exact matching. This supports the use of both exact-match accuracy and secondary overlap metrics in the quantitative evaluation.

## 7. Error Analysis

I manually labeled 100 failures from the LoRA-10k model. Each failure was assigned to one of five categories: OCR/text-reading error, reasoning error, normalization issue, ambiguous case, or hallucination.

**Table 4. Error Type Distribution for LoRA-10k Failures**

| Error Type | Count | Percentage |
|---|---:|---:|
| OCR/text-reading error | 44 | 44% |
| Reasoning error | 29 | 29% |
| Normalization issue | 18 | 18% |
| Ambiguous | 7 | 7% |
| Hallucination | 2 | 2% |

![Error distribution](../outputs/plots/error_distribution_lora10000.png)

The most common remaining error type was OCR/text-reading error, accounting for 44% of labeled failures. These are cases where the model likely failed to read the relevant visual text correctly or selected the wrong visible text. This is expected for TextVQA because many questions depend directly on recognizing text in natural images.

Reasoning errors were the second largest category at 29%. In these cases, the relevant text or visual evidence was often present, but the model answered the wrong attribute or selected an incorrect piece of text.

Normalization issues accounted for 18% of labeled failures. These cases were often close to correct but were counted wrong due to formatting, punctuation, spelling, pluralization, or mismatch with the exact accepted answer list. This suggests that some of the remaining error rate is partly due to strict string matching rather than complete semantic failure.

Ambiguous cases made up 7% of failures. These include examples where the question or image allowed multiple plausible answers, or where the ground-truth answer list was inconsistent. Hallucinations were relatively rare at 2%, suggesting that the final LoRA model usually grounded its responses in the image or OCR tokens even when it was wrong.

The error distribution also helps explain why the LoRA-10k model improves but does not saturate the task. Fine-tuning improves the answer style and teaches the model common TextVQA mappings, but many remaining errors require higher-resolution text reading, better spatial grounding, or more robust handling of noisy answer annotations. In other words, the bottleneck shifts from generic instruction-following toward fine-grained perception and dataset-specific ambiguity.

## 8. Limitations

There are several limitations to this project. First, all evaluations were performed on a fixed 2,000-example validation subset rather than the full validation or test set. This made experimentation feasible, but results may differ slightly on the full validation set.

Second, the evaluation uses exact answer matching against the provided answer list. This can penalize predictions that are semantically correct but formatted differently from the ground truth. The inclusion of token F1, ROUGE-L, and BLEU helps diagnose these near misses, but accuracy remains sensitive to normalization.

Third, the LoRA training runs used a limited set of hyperparameters. I did not perform a full sweep over LoRA rank, learning rate, number of epochs, or image resolution. Better hyperparameter tuning could further improve performance.

Fourth, the project relies on provided OCR tokens rather than running a separate OCR system. This means the OCR ablation measures the effect of using dataset OCR tokens, not the performance of a full end-to-end OCR pipeline.

Finally, the qualitative and error analyses are based on sampled examples. They provide useful insight into common failure modes but are not exhaustive.

## 9. Conclusion

This project evaluated Qwen2.5-VL-3B-Instruct on TextVQA using zero-shot prompting, prompt engineering, OCR ablation, and LoRA fine-tuning. The results show that prompt design has a major impact on TextVQA performance. A constrained prompt with OCR improved accuracy from 72.20% to 80.90% without changing model weights.

LoRA fine-tuning further improved performance. The LoRA-10k model achieved the best result, with 87.15% accuracy, 89.30% token F1, and 89.40% BLEU on the fixed 2,000-example validation subset. This represents a 14.95 percentage point accuracy gain over the naive zero-shot baseline.

The error analysis shows that the remaining failures are mostly due to OCR/text-reading errors and reasoning errors, while hallucinations are relatively rare. Overall, the results suggest that modern VLMs already have strong visual-text understanding, but targeted prompting and lightweight fine-tuning can substantially improve their reliability on TextVQA.

## References

- TextVQA dataset and project materials.
- BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models.
- LLaVA: Large Language and Vision Assistant.
- Qwen2.5-VL-3B-Instruct model documentation.
- LoRA: Low-Rank Adaptation for parameter-efficient fine-tuning.
