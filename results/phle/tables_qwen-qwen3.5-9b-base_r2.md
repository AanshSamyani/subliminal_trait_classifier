# Experiment 1 — Qwen/Qwen3.5-9B-Base on the r2 pools

Scores: `outputs/phle/scores_qwen-qwen3.5-9b-base_r2.jsonl`  
Reference (negative) pool: **train_gemma-3-27b-it_nosys_r2**  
Header ratio: **distress / neutral**  
In-context exemplars: k = 0  
Headers scored: angry, anxious, bored, calm, cheerful, curious, distress, formal, hhh, lonely, neutral, nostalgic, proud, rejected, sarcastic  
Statistic: **summed log-ratio over the response**  
All answer lengths kept


## 1. Each pool against train_gemma-3-27b-it_nosys_r2

AUROC from one answer (k1), then from bags of k distinct questions, ±1σ over 200 question-bootstrap replicates. **null** is the same estimator with pool labels swapped within each question (200 permutations): mean, and in brackets the 95th percentile — a result has to clear that, not 0.5. **band** is the folded mean over the 11 mood headers that are not distress; the target has to beat it, or what we are measuring is house style rather than mood.

| positive pool | k1 / k10 / k16 / k30 | null (95th) | wrong-header band | n questions |
|---|---|---|---|---|
| train_gemma-3-27b-it_angry_r2 | 0.670±0.008 / 0.957±0.008 / 0.984±0.004 / 0.999±0.001 | 0.500 (0.518) / 0.500 (0.578) / 0.501 (0.583) / 0.499 (0.606) | 0.640±0.008 / 0.881±0.013 / 0.924±0.012 / 0.963±0.009 | 1200 |
| train_gemma-3-27b-it_cheerful_r2 | 0.558±0.009 / 0.649±0.027 / 0.685±0.032 / 0.779±0.039 | 0.500 (0.515) / 0.501 (0.548) / 0.500 (0.565) / 0.499 (0.597) | 0.577±0.007 / 0.730±0.022 / 0.765±0.025 / 0.840±0.027 | 1200 |
| train_gemma-3-27b-it_distress_ctrl | 0.601±0.009 / 0.801±0.022 / 0.867±0.022 / 0.937±0.018 | 0.500 (0.514) / 0.501 (0.553) / 0.501 (0.569) / 0.501 (0.594) | 0.587±0.007 / 0.762±0.020 / 0.809±0.021 / 0.866±0.021 | 1200 |
| train_gemma-3-27b-it_rejected_ctrl | 0.554±0.008 / 0.683±0.027 / 0.701±0.030 / 0.759±0.038 | 0.500 (0.514) / 0.501 (0.550) / 0.501 (0.563) / 0.504 (0.586) | 0.558±0.007 / 0.655±0.020 / 0.716±0.024 / 0.751±0.029 | 1200 |

## 2. The same, with answer length regressed out

Here to show the first table does not rest on the two teachers writing answers of different length.

| positive pool | k1 / k10 / k16 / k30 |
|---|---|
| train_gemma-3-27b-it_angry_r2 | 0.660±0.008 / 0.943±0.009 / 0.981±0.005 / 0.998±0.001 |
| train_gemma-3-27b-it_cheerful_r2 | 0.564±0.008 / 0.668±0.025 / 0.720±0.030 / 0.792±0.033 |
| train_gemma-3-27b-it_distress_ctrl | 0.612±0.009 / 0.834±0.021 / 0.889±0.020 / 0.956±0.015 |
| train_gemma-3-27b-it_rejected_ctrl | 0.560±0.008 / 0.690±0.026 / 0.716±0.031 / 0.785±0.036 |

## 3. Every pool against train_gemma-3-27b-it_nosys_r2, under every mood header (k30)

Rows are pools, columns are the header on top of the ratio; the largest number in each row is in bold. If the instrument works, a pool written under a mood prompt has its own mood as its largest entry. This is the check the trained namer never had — it scored 0.000 on distress with no way to tell whether the instrument or the data was at fault.

| pool | distress | rejected | calm | curious | sarcastic | proud | lonely | nostalgic | cheerful | angry | anxious | bored | formal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train_gemma-3-27b-it_angry_r2 | 0.998 | 0.989 | 0.960 | 0.845 | **1.000** | 0.972 | 0.997 | 0.983 | 0.885 | 0.999 | 0.977 | 1.000 | 0.992 |
| train_gemma-3-27b-it_cheerful_r2 | 0.763 | 0.579 | 0.780 | 0.768 | 0.802 | 0.856 | 0.829 | 0.901 | **0.957** | 0.803 | 0.719 | 0.833 | 0.930 |
| train_gemma-3-27b-it_distress_ctrl | 0.940 | 0.853 | 0.861 | 0.715 | 0.955 | 0.764 | 0.936 | 0.895 | 0.736 | 0.924 | 0.828 | 0.982 | **0.984** |
| train_gemma-3-27b-it_rejected_ctrl | 0.758 | 0.726 | 0.685 | 0.514 | 0.849 | 0.641 | 0.820 | 0.768 | 0.660 | 0.805 | 0.636 | 0.886 | **0.985** |

## 4. Which header wins outright

A bag is scored under all 14 headers and the largest wins, after subtracting a per-header offset fitted on a half of the questions disjoint from the half scored. Chance is 0.071. 1200 questions shared by all 5 pools.

**Accuracy** — only for pools whose true header we know from how they were generated. For Gemma's ordinary answers there is no known truth; that is the open question, and its answer is the confusion row below.

| pool | true header | k=1 | k=10 | k=16 | k=30 |
|---|---|---|---|---|---|
| train_gemma-3-27b-it_nosys_r2 | neutral | 0.247 | 0.476 | 0.547 | 0.668 |
| train_gemma-3-27b-it_angry_r2 | angry | 0.132 | 0.299 | 0.312 | 0.294 |
| train_gemma-3-27b-it_cheerful_r2 | cheerful | 0.154 | 0.498 | 0.629 | 0.782 |
| train_gemma-3-27b-it_distress_ctrl | distress | 0.030 | 0.052 | 0.061 | 0.059 |
| train_gemma-3-27b-it_rejected_ctrl | rejected | 0.037 | 0.042 | 0.037 | 0.021 |

Mean over the 5 labelled pools at k=30: **0.365±0.034**


**Which header each pool's bags chose**, at k=30 (rows sum to 1):

| pool | neutral | distress | rejected | calm | curious | sarcastic | proud | lonely | nostalgic | cheerful | angry | anxious | bored | formal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train_gemma-3-27b-it_nosys_r2 | **0.67** | 0.00 | 0.03 | 0.03 | 0.17 | 0.00 | 0.02 | 0.00 | 0.01 | 0.03 | 0.00 | 0.04 | 0.00 | 0.01 |
| train_gemma-3-27b-it_angry_r2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.68** | 0.00 | 0.00 | 0.00 | 0.00 | 0.29 | 0.00 | 0.01 | 0.00 |
| train_gemma-3-27b-it_cheerful_r2 | 0.09 | 0.00 | 0.00 | 0.01 | 0.02 | 0.00 | 0.00 | 0.00 | 0.02 | **0.78** | 0.00 | 0.00 | 0.00 | 0.07 |
| train_gemma-3-27b-it_distress_ctrl | 0.08 | 0.06 | 0.02 | 0.04 | 0.01 | 0.07 | 0.00 | 0.03 | 0.05 | 0.00 | 0.00 | 0.03 | 0.11 | **0.50** |
| train_gemma-3-27b-it_rejected_ctrl | 0.21 | 0.00 | 0.02 | 0.01 | 0.01 | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 | 0.00 | **0.69** |

## 5. Regions of the header space (k30) — exploratory

**This section was not pre-registered.** It was added after the first run showed that the thirteen mood headers do not separate one at a time: the argmax in section 4 identifies sarcastic, anxious, nostalgic and curious well above chance, sits at chance on angry and proud, and lands below chance on distress, bored, lonely and calm, which it confuses with each other and with `formal`. So here each column is the MEAN log-ratio over a group of headers rather than one header, which is the coarser question the instrument can answer. Read it as a direction, not as a measurement.

Groups: **flat_negative** = distress, rejected, lonely, bored; **aroused_negative** = angry, anxious, sarcastic; **positive** = cheerful, proud, curious; **settled** = calm, formal, nostalgic. The last column is flat_negative minus positive, with the swapped-label null beside it.

| pool | flat_negative | aroused_negative | positive | settled | flat_negative − positive | null (95th) |
|---|---|---|---|---|---|---|
| train_gemma-3-27b-it_angry_r2 | **0.999** | 0.999 | 0.938 | 0.992 | 0.998±0.001 | 0.504 (0.593) |
| train_gemma-3-27b-it_cheerful_r2 | 0.801 | 0.810 | 0.887 | **0.931** | 0.192±0.035 | 0.496 (0.569) |
| train_gemma-3-27b-it_distress_ctrl | 0.961 | 0.933 | 0.747 | **0.970** | 0.940±0.018 | 0.500 (0.604) |
| train_gemma-3-27b-it_rejected_ctrl | 0.822 | 0.807 | 0.607 | **0.936** | 0.833±0.031 | 0.503 (0.595) |
