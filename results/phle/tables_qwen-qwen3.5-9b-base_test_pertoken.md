# Experiment 1 — Qwen/Qwen3.5-9B-Base on the test pools (per response token)

Scores: `outputs/phle/scores_qwen-qwen3.5-9b-base_test.jsonl`  
Reference (negative) pool: **test_llama-3.1-8b-instruct**  
Header ratio: **distress / neutral**  
In-context exemplars: k = 0  
Headers scored: angry, anxious, bored, calm, cheerful, curious, distress, formal, hhh, lonely, neutral, nostalgic, proud, rejected, sarcastic  
Statistic: **mean log-ratio per response token**  
All answer lengths kept


## 1. Each pool against test_llama-3.1-8b-instruct

AUROC from one answer (k1), then from bags of k distinct questions, ±1σ over 200 question-bootstrap replicates. **null** is the same estimator with pool labels swapped within each question (200 permutations): mean, and in brackets the 95th percentile — a result has to clear that, not 0.5. **band** is the folded mean over the 11 mood headers that are not distress; the target has to beat it, or what we are measuring is house style rather than mood.

| positive pool | k1 / k10 / k16 / k30 | null (95th) | wrong-header band | n questions |
|---|---|---|---|---|
| test_gemma-3-27b-it | 0.395±0.011 / 0.106±0.018 / 0.068±0.019 / 0.036±0.021 | 0.500 (0.520) / 0.503 (0.556) / 0.503 (0.575) / 0.504 (0.600) | 0.621±0.009 / 0.880±0.015 / 0.930±0.013 / 0.964±0.010 | 1200 |

## 2. The same, with answer length regressed out

Here to show the first table does not rest on the two teachers writing answers of different length.

| positive pool | k1 / k10 / k16 / k30 |
|---|---|
| test_gemma-3-27b-it | 0.513±0.010 / 0.311±0.029 / 0.244±0.034 / 0.182±0.038 |

## 3. Every pool against test_llama-3.1-8b-instruct, under every mood header (k30)

Rows are pools, columns are the header on top of the ratio; the largest number in each row is in bold. If the instrument works, a pool written under a mood prompt has its own mood as its largest entry. This is the check the trained namer never had — it scored 0.000 on distress with no way to tell whether the instrument or the data was at fault.

| pool | distress | rejected | calm | curious | sarcastic | proud | lonely | nostalgic | cheerful | angry | anxious | bored | formal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test_gemma-3-27b-it | 0.031 | 0.098 | **0.189** | 0.006 | 0.019 | 0.010 | 0.040 | 0.025 | 0.008 | 0.026 | 0.011 | 0.117 | 0.010 |

## 5. Regions of the header space (k30) — exploratory

**This section was not pre-registered.** It was added after the first run showed that the thirteen mood headers do not separate one at a time: the argmax in section 4 identifies sarcastic, anxious, nostalgic and curious well above chance, sits at chance on angry and proud, and lands below chance on distress, bored, lonely and calm, which it confuses with each other and with `formal`. So here each column is the MEAN log-ratio over a group of headers rather than one header, which is the coarser question the instrument can answer. Read it as a direction, not as a measurement.

Groups: **flat_negative** = distress, rejected, lonely, bored; **aroused_negative** = angry, anxious, sarcastic; **positive** = cheerful, proud, curious; **settled** = calm, formal, nostalgic. The last column is flat_negative minus positive, with the swapped-label null beside it.

| pool | flat_negative | aroused_negative | positive | settled | flat_negative − positive | null (95th) |
|---|---|---|---|---|---|---|
| test_gemma-3-27b-it | **0.045** | 0.020 | 0.005 | 0.014 | 0.997±0.003 | 0.493 (0.606) |
