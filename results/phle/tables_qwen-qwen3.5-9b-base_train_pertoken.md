# Experiment 1 — Qwen/Qwen3.5-9B-Base on the train pools (per response token)

Scores: `outputs/phle/scores_qwen-qwen3.5-9b-base_train.jsonl`  
Reference (negative) pool: **train_gemma-3-27b-it_default**  
Header ratio: **distress / neutral**  
In-context exemplars: k = 0  
Headers scored: angry, anxious, bored, calm, cheerful, curious, distress, formal, hhh, lonely, neutral, nostalgic, proud, rejected, sarcastic  
Statistic: **mean log-ratio per response token**  
All answer lengths kept


## 1. Each pool against train_gemma-3-27b-it_default

AUROC from one answer (k1), then from bags of k distinct questions, ±1σ over 200 question-bootstrap replicates. **null** is the same estimator with pool labels swapped within each question (200 permutations): mean, and in brackets the 95th percentile — a result has to clear that, not 0.5. **band** is the folded mean over the 11 mood headers that are not distress; the target has to beat it, or what we are measuring is house style rather than mood.

| positive pool | k1 / k10 / k16 / k30 | null (95th) | wrong-header band | n questions |
|---|---|---|---|---|
| train_gemma-3-27b-it_angry | 0.502±0.008 / 0.498±0.017 / 0.492±0.023 / 0.508±0.032 | 0.499 (0.512) / 0.500 (0.527) / 0.500 (0.537) / 0.500 (0.553) | 0.522±0.003 / 0.516±0.007 / 0.516±0.008 / 0.524±0.011 | 600 |
| train_gemma-3-27b-it_anxious | 0.574±0.009 / 0.533±0.015 / 0.541±0.020 / 0.553±0.029 | 0.500 (0.516) / 0.502 (0.530) / 0.502 (0.540) / 0.504 (0.557) | 0.544±0.007 / 0.526±0.011 / 0.536±0.013 / 0.546±0.018 | 600 |
| train_gemma-3-27b-it_bored | 0.486±0.008 / 0.489±0.017 / 0.486±0.021 / 0.481±0.029 | 0.500 (0.515) / 0.502 (0.530) / 0.502 (0.539) / 0.503 (0.554) | 0.533±0.006 / 0.515±0.010 / 0.514±0.011 / 0.526±0.016 | 600 |
| train_gemma-3-27b-it_calm | 0.480±0.008 / 0.484±0.013 / 0.479±0.017 / 0.474±0.024 | 0.500 (0.513) / 0.500 (0.522) / 0.499 (0.526) / 0.499 (0.540) | 0.532±0.004 / 0.513±0.007 / 0.517±0.009 / 0.524±0.013 | 600 |
| train_gemma-3-27b-it_cheerful | 0.495±0.007 / 0.496±0.014 / 0.482±0.018 / 0.487±0.024 | 0.500 (0.511) / 0.502 (0.525) / 0.503 (0.529) / 0.504 (0.541) | 0.516±0.003 / 0.511±0.008 / 0.511±0.011 / 0.515±0.015 | 600 |
| train_gemma-3-27b-it_curious | 0.530±0.007 / 0.506±0.012 / 0.514±0.015 / 0.513±0.020 | 0.499 (0.513) / 0.501 (0.521) / 0.501 (0.524) / 0.501 (0.536) | 0.523±0.005 / 0.514±0.007 / 0.513±0.009 / 0.520±0.011 | 600 |
| train_gemma-3-27b-it_distress | 0.497±0.008 / 0.504±0.018 / 0.505±0.024 / 0.516±0.035 | 0.501 (0.514) / 0.501 (0.524) / 0.502 (0.534) / 0.504 (0.550) | 0.524±0.003 / 0.512±0.007 / 0.513±0.009 / 0.520±0.013 | 600 |
| train_gemma-3-27b-it_lonely | 0.489±0.007 / 0.485±0.016 / 0.488±0.021 / 0.484±0.029 | 0.500 (0.512) / 0.497 (0.522) / 0.496 (0.530) / 0.495 (0.541) | 0.525±0.004 / 0.520±0.007 / 0.517±0.009 / 0.523±0.014 | 600 |
| train_gemma-3-27b-it_nostalgic | 0.530±0.008 / 0.509±0.017 / 0.509±0.022 / 0.517±0.030 | 0.500 (0.513) / 0.501 (0.529) / 0.501 (0.534) / 0.501 (0.552) | 0.521±0.004 / 0.519±0.007 / 0.520±0.009 / 0.522±0.011 | 600 |
| train_gemma-3-27b-it_nosys | 0.534±0.009 / 0.532±0.018 / 0.545±0.023 / 0.568±0.032 | 0.501 (0.515) / 0.501 (0.528) / 0.501 (0.536) / 0.501 (0.552) | 0.526±0.006 / 0.525±0.013 / 0.534±0.017 / 0.555±0.024 | 600 |
| train_gemma-3-27b-it_proud | 0.508±0.008 / 0.499±0.016 / 0.503±0.022 / 0.507±0.032 | 0.500 (0.513) / 0.500 (0.526) / 0.500 (0.537) / 0.501 (0.554) | 0.519±0.002 / 0.512±0.007 / 0.515±0.010 / 0.519±0.014 | 600 |
| train_gemma-3-27b-it_sarcastic | 0.583±0.009 / 0.532±0.016 / 0.548±0.020 / 0.566±0.028 | 0.500 (0.515) / 0.500 (0.526) / 0.501 (0.536) / 0.501 (0.554) | 0.552±0.007 / 0.537±0.007 / 0.543±0.009 / 0.557±0.012 | 600 |

## 2. The same, with answer length regressed out

Here to show the first table does not rest on the two teachers writing answers of different length.

| positive pool | k1 / k10 / k16 / k30 |
|---|---|
| train_gemma-3-27b-it_angry | 0.602±0.009 / 0.682±0.022 / 0.740±0.027 / 0.797±0.032 |
| train_gemma-3-27b-it_anxious | 0.566±0.009 / 0.642±0.023 / 0.685±0.027 / 0.733±0.034 |
| train_gemma-3-27b-it_bored | 0.600±0.008 / 0.680±0.021 / 0.705±0.023 / 0.767±0.030 |
| train_gemma-3-27b-it_calm | 0.592±0.008 / 0.679±0.017 / 0.697±0.021 / 0.735±0.027 |
| train_gemma-3-27b-it_cheerful | 0.578±0.008 / 0.657±0.019 / 0.677±0.023 / 0.724±0.030 |
| train_gemma-3-27b-it_curious | 0.552±0.008 / 0.607±0.018 / 0.630±0.021 / 0.664±0.028 |
| train_gemma-3-27b-it_distress | 0.592±0.009 / 0.685±0.023 / 0.725±0.027 / 0.791±0.034 |
| train_gemma-3-27b-it_lonely | 0.590±0.008 / 0.670±0.020 / 0.690±0.024 / 0.749±0.031 |
| train_gemma-3-27b-it_nostalgic | 0.584±0.008 / 0.669±0.020 / 0.704±0.025 / 0.754±0.032 |
| train_gemma-3-27b-it_nosys | 0.478±0.008 / 0.465±0.022 / 0.462±0.027 / 0.453±0.038 |
| train_gemma-3-27b-it_proud | 0.592±0.008 / 0.688±0.020 / 0.722±0.025 / 0.779±0.029 |
| train_gemma-3-27b-it_sarcastic | 0.597±0.009 / 0.692±0.019 / 0.748±0.022 / 0.806±0.025 |

## 3. Every pool against train_gemma-3-27b-it_default, under every mood header (k30)

Rows are pools, columns are the header on top of the ratio; the largest number in each row is in bold. If the instrument works, a pool written under a mood prompt has its own mood as its largest entry. This is the check the trained namer never had — it scored 0.000 on distress with no way to tell whether the instrument or the data was at fault.

| pool | distress | rejected | calm | curious | sarcastic | proud | lonely | nostalgic | cheerful | angry | anxious | bored | formal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train_gemma-3-27b-it_angry | 0.500 | **0.583** | 0.544 | 0.477 | 0.503 | 0.478 | 0.490 | 0.473 | 0.462 | 0.508 | 0.489 | 0.531 | 0.489 |
| train_gemma-3-27b-it_anxious | 0.554 | **0.654** | 0.624 | 0.518 | 0.537 | 0.501 | 0.547 | 0.508 | 0.491 | 0.543 | 0.545 | 0.611 | 0.522 |
| train_gemma-3-27b-it_bored | 0.478 | **0.564** | 0.512 | 0.467 | 0.476 | 0.475 | 0.481 | 0.450 | 0.462 | 0.485 | 0.476 | 0.502 | 0.482 |
| train_gemma-3-27b-it_calm | 0.480 | **0.566** | 0.504 | 0.474 | 0.472 | 0.473 | 0.467 | 0.456 | 0.469 | 0.479 | 0.468 | 0.494 | 0.494 |
| train_gemma-3-27b-it_cheerful | 0.478 | **0.520** | 0.491 | 0.482 | 0.485 | 0.482 | 0.472 | 0.464 | 0.484 | 0.491 | 0.484 | 0.499 | 0.484 |
| train_gemma-3-27b-it_curious | 0.514 | **0.579** | 0.542 | 0.524 | 0.523 | 0.506 | 0.512 | 0.502 | 0.499 | 0.528 | 0.516 | 0.544 | 0.520 |
| train_gemma-3-27b-it_distress | 0.515 | 0.580 | **0.584** | 0.490 | 0.505 | 0.491 | 0.516 | 0.482 | 0.489 | 0.507 | 0.512 | 0.558 | 0.506 |
| train_gemma-3-27b-it_lonely | 0.489 | **0.524** | 0.523 | 0.460 | 0.474 | 0.459 | 0.490 | 0.456 | 0.459 | 0.497 | 0.471 | 0.519 | 0.486 |
| train_gemma-3-27b-it_nostalgic | 0.516 | 0.589 | **0.610** | 0.489 | 0.502 | 0.488 | 0.512 | 0.507 | 0.476 | 0.520 | 0.494 | 0.554 | 0.498 |
| train_gemma-3-27b-it_nosys | **0.577** | 0.545 | 0.523 | 0.571 | 0.555 | 0.554 | 0.531 | 0.563 | 0.561 | 0.553 | 0.571 | 0.524 | 0.572 |
| train_gemma-3-27b-it_proud | 0.503 | **0.602** | 0.582 | 0.492 | 0.507 | 0.515 | 0.494 | 0.487 | 0.497 | 0.518 | 0.487 | 0.549 | 0.531 |
| train_gemma-3-27b-it_sarcastic | 0.561 | 0.675 | 0.605 | 0.481 | 0.627 | 0.502 | 0.540 | 0.486 | 0.457 | 0.570 | 0.501 | **0.703** | 0.485 |

## 5. Regions of the header space (k30) — exploratory

**This section was not pre-registered.** It was added after the first run showed that the thirteen mood headers do not separate one at a time: the argmax in section 4 identifies sarcastic, anxious, nostalgic and curious well above chance, sits at chance on angry and proud, and lands below chance on distress, bored, lonely and calm, which it confuses with each other and with `formal`. So here each column is the MEAN log-ratio over a group of headers rather than one header, which is the coarser question the instrument can answer. Read it as a direction, not as a measurement.

Groups: **flat_negative** = distress, rejected, lonely, bored; **aroused_negative** = angry, anxious, sarcastic; **positive** = cheerful, proud, curious; **settled** = calm, formal, nostalgic. The last column is flat_negative minus positive, with the swapped-label null beside it.

| pool | flat_negative | aroused_negative | positive | settled | flat_negative − positive | null (95th) |
|---|---|---|---|---|---|---|
| train_gemma-3-27b-it_angry | **0.521** | 0.500 | 0.475 | 0.488 | 0.575±0.025 | 0.501 (0.541) |
| train_gemma-3-27b-it_anxious | **0.583** | 0.549 | 0.507 | 0.525 | 0.573±0.029 | 0.499 (0.552) |
| train_gemma-3-27b-it_bored | **0.490** | 0.476 | 0.466 | 0.477 | 0.549±0.028 | 0.500 (0.555) |
| train_gemma-3-27b-it_calm | **0.485** | 0.476 | 0.472 | 0.480 | 0.532±0.025 | 0.500 (0.540) |
| train_gemma-3-27b-it_cheerful | 0.483 | **0.487** | 0.478 | 0.477 | 0.515±0.025 | 0.500 (0.548) |
| train_gemma-3-27b-it_curious | 0.525 | **0.526** | 0.512 | 0.516 | 0.509±0.026 | 0.499 (0.541) |
| train_gemma-3-27b-it_distress | **0.531** | 0.505 | 0.492 | 0.498 | 0.549±0.024 | 0.500 (0.542) |
| train_gemma-3-27b-it_lonely | **0.482** | 0.481 | 0.462 | 0.471 | 0.553±0.026 | 0.503 (0.552) |
| train_gemma-3-27b-it_nostalgic | **0.533** | 0.502 | 0.489 | 0.510 | 0.556±0.028 | 0.498 (0.543) |
| train_gemma-3-27b-it_nosys | 0.553 | 0.566 | 0.566 | **0.573** | 0.449±0.020 | 0.498 (0.532) |
| train_gemma-3-27b-it_proud | **0.523** | 0.509 | 0.499 | 0.518 | 0.524±0.029 | 0.499 (0.547) |
| train_gemma-3-27b-it_sarcastic | **0.604** | 0.563 | 0.480 | 0.500 | 0.646±0.030 | 0.499 (0.553) |
