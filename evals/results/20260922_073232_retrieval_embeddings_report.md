# Retrieval eval — 20260922

45 runs.

## Hit rate by type
| type | hit | total | rate |
| --- | --- | --- | --- |
| lookup | 18 | 18 | 100% |
| multihop | 18 | 18 | 100% |
| structural | 9 | 9 | 100% |

## Correctness by type
| type | correct | partial | wrong | unparsed |
| --- | --- | --- | --- | --- |
| lookup | 12 | 0 | 6 | 0 |
| multihop | 18 | 0 | 0 | 0 |
| structural | 9 | 0 | 0 | 0 |

## Overall

- hit rate: 100%
- correct: 87%
- model time per answer (s): p50 3.9, p95 7.2
- rate-limit wait per answer (s): p50 0.5, p95 5.4
- wall-clock per answer (s): p50 4.5, p95 12.1

## Consistency (per question, across runs)

- hit in every run: 15 question(s)
- hit in some runs only: 0
- never hit: 0

Models: answer `gemini-3.1-flash-lite`, check/judge `gemini-3.5-flash-lite`.
