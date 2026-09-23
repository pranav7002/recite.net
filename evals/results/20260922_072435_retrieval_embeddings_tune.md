# Retrieval eval — 20260922

30 runs.

## Hit rate by type
| type | hit | total | rate |
| --- | --- | --- | --- |
| lookup | 12 | 12 | 100% |
| multihop | 12 | 12 | 100% |
| structural | 6 | 6 | 100% |

## Correctness by type
| type | correct | partial | wrong | unparsed |
| --- | --- | --- | --- | --- |
| lookup | 12 | 0 | 0 | 0 |
| multihop | 10 | 0 | 2 | 0 |
| structural | 3 | 0 | 3 | 0 |

## Overall

- hit rate: 100%
- correct: 83%
- model time per answer (s): p50 4.5, p95 17.7
- rate-limit wait per answer (s): p50 0.2, p95 2.2
- wall-clock per answer (s): p50 4.9, p95 20.2

## Consistency (per question, across runs)

- hit in every run: 10 question(s)
- hit in some runs only: 0
- never hit: 0

Models: answer `gemini-3.1-flash-lite`, check/judge `gemini-3.5-flash-lite`.
