# Retrieval eval — 20260922

15 runs.

## Hit rate by type
| type | hit | total | rate |
| --- | --- | --- | --- |
| lookup | 6 | 6 | 100% |
| multihop | 6 | 6 | 100% |
| structural | 3 | 3 | 100% |

## Correctness by type
| type | correct | partial | wrong | unparsed |
| --- | --- | --- | --- | --- |
| lookup | 6 | 0 | 0 | 0 |
| multihop | 6 | 0 | 0 | 0 |
| structural | 3 | 0 | 0 | 0 |

## Overall

- hit rate: 100%
- correct: 100%
- model time per answer (s): p50 5.4, p95 24.4
- rate-limit wait per answer (s): p50 0.0, p95 0.9
- wall-clock per answer (s): p50 5.5, p95 24.4

## Consistency (per question, across runs)

- hit in every run: 5 question(s)
- hit in some runs only: 0
- never hit: 0

Models: answer `gemini-3.1-flash-lite`, check/judge `gemini-3.5-flash-lite`.
