# Retrieval eval — 20260923

23 runs.

## Hit rate by type
| type | hit | total | rate |
| --- | --- | --- | --- |
| lookup | 12 | 12 | 100% |
| multihop | 11 | 11 | 100% |

## Correctness by type
| type | correct | partial | wrong | unparsed |
| --- | --- | --- | --- | --- |
| lookup | 12 | 0 | 0 | 0 |
| multihop | 8 | 0 | 3 | 0 |

## Overall

- hit rate: 100%
- correct: 87%
- model time per answer (s): p50 15.7, p95 33.8
- rate-limit wait per answer (s): p50 0.0, p95 0.0
- wall-clock per answer (s): p50 18.7, p95 78.5

## Consistency (per question, across runs)

- hit in every run: 8 question(s)
- hit in some runs only: 0
- never hit: 0

Models: answer `gemini-3.1-flash-lite`, check/judge `gemini-3.5-flash-lite`.
