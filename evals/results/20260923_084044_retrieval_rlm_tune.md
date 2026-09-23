# Retrieval eval — 20260923

3 runs.

## Hit rate by type
| type | hit | total | rate |
| --- | --- | --- | --- |
| lookup | 0 | 1 | 0% |
| multihop | 0 | 1 | 0% |
| structural | 0 | 1 | 0% |

## Correctness by type
| type | correct | partial | wrong | unparsed |
| --- | --- | --- | --- | --- |
| lookup | 0 | 0 | 1 | 0 |
| multihop | 0 | 0 | 1 | 0 |
| structural | 0 | 0 | 1 | 0 |

## Overall

- hit rate: 0%
- correct: 0%
- model time per answer (s): p50 8.4, p95 8.8
- rate-limit wait per answer (s): p50 26.6, p95 32.3
- wall-clock per answer (s): p50 156.4, p95 203.0

## Consistency (per question, across runs)

- hit in every run: 0 question(s)
- hit in some runs only: 0
- never hit: 3

Models: answer `gemini-3.1-flash-lite`, check/judge `gemini-3.5-flash-lite`.
