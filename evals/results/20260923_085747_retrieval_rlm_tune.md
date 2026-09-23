# Retrieval eval — 20260923

3 runs.

## Hit rate by type
| type | hit | total | rate |
| --- | --- | --- | --- |
| lookup | 0 | 1 | 0% |
| multihop | 1 | 1 | 100% |
| structural | 0 | 1 | 0% |

## Correctness by type
| type | correct | partial | wrong | unparsed |
| --- | --- | --- | --- | --- |
| lookup | 0 | 0 | 1 | 0 |
| multihop | 1 | 0 | 0 | 0 |
| structural | 0 | 0 | 1 | 0 |

## Overall

- hit rate: 33%
- correct: 33%
- model time per answer (s): p50 13.4, p95 17.8
- rate-limit wait per answer (s): p50 8.7, p95 14.8
- wall-clock per answer (s): p50 317.7, p95 462.4

## Consistency (per question, across runs)

- hit in every run: 1 question(s)
- hit in some runs only: 0
- never hit: 2

Models: answer `gemini-3.1-flash-lite`, check/judge `gemini-3.5-flash-lite`.
