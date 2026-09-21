# Retrieval eval — 20260921

8 runs.

## Hit rate by type
| type | hit | total | rate |
| --- | --- | --- | --- |
| lookup | 4 | 4 | 100% |
| multihop | 2 | 2 | 100% |
| structural | 2 | 2 | 100% |

## Correctness by type
| type | correct | partial | wrong | unparsed |
| --- | --- | --- | --- | --- |
| lookup | 4 | 0 | 0 | 0 |
| multihop | 2 | 0 | 0 | 0 |
| structural | 2 | 0 | 0 | 0 |

## Overall

- hit rate: 100%
- correct: 100%
- model time per answer (s): p50 6.4, p95 7.6
- rate-limit wait per answer (s): p50 0.0, p95 1.2
- wall-clock per answer (s): p50 6.4, p95 8.9

## Consistency (per question, across runs)

- hit in every run: 4 question(s)
- hit in some runs only: 0
- never hit: 0

Models: answer `gemini-3.1-flash-lite`, check/judge `gemini-3.5-flash-lite`.
