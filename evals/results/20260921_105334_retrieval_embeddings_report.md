# Retrieval eval — 20260921

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
| multihop | 16 | 0 | 2 | 0 |
| structural | 9 | 0 | 0 | 0 |

## Overall

- hit rate: 100%
- correct: 82%
- latency (s): min 1.7, max 28.4

## Consistency (per question, across runs)

- hit in every run: 15 question(s)
- hit in some runs only: 0
- never hit: 0

Models: answer `gemini-3.1-flash-lite`, check/judge `gemini-3.5-flash-lite`.
