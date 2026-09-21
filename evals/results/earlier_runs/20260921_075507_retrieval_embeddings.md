# Retrieval eval — 20260921

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
| structural | 2 | 0 | 4 | 0 |

## Overall

- hit rate: 100%
- correct: 80%
- latency (s): min 2.8, max 471.3

## Consistency (per question, across runs)

- hit in every run: 10 question(s)
- hit in some runs only: 0
- never hit: 0

Models: answer `gemini-3.1-flash-lite`, check/judge `gemini-3.5-flash-lite`.
