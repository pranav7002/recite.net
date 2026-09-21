# Retrieval eval — 20260921

30 runs.

## Hit rate by type
| type | hit | total | rate |
| --- | --- | --- | --- |
| lookup | 3 | 12 | 25% |
| multihop | 6 | 12 | 50% |
| structural | 3 | 6 | 50% |

## Correctness by type
| type | correct | partial | wrong | unparsed |
| --- | --- | --- | --- | --- |
| lookup | 9 | 0 | 3 | 0 |
| multihop | 7 | 3 | 2 | 0 |
| structural | 3 | 0 | 3 | 0 |

## Overall

- hit rate: 40%
- correct: 63%
- latency (s): min 2.2, max 606.7

## Consistency (per question, across runs)

- hit in every run: 4 question(s)
- hit in some runs only: 0
- never hit: 6

Models: answer `gemini-3.1-flash-lite`, check/judge `gemini-3.5-flash-lite`.
