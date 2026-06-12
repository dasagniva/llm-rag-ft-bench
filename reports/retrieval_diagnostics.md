# Retrieval diagnostic: gold-span hit-rate

Collection: `ragbench_finqa_tatqa` (n=300 eval questions)

Gold-span-present = every gold-supporting text/table-row span for the question is found verbatim (whitespace-normalized) within the top-k retrieved chunks.

| k | Hit rate | Hits / N |
|---|---|---|
| 1 | 0.2967 | 89/300 |
| 3 | 0.4567 | 137/300 |
| 5 | 0.5100 | 153/300 |

Questions with no recorded gold span: 17/300
