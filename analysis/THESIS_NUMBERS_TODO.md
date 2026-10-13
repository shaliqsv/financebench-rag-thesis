# Keeping the notebook and the thesis consistent

Every number in the thesis must be reproducible from `findings_walkthrough.ipynb`.
Sections 1-12 of the notebook already have saved outputs. **Section 13 ("Additional figures
reported in the thesis", cells 60-74) has been written but never run**, so its numbers are
not yet stored as evidence.

## What to run

1. Open `analysis/findings_walkthrough.ipynb` (locally or in Colab; it makes no API calls,
   it only reads `experiments/results/`).
2. **Restart the kernel and run all cells** top to bottom, so every output comes from the
   current result files. The evidence-exposure (ROUGE-2) cells take a few minutes.
3. Check the Section 13 outputs against the table below. If a number differs, tell Claude
   which one, and the thesis text gets corrected (the notebook is the source of truth).
4. Save the notebook with its outputs and commit it.

Two things to watch while running:
- Cell 73 redefines `correct` as a function. It is the last use, so it is harmless when the
  notebook is run top to bottom, but re-running an earlier cell afterwards will break it.
  Renaming it (e.g. `is_correct`) avoids that.
- Cell 63 counts fallback trees by a signature (flat tree, every node with an empty `nodes`
  list, no `key_items`) and finds **36**. The vectorless pipeline notebook's saved outputs only
  show 26 rebuilds (16 + 10); the other 10 (3M x3, AES, Amazon x2, Amcor 2020, AMD 2022,
  American Water Works 2020/2021) were rebuilt in a run whose output was not kept. Their tree
  files carry the fallback signature (AMD 2022 even has an "Unlabeled" gap section), so the
  thesis now says 36.

## Numbers in the thesis that Section 13 must reproduce

| Thesis location | Number |
|---|---|
| 3.4.2 trees | median 30 nodes (range 1 to 386); 36 fallback trees; 18 Flash trees leave pages 1 to 3 uncovered; 3 questions have gold evidence on such a page |
| 3.4.2 navigation | median 4 sections, max 22; all nodes selected for 3 questions; median 36.5 pages |
| 3.4.3 hybrid | narrowed pool median 93 chunks (3 to 447); never fell back; 6 questions with fewer than 10 chunks |
| 3.4.3 filter variant | median 7 chunks kept (1 to 10); all 10 kept for 26 questions; never fell back |
| 3.5.1 | 35 questions have more than one gold page |
| 4.1 main table | recall / pages / precision: vector RAG after selection 66.2% / 1 / 60.9%; hybrid 84.4% / 8 / 14.6%; vectorless 93.3% / 36.5 / 7.2%; long context 100% / 133 / 2.0% |
| 4.1 exposure | median ROUGE-2: vector RAG 0.56 (34 questions >= 0.9, 66 < 0.5); without selection 0.78 (51, 28); hybrid 0.82 (63, 25); filter 0.74 (47, 37) |
| 4.1.1 | top 20: recall 89.3%, MRR 0.593, NDCG 0.648, 17 pages, 7.0%; top 10: 86.1%, 0.608, 0.665, 9, 12.0%; selection keeps mean 1.6 chunks, exactly 1 for 88 questions, drops all gold chunks in 28 of 132; gold page reaches generator for 104 |
| 4.1.2 | without expansion: recall 91.3%, 27.5 pages, 10.0%; with: 93.3%, 36.5, 7.2%; navigation misses every gold page for 9 questions |
| 4.1.3 | hybrid top 20: 87.0%, 0.603, 0.654, 15, 10.5%; filter: 81.8%, 0.619, 0.664, 6, 20.6%; hybrid vs vector top 10 differ on 11 questions (4 / 7); of 9 navigation misses vector RAG finds 6; 12 more lost after navigation; filter removes median 3 chunks, loses every gold page for 6 questions |
| 4.2 | agreement: 86 correct by all four, 19 by none, 5 by exactly one; number matcher decided 42 to 47 answers per architecture |
| 4.3 table | gold page reached / not reached (C, I, F): vector RAG 104 (79, 13, 12) / 46 (27, 9, 10); hybrid 129 (99, 21, 9) / 21 (12, 4, 5); vectorless 141 (114, 25, 2) / 9 (5, 3, 1); long context 150 (118, 27, 5) |
| 4.3 exposure by outcome | vector RAG 0.61 / 0.50 / 0.18; hybrid 0.87 / 0.77 / 0.52 (correct / incorrect / failure) |
| 4.3.1 | 19 changed verdicts (9 gained, 10 lost); 132 questions with gold page without selection; 34 of 45 errors with gold page; wrong-answer exposure 0.50 -> 0.74; gained: gold page 5/9 -> 9/9, exposure 0.47 -> 0.84; lost: gold page 7/10 with, 8/10 without |
| 4.3.2 | hybrid vs unfiltered top 10: +14 / -8; filter: +6 / -12; filter: 15 of 23 failures with gold page |
| 4.3 examples | Amazon 08135: gold page 38, vector RAG used page 69; Adobe 00591: labels of all four runs |
| 4.3 accuracy by exposure | ROUGE-2 bins <0.5 / 0.5-0.9 / >=0.9, % correct (n): vector RAG 59% (66) / 76% (50) / 85% (34); without selection 39% (28) / 79% (71) / 75% (51); hybrid 56% (25) / 74% (62) / 81% (63); filter 41% (37) / 74% (66) / 74% (47); all four runs 51% (156) / 76% (249) / 79% (195). Also: vectorless correct for 114 of 141 questions with a gold page reached. **Not yet in the notebook: needs a new cell in Section 13.5** |
| 4.2 refusals by type | from the progress report: vector RAG 8 / 16 / 20%, hybrid 6 / 6 / 16%, vectorless 0 / 0 / 6%, long context 0 / 2 / 8% (metrics / domain / novel) |
| 4.2 variation between runs | 27 questions where the filter variant and vector RAG without selection passed identical chunk lists; 3 verdicts differ. The progress report says "3 of 26": 26 is where the filter kept all 10 chunks, plus 1 question whose top 10 had fewer than 10 chunks. **Needs a cell in Section 13.4** |
| 4.2.1 | right when it answers: 82.8% (106 of 128) with selection, 77.2% (105 of 136) without |
| 4.2.2 | hybrid vs unfiltered top 10: wrong answers 25 vs 31, refusals 14 vs 14; correct by type 44 vs 42 / 34 vs 30 / 33 vs 33 (metrics / domain / novel). **Needs a cell** |
| 4.3 novel-generated exposure | median ROUGE-2 on novel questions: vector RAG correct 0.93, wrong 0.17, declined 0.56; hybrid correct 0.97, wrong 0.99, declined 0.68 (notebook cell 34 has these by type) |
| 4.3.2 hybrid gains | 14 gained (9 were incorrect, 5 declined): median ROUGE-2 0.65 narrowed vs 0.46 unfiltered; 8 lost: 0.77 vs 0.90. **Needs a cell in Section 13.5** |
| 4.4 latency | median / p95 / max (s): vector RAG 5.1 / 7.8 / 9.2; vectorless 3.6 / 4.2 / 4.2; hybrid 4.6 / 28.8 / 49.3; long context 30.3 / 134.6 / 168.3 (notebook cell 48) |
| 4.4 stage medians | vector: expansion 1.30, embedding 0.04, search 0.01, rerank 0.66, selection 1.22, generation 1.56; vectorless: navigation 0.46, generation 0.68, + borrowed expansion 2.46; hybrid: search 0.04, rerank 0.36, generation 1.26, + borrowed navigation 0.46, expansion 2.46, embedding 0.05; long context: generation 30.29. **Needs a cell** |
| 4.4 setup time | `vectorless_rag_indexing_latency.jsonl`, last Flash run per filing: median 188 s, max 2,512 s, total 671 min (about 11 h) for 84 filings. **Not used anywhere in the notebook yet; needs a cell.** Vector RAG embedding time: not recorded (ask / add if known) |
| 4.5 context size | generation input tokens median / mean / max: vector 801 / 1,041 / 5,426; hybrid 4,399 / 4,307 / 5,203; vectorless 33,127 / 37,686 / 154,434; long context 70,165 / 78,681 / 292,263 (notebook cell 40) |
| 4.5 cost by step ($/1k) | vector: expansion 0.10, selection 1.29, generation 0.51; vectorless: expansion 0.09, navigation 3.03, generation 9.68; hybrid: expansions 0.19, navigation 3.03, generation 1.33; long context: generation 19.96. Totals are in cell 48; **the split per step needs a cell** |
| 4.5 variant costs | without selection: 0.10 + generation 1.36 = 1.46 $/1k (median 4,527 tokens); filter: 0.10 + 0.09 + 3.03 + generation 1.01 = about 4.23 $/1k (median 3,100 tokens, 31% fewer). **Needs a cell** |
| 4.5 indexing and break-even | 14.65 USD (7.41 Flash over 48 docs + 1.69 fallback + 36 x 0.154 estimated); vectorless beats long context after ~2,046 queries, hybrid after ~951 (notebook cells 48 and 50) |
| 4.6 | Pareto: long context dominated; others optimal (cell 48) |
| 4.4 (simplified) | median / 90th percentile (s), borrowed steps added as in cell 48: vector RAG 5.1 / 6.3; vectorless 3.6 / 4.1; hybrid 4.6 / 5.2; long context 30.3 / 119.3. **The 90th percentile is not in the notebook (it uses p95); needs a cell** |
| 4.5 (simplified) | mean cost per query (USD) = cell 48 cost per 1k / 1000: 0.0019 / 0.0045 / 0.0128 / 0.0200; total for 150 queries: 0.29 / 0.68 / 1.92 / 2.99; mean generation input tokens 1,041 / 4,307 / 37,686 / 78,681 (cell 40); without selection agent 0.0015 per query |
| 4.3 Adobe example | page 57 (cash flow statement) is split into chunks `ADOBE_2022_10K__p0056_c00` (operating activities, 7,838) and `..._c01` (investing activities, capex 442 / 348); vector RAG and the hybrid both retrieved c00 but not c01 |

## Note on the notebook text (Section 8, point 4 under "Why the chunk-based pipelines struggle")

It says that for Adobe's free cash flow conversion "both pipelines had the capex line ... and both
refused". That is not quite right: their chunks contain the phrase "purchases of property and
equipment" only in a narrative paragraph on page 47, without the figures. The capex figures are in
chunk `p0056_c01`, which neither pipeline retrieved. The refusal was therefore correct. Worth
rewording that point before the notebook is shown as evidence.
