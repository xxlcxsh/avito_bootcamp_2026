# Avito service retrieval

`solution.py` recreates `answer.csv` for the task in [TEST.md](TEST.md). The file is UTF-8 CSV with exactly `query_id,answer`; `answer` contains up to 50 space-separated `item_id` values. The generated file is included in this repository and in `dist/avito-retrieval.zip`.

## Architecture

1. Candidate generation: three TF-IDF channels (title character n-grams, title word n-grams, description word n-grams) plus frozen `intfloat/multilingual-e5-base` in local INT8 ONNX inference.
2. Geographic retrieval: candidates are searched globally and within the exact or train-inferred item-location scope.
3. Reranking: `HistGradientBoostingClassifier` selects the final 50 from the union using channel ranks, E5 cosine scores, geo history, title overlap, rating/price/contact signals, and six parameter-match signals (token and number overlap between query and item filters).

E5 uses `query:` / `passage:` prefixes. A passage contains title (<=40 tokens) and item parameters (<=56); the description is handled by its separate TF-IDF channel. The model is pretrained only; no task fine-tuning or external inference API is used.

## Validation

Query groups are split by all `search_*` fields. The accepted parameter features achieved Recall@50 **0.781821** and **0.786416** on two separate, previously unused benchmark-shaped validation sets of 2,452 groups. These are local measurements, not the platform score.

## Run

```bash
python -m pip install -r requirements.txt
python dense_retrieval.py --download  # first clone only
python -u solution.py
```

The first run needs Python 3.12, an Intel i7-12700H / 16 GB RAM-class machine, and downloads the open-source E5 weights. On the measured i7-12700H with 16 GB RAM:

| Stage | Time |
| --- | ---: |
| E5 train corpus, 344,825 items | ~88 min, 65.3 texts/s |
| E5 benchmark service corpus, 187,336 items | ~45 min, 69.8 texts/s |
| Train TF-IDF retrieval, candidates and HGB with ready E5 caches | 278.8 s |
| Full repeat run with ready E5 caches | 525.2 s (8.75 min) |

E5 caches are content-addressed and resumable in `cache/`; they are not committed. A repeat run reuses them and does not re-encode documents. Rebuild occurs only if the source data, model revision, ONNX file, or text format changes.

## Delivery

Use a public **GitHub repository** as the code link required by TEST.md: it is easier for a reviewer to inspect, run and cite. The generated ZIP is a backup, not the primary link. Upload only the root `answer.csv` to the competition form; do not upload the repository archive instead.

Current `answer.csv` SHA-256: `615E5AE2A94E7FEA34908F571FDE2E19B70D1DF90E36B9B274165C82E56AAFED`.
