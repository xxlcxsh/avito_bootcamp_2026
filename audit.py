"""Inspect the retrieval data without using benchmark labels.

Run: python audit.py

The report guides validation design and prevents a misleading score from a
row-level train/validation split. All IDs remain strings throughout.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
QUERY_COLUMNS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]


def normalize_text(value: object) -> str:
    """Normalize spelling for overlap analysis, preserving numbers and words."""
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKC", str(value)).casefold().replace("ё", "е")
    return " ".join(value.split())


def pct(part: int, whole: int) -> str:
    return f"{part:,} / {whole:,} ({part / whole:.1%})" if whole else "n/a"


def main() -> None:
    train = pd.read_parquet(
        ROOT / "train.parquet",
        columns=QUERY_COLUMNS + ["item_id", "item_location_id", "item_category_id", "item_microcat_id"],
    )
    benchmark_queries = pd.read_parquet(ROOT / "benchmark_queries.parquet")
    benchmark_items = pd.read_parquet(
        ROOT / "benchmark_items.parquet", columns=["item_id", "item_category_id"]
    )

    # A query group is an observable feature tuple, not necessarily a unique
    # search session. We use distinct clicked items as its known positives.
    train["query_group"] = train.groupby(QUERY_COLUMNS, dropna=False, sort=False).ngroup()
    train_queries = train.drop_duplicates("query_group")[QUERY_COLUMNS + ["query_group"]].copy()
    counts = train.groupby("query_group")["item_id"].nunique()

    train_queries["normalized_text"] = train_queries.search_query.map(normalize_text)
    benchmark_queries["normalized_text"] = benchmark_queries.search_query.map(normalize_text)
    known_texts = set(train_queries.normalized_text)
    known_exact = train_queries[QUERY_COLUMNS].drop_duplicates()
    exact_match = benchmark_queries[QUERY_COLUMNS].merge(
        known_exact.assign(seen_exact=True), on=QUERY_COLUMNS, how="left"
    ).seen_exact.fillna(False).astype(bool)

    train_item_ids = set(train.item_id.astype(str))
    benchmark_item_ids = set(benchmark_items.item_id.astype(str))
    text_match = benchmark_queries.normalized_text.isin(known_texts)

    print("DATA")
    print(f"train rows: {len(train):,}")
    print(f"train query groups: {len(train_queries):,}")
    print(f"train unique normalized texts: {len(known_texts):,}")
    print(f"train unique items: {len(train_item_ids):,}")
    print(f"benchmark queries: {len(benchmark_queries):,}")
    print(f"benchmark items: {len(benchmark_item_ids):,}")
    print()

    print("BENCHMARK OVERLAP")
    print("seen normalized text:", pct(int(text_match.sum()), len(benchmark_queries)))
    print("seen exact feature tuple:", pct(int(exact_match.sum()), len(benchmark_queries)))
    print("seen benchmark items:", pct(len(train_item_ids & benchmark_item_ids), len(benchmark_item_ids)))
    print("train items available in benchmark:", pct(len(train_item_ids & benchmark_item_ids), len(train_item_ids)))
    print()

    print("TRAIN LABEL STRUCTURE")
    print("positive items per query group percentiles:", counts.quantile([0, .5, .9, .99, 1]).to_dict())
    print("query groups with >50 known positives:", int((counts > 50).sum()))
    print("duplicate positive pairs:", int(len(train) - len(train[["query_group", "item_id"]].drop_duplicates())))
    print("positive item location matches search:", f"{(train.search_location_id == train.item_location_id).mean():.1%}")
    print("distinct search categories:", train.search_category.nunique())
    print("distinct item categories:", train.item_category_id.nunique())
    print("distinct item microcategories:", train.item_microcat_id.nunique())

    # Random query-group splitting tests generalization across feature tuples.
    # Cold-text splitting is stricter and exposes a different failure mode.
    rng = np.random.default_rng(67)
    is_valid = rng.random(len(train_queries)) < 0.2
    train_part = train_queries.loc[~is_valid]
    valid_part = train_queries.loc[is_valid]
    warm_valid = valid_part.normalized_text.isin(set(train_part.normalized_text))
    print()
    print("VALIDATION DESIGN DIAGNOSTIC")
    print("random query-group holdout:", len(valid_part))
    print("warm-text fraction in holdout:", f"{warm_valid.mean():.1%}")
    print("cold-text fraction in holdout:", f"{(~warm_valid).mean():.1%}")
    print("benchmark warm-text fraction:", f"{text_match.mean():.1%}")
    print("Use separate warm/cold scores; raw random-split mean has a different mix.")


if __name__ == "__main__":
    main()
