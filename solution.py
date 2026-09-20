"""Reproduce answer.csv for the Avito service retrieval task.

Run from this directory: python -u solution.py

The pipeline uses no external API. It trains a compact candidate selector on
held-out train query groups, retrieves candidates from benchmark_items, and
writes exactly the required two-column CSV.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from audit import QUERY_COLUMNS, normalize_text
from retrieval_core import ROOT, make_validation
from geo_retrieval import LocationPrior, build_geo_channels, geo_features
from retrieval_core import candidate_features
from dense_retrieval import embedding_cache, hybrid_features, parameter_sets, scoped_dense, topk_dense


ITEM_COLUMNS = [
    "item_id", "item_title_raw", "item_description_raw", "item_location_id",
    "item_rating", "item_rating_reviews_count", "item_price",
    "item_is_phone_hidden", "item_is_message_forbidden", "item_category_id",
]
ANSWER_PATH = ROOT / "answer.csv"


def prepare_items(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ITEM_COLUMNS + (["item_infm_params_text"] if "item_infm_params_text" in frame else [])
    items = frame[columns].drop_duplicates("item_id", keep="last").reset_index(drop=True)
    items["item_id"] = items.item_id.astype(str)
    items["price_numeric"] = pd.to_numeric(items.item_price, errors="coerce")
    return items


def build_channels(
    items: pd.DataFrame,
    queries: pd.DataFrame,
) -> tuple[list[tuple[list[list[int]], list[list[int]]]], list[set[str]]]:
    """Build three lexical indexes and query them globally and by location."""
    item_locations = items.item_location_id.to_numpy()
    query_locations = queries.search_location_id.to_numpy()
    query_texts = [normalize_text(value) for value in queries.search_query]
    title_texts = [normalize_text(value) for value in items.item_title_raw]
    descriptions = [normalize_text(value)[:500] for value in items.item_description_raw]
    item_tokens = [set(re.findall(r"\w+", value)) for value in title_texts]
    query_tokens = [set(re.findall(r"\w+", value)) for value in query_texts]

    channels = []
    for name, corpus_texts, analyzer, ngrams, min_df in (
        ("title char", title_texts, "char_wb", (3, 4), 5),
        ("title word", title_texts, "word", (1, 2), 2),
        ("description word", descriptions, "word", (1, 2), 3),
    ):
        vectorizer, item_matrix = build_matrix(corpus_texts, analyzer, ngrams, min_df)
        query_matrix = vectorizer.transform(query_texts)
        global_ranking = retrieve(query_matrix, item_matrix, 1000)
        local_ranking = location_rankings(
            query_matrix, item_matrix, query_locations, item_locations, 200
        )
        channels.append((global_ranking, local_ranking))
        print(f"Ready: {name}", flush=True)
    return channels, (item_tokens, query_tokens)


def train_selector(
    items: pd.DataFrame,
    validation: pd.DataFrame,
    positives: dict[int, set[str]],
    channels: list[tuple[list[list[int]], list[list[int]]]],
    item_tokens: list[set[str]],
    query_tokens: list[set[str]],
    prior: LocationPrior | None = None, dense: bool = False, params=None,
) -> HistGradientBoostingClassifier:
    item_ids = items.item_id.to_numpy()
    rng = np.random.default_rng(67)
    features, labels = [], []
    for query_idx in range(len(validation)):
        if dense:
            candidates, query_features = hybrid_features(
                query_idx, query_tokens[query_idx], validation, items, item_tokens, channels, prior, params
            )
        elif prior is None:
            candidates, query_features = candidate_features(
                query_idx, query_tokens[query_idx], validation, items, item_tokens, channels
            )
        else:
            candidates, query_features = geo_features(
                query_idx, query_tokens[query_idx], validation, items, item_tokens, channels, prior
            )
        relevant = positives[validation.query_group.iat[query_idx]]
        query_labels = np.asarray([item_ids[idx] in relevant for idx in candidates], dtype=np.int8)
        positive_rows = np.flatnonzero(query_labels)
        negative_rows = np.flatnonzero(query_labels == 0)
        sampled_negative = rng.choice(
            negative_rows, size=min(150, len(negative_rows)), replace=False
        )
        selected = np.concatenate([positive_rows, sampled_negative])
        features.append(query_features[selected])
        labels.append(query_labels[selected])
    X = np.vstack(features)
    y = np.concatenate(labels)
    print(f"Selector pairs: {len(y):,}; known positives: {int(y.sum()):,}", flush=True)
    model = HistGradientBoostingClassifier(
        max_iter=160, learning_rate=0.06, max_leaf_nodes=15,
        min_samples_leaf=30, l2_regularization=5, random_state=67,
        class_weight="balanced",
    )
    model.fit(X, y)
    return model


def predict(
    model: HistGradientBoostingClassifier,
    items: pd.DataFrame,
    queries: pd.DataFrame,
    channels: list[tuple[list[list[int]], list[list[int]]]],
    item_tokens: list[set[str]],
    query_tokens: list[set[str]],
    prior: LocationPrior | None = None, dense: bool = False, params=None,
) -> list[list[str]]:
    item_ids = items.item_id.to_numpy()
    predictions = []
    for query_idx in range(len(queries)):
        if dense:
            candidates, features = hybrid_features(
                query_idx, query_tokens[query_idx], queries, items, item_tokens, channels, prior, params
            )
        elif prior is None:
            candidates, features = candidate_features(
                query_idx, query_tokens[query_idx], queries, items, item_tokens, channels
            )
        else:
            candidates, features = geo_features(
                query_idx, query_tokens[query_idx], queries, items, item_tokens, channels, prior
            )
        if candidates:
            scores = model.predict_proba(features)[:, 1]
            chosen = [candidates[idx] for idx in np.argsort(-scores, kind="stable")[:50]]
        else:
            chosen = []
        seen = set(chosen)
        # A rare query can have fewer than 50 lexical matches. Fill available
        # slots deterministically using the global retrieval channels.
        if len(chosen) < 50:
            for global_ranking, *_ in channels:
                for idx in global_ranking[query_idx]:
                    if idx not in seen:
                        chosen.append(idx)
                        seen.add(idx)
                    if len(chosen) == 50:
                        break
                if len(chosen) == 50:
                    break
        if len(chosen) < 50:
            for idx in range(len(items)):
                if idx not in seen:
                    chosen.append(idx)
                    seen.add(idx)
                if len(chosen) == 50:
                    break
        predictions.append([item_ids[idx] for idx in chosen])
    return predictions


def validate_answer(
    path: Path, benchmark_queries: pd.DataFrame, benchmark_items: pd.DataFrame
) -> None:
    answer = pd.read_csv(path, dtype=str, keep_default_na=False)
    expected_queries = set(benchmark_queries.query_id.astype(str))
    valid_items = set(benchmark_items.item_id.astype(str))
    assert list(answer.columns) == ["query_id", "answer"]
    assert len(answer) == len(expected_queries)
    assert answer.query_id.is_unique
    assert set(answer.query_id) == expected_queries
    assert answer.query_id.map(lambda value: bool(re.fullmatch(r"[A-Za-z0-9]{16}", value))).all()
    for value in answer.answer:
        ids = value.split(" ") if value else []
        assert len(ids) <= 50
        assert len(ids) == len(set(ids))
        assert all(re.fullmatch(r"[0-9a-f]{16}", item_id) for item_id in ids)
        assert all(item_id in valid_items for item_id in ids)
    print(f"Validated {len(answer):,} answer rows", flush=True)


def main() -> None:
    started = time.perf_counter()
    train = pd.read_parquet(ROOT / "train.parquet", columns=QUERY_COLUMNS + ITEM_COLUMNS + ["item_infm_params_text"])
    validation, positives, history_groups = make_validation(train)
    # Geographic features for selector training must exclude its own labels.
    # The original 80% history split is disjoint from every selector query.
    train_prior = LocationPrior(train.loc[train.query_group.isin(history_groups)])
    # At benchmark inference all train labels are legitimately available.
    benchmark_prior = LocationPrior(train)
    train_items = prepare_items(train)
    train_items["dense_vector"] = list(embedding_cache(train_items, "passage", "train_e5_title_params"))
    validation["dense_vector"] = list(embedding_cache(validation, "query", "validation_e5_queries"))
    train_channels, _, train_item_tokens, train_query_tokens = build_geo_channels(
        train_items, validation, train_prior
    )
    train_global_dense, train_global_scores = topk_dense(
        np.asarray(list(validation.dense_vector)), np.asarray(list(train_items.dense_vector)), k=1000)
    train_local_dense, train_local_scores = scoped_dense(validation, train_items,
                                                          train_prior.scopes(validation, train_items), 200)
    train_channels = [(global_rows, local_rows, [[] for _ in range(len(validation))],
                       [[] for _ in range(len(validation))])
                      for global_rows, local_rows in train_channels]
    train_channels.append((train_global_dense, train_local_dense, train_global_scores, train_local_scores))
    train_params = parameter_sets(validation, train_items)
    model = train_selector(
        train_items, validation, positives, train_channels,
        train_item_tokens, train_query_tokens, prior=train_prior,
        dense=True, params=train_params,
    )
    print(f"Selector trained in {time.perf_counter() - started:.1f}s", flush=True)
    del train_channels, train_item_tokens, train_items, train, train_prior

    benchmark_queries = pd.read_parquet(ROOT / "benchmark_queries.parquet")
    benchmark_items = pd.read_parquet(ROOT / "benchmark_items.parquet")
    # Every observed positive in train belongs to the service category 114.
    # Restricting the benchmark corpus removes a small set of category decoys.
    service_items = prepare_items(benchmark_items.loc[benchmark_items.item_category_id == 114])
    service_items["dense_vector"] = list(embedding_cache(service_items, "passage", "benchmark_e5_title_params"))
    benchmark_queries["dense_vector"] = list(embedding_cache(benchmark_queries, "query", "benchmark_e5_queries"))
    bench_channels, _, bench_item_tokens, bench_query_tokens = build_geo_channels(
        service_items, benchmark_queries, benchmark_prior
    )
    bench_global_dense, bench_global_scores = topk_dense(
        np.asarray(list(benchmark_queries.dense_vector)), np.asarray(list(service_items.dense_vector)), k=1000)
    bench_local_dense, bench_local_scores = scoped_dense(benchmark_queries, service_items,
                                                          benchmark_prior.scopes(benchmark_queries, service_items), 200)
    bench_channels = [(global_rows, local_rows, [[] for _ in range(len(benchmark_queries))],
                       [[] for _ in range(len(benchmark_queries))])
                      for global_rows, local_rows in bench_channels]
    bench_channels.append((bench_global_dense, bench_local_dense, bench_global_scores, bench_local_scores))
    benchmark_params = parameter_sets(benchmark_queries, service_items)
    predictions = predict(
        model, service_items, benchmark_queries, bench_channels,
        bench_item_tokens, bench_query_tokens, prior=benchmark_prior,
        dense=True, params=benchmark_params,
    )
    answer = pd.DataFrame({
        "query_id": benchmark_queries.query_id.astype(str),
        "answer": [" ".join(ids) for ids in predictions],
    })
    answer.to_csv(ANSWER_PATH, index=False, encoding="utf-8")
    validate_answer(ANSWER_PATH, benchmark_queries, benchmark_items)
    print(f"Saved {ANSWER_PATH}", flush=True)
    print(f"Elapsed: {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
