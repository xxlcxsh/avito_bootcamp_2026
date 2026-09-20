"""Shared retrieval primitives used by the reproducible solution.

The functions in this module contain no experiment reporting or tuning loops.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from audit import QUERY_COLUMNS, normalize_text

ROOT = Path(__file__).resolve().parent
SEED = 67
VALIDATION_SIZE = 2452
WARM_FRACTION = 916 / 2452


def make_validation(train: pd.DataFrame):
    """Fixed benchmark-shaped query groups for fitting the final selector."""
    train['query_group'] = train.groupby(QUERY_COLUMNS, dropna=False, sort=False).ngroup()
    queries = train.drop_duplicates('query_group')[QUERY_COLUMNS + ['query_group']].copy()
    queries['normalized_text'] = queries.search_query.map(normalize_text)
    rng = np.random.default_rng(SEED)
    held_out = rng.random(len(queries)) < .2
    history_groups = set(queries.loc[~held_out, 'query_group'])
    history_texts = set(queries.loc[~held_out, 'normalized_text'])
    candidates = queries.loc[held_out].copy()
    candidates['segment'] = np.where(candidates.normalized_text.isin(history_texts), 'warm', 'cold')
    parts = []
    for segment, size in [('warm', round(VALIDATION_SIZE * WARM_FRACTION)),
                          ('cold', VALIDATION_SIZE - round(VALIDATION_SIZE * WARM_FRACTION))]:
        parts.append(candidates.loc[candidates.segment == segment].sample(n=size, random_state=SEED))
    validation = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    positives = train.loc[train.query_group.isin(set(validation.query_group))].groupby(
        'query_group').item_id.agg(lambda s: set(s.astype(str))).to_dict()
    return validation, positives, history_groups


def fresh_validation(train: pd.DataFrame, prior_validation: pd.DataFrame, history_groups: set[int]):
    """Independent development groups excluding all earlier normalized texts."""
    queries = train.drop_duplicates('query_group')[['query_group', *QUERY_COLUMNS]].copy()
    queries['normalized_text'] = queries.search_query.map(normalize_text)
    known = set(queries.loc[queries.query_group.isin(history_groups), 'normalized_text'])
    remaining = queries.loc[~queries.query_group.isin(history_groups)
                            & ~queries.normalized_text.isin(set(prior_validation.normalized_text))].copy()
    remaining['segment'] = np.where(remaining.normalized_text.isin(known), 'warm', 'cold')
    parts = [remaining.loc[remaining.segment == segment].sample(n=size, random_state=20260919)
             for segment, size in [('warm', 916), ('cold', 1536)]]
    return pd.concat(parts).sample(frac=1, random_state=20260919).reset_index(drop=True)


def confirmation_validation(train: pd.DataFrame, prior_validations: list[pd.DataFrame],
                            history_groups: set[int], seed: int = 20260920):
    """An untouched benchmark-shaped split for accepting a new method."""
    queries = train.drop_duplicates('query_group')[['query_group', *QUERY_COLUMNS]].copy()
    queries['normalized_text'] = queries.search_query.map(normalize_text)
    known = set(queries.loc[queries.query_group.isin(history_groups), 'normalized_text'])
    seen_groups = set().union(*(set(frame.query_group) for frame in prior_validations))
    seen_texts = set().union(*(set(frame.normalized_text) for frame in prior_validations))
    remaining = queries.loc[~queries.query_group.isin(history_groups | seen_groups)
                            & ~queries.normalized_text.isin(seen_texts)].copy()
    remaining['segment'] = np.where(remaining.normalized_text.isin(known), 'warm', 'cold')
    parts = [remaining.loc[remaining.segment == segment].sample(n=size, random_state=seed)
             for segment, size in [('warm', 916), ('cold', 1536)]]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def retrieve(query_matrix, item_matrix, top_k):
    scores = sp_matmul_topn(query_matrix.tocsr(), item_matrix.T.tocsr(), top_n=top_k,
                            sort=True, n_threads=8)
    return [scores.indices[scores.indptr[i]:scores.indptr[i + 1]].tolist()
            for i in range(scores.shape[0])]


def build_matrix(texts, analyzer, ngrams, min_df):
    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngrams, min_df=min_df,
                                 sublinear_tf=True, dtype=np.float32, norm='l2')
    return vectorizer, vectorizer.fit_transform(texts)


def scoped_rankings(query_matrix, item_matrix, query_locations, item_locations, k=200):
    """Retrieve by exact location; inferred scopes are implemented separately."""
    query_groups, item_groups = defaultdict(list), defaultdict(list)
    for i, location in enumerate(query_locations):
        query_groups[int(location)].append(i)
    for i, location in enumerate(item_locations):
        item_groups[int(location)].append(i)
    output = [[] for _ in query_locations]
    for location, qrows in query_groups.items():
        irows = np.asarray(item_groups.get(location, []), dtype=np.int32)
        if not len(irows):
            continue
        rows = retrieve(query_matrix[qrows], item_matrix[irows], min(k, len(irows)))
        for qi, ranking in zip(qrows, rows):
            output[qi] = irows[ranking].tolist()
    return output


def candidate_features(query_idx, query_tokens, queries, items, item_tokens, channels,
                       global_per_channel=50):
    """Original lexical candidate pool and 16 v1 features, kept for v2 geography."""
    row = queries.iloc[query_idx]
    global_rows = [pair[0][query_idx] for pair in channels]
    local_rows = [pair[1][query_idx] for pair in channels]
    candidates = list(dict.fromkeys(idx for ranking in local_rows for idx in ranking))
    seen = set(candidates)
    for ranking in global_rows:
        added = 0
        for idx in ranking:
            if idx not in seen and items.item_location_id.iat[idx] != row.search_location_id:
                candidates.append(idx); seen.add(idx); added += 1
            if added >= global_per_channel:
                break
    rank_maps = [{idx: rank for rank, idx in enumerate(ranking, 1)}
                 for ranking in local_rows + global_rows]
    ratings = items.item_rating.to_numpy(); reviews = items.item_rating_reviews_count.to_numpy()
    prices = items.price_numeric.to_numpy(); locations = items.item_location_id.to_numpy()
    phones = items.item_is_phone_hidden.to_numpy(); messages = items.item_is_message_forbidden.to_numpy()
    features = np.empty((len(candidates), 16), dtype=np.float32)
    for ri, idx in enumerate(candidates):
        ranks = [1 / (60 + ranks[idx]) if idx in ranks else 0. for ranks in rank_maps]
        words = item_tokens[idx]; overlap = len(query_tokens & words)
        features[ri] = [*ranks, sum(ranks[:3]), sum(ranks[3:]),
                        float(locations[idx] == row.search_location_id),
                        overlap / max(len(query_tokens), 1), overlap / max(len(words), 1),
                        np.nan_to_num(ratings[idx], nan=0.),
                        np.log1p(max(np.nan_to_num(reviews[idx], nan=0.), 0.)),
                        np.log1p(max(np.nan_to_num(prices[idx], nan=0.), 0.)),
                        float(phones[idx]), float(messages[idx])]
    return candidates, features


def title_token_sets(items):
    return [set(re.findall(r'\w+', normalize_text(value))) for value in items.item_title_raw]
