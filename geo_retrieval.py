"""Learn geography from train clicks without assuming that location IDs match.

Search locations absent from the item corpus may describe larger areas. Their
retrieval scope is inferred from OTHER query groups, never evaluation labels.
No query IDs or hand-written location correspondences are used.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd

from audit import normalize_text
from retrieval_core import build_matrix, candidate_features, retrieve


class LocationPrior:
    def __init__(self, history: pd.DataFrame):
        # Count each observed query-group/location pair once, so repeated clicks
        # and groups with many ads in one city cannot dominate the geography.
        unique = history.drop_duplicates(['query_group', 'item_location_id'])
        counts = unique.groupby(['search_location_id', 'item_location_id']).size()
        self.counts = defaultdict(dict)
        for (search_loc, item_loc), count in counts.items():
            self.counts[int(search_loc)][int(item_loc)] = int(count)
        self.totals = {loc: sum(counts.values()) for loc, counts in self.counts.items()}

    def scopes(self, queries: pd.DataFrame, items: pd.DataFrame) -> list[tuple[int, ...]]:
        available = set(items.item_location_id.astype(int))
        scopes = []
        for value in queries.search_location_id:
            loc = int(value)
            if loc in available:
                scopes.append((loc,))
                continue
            counts = self.counts.get(loc, {})
            if self.totals.get(loc, 0) < 5:
                scopes.append(())
                continue
            ranked = sorted(((target, n) for target, n in counts.items()
                             if target in available), key=lambda pair: (-pair[1], pair[0]))
            chosen, mass = [], 0
            for target, count in ranked[:8]:
                chosen.append(target)
                mass += count
                if mass >= .9 * self.totals[loc]:
                    break
            scopes.append(tuple(sorted(chosen)))
        return scopes


def scoped_rankings(query_matrix, item_matrix, scopes, item_locations, k=200):
    """Retrieve within a union of inferred locations; batch identical scopes."""
    item_groups = defaultdict(list)
    for i, loc in enumerate(item_locations):
        item_groups[int(loc)].append(i)
    query_groups = defaultdict(list)
    for i, scope in enumerate(scopes):
        if scope:
            query_groups[scope].append(i)
    output = [[] for _ in scopes]
    for scope, query_rows in query_groups.items():
        item_rows = np.asarray(sorted(i for loc in scope for i in item_groups[loc]), dtype=np.int32)
        rows = retrieve(query_matrix[query_rows], item_matrix[item_rows], min(k, len(item_rows)))
        for qi, ranking in zip(query_rows, rows):
            output[qi] = item_rows[ranking].tolist()
    return output


def build_geo_channels(items, queries, prior, include_baseline=False):
    """Use the same three TF-IDF indexes as v1, adding learned search scopes."""
    query_texts = [normalize_text(value) for value in queries.search_query]
    titles = [normalize_text(value) for value in items.item_title_raw]
    descriptions = [normalize_text(value)[:500] for value in items.item_description_raw]
    item_tokens = [set(re.findall(r'\w+', value)) for value in titles]
    query_tokens = [set(re.findall(r'\w+', value)) for value in query_texts]
    item_locations = items.item_location_id.to_numpy()
    scopes = prior.scopes(queries, items)
    inferred = sum(bool(scope) and scope != (int(loc),)
                   for scope, loc in zip(scopes, queries.search_location_id))
    print(f'Queries with inferred geography: {inferred}/{len(queries)}', flush=True)
    original, enhanced = [], []
    for name, texts, analyzer, ngrams, min_df in (
        ('title char', titles, 'char_wb', (3, 4), 5),
        ('title word', titles, 'word', (1, 2), 2),
        ('description word', descriptions, 'word', (1, 2), 3),
    ):
        vectorizer, matrix = build_matrix(texts, analyzer, ngrams, min_df)
        query_matrix = vectorizer.transform(query_texts)
        global_rows = retrieve(query_matrix, matrix, 1000)
        scoped_rows = scoped_rankings(query_matrix, matrix, scopes, item_locations)
        enhanced.append((global_rows, scoped_rows))
        if include_baseline:
            # Scopes of queries with an exact location are unchanged. Reuse
            # those rankings; v1 has no local candidates for every other query.
            original.append((global_rows, [scoped_rows[i] if scopes[i] == (int(loc),) else []
                                           for i, loc in enumerate(queries.search_location_id)]))
        print(f'Ready geo: {name}', flush=True)
    return enhanced, original, item_tokens, query_tokens


def geo_features(i, query_tokens, queries, items, item_tokens, channels, prior):
    candidates, base = candidate_features(i, query_tokens, queries, items, item_tokens, channels)
    loc = int(queries.search_location_id.iat[i])
    counts = prior.counts.get(loc, {})
    total = prior.totals.get(loc, 0)
    locations = items.item_location_id.to_numpy()[candidates]
    count = np.asarray([counts.get(int(target), 0) for target in locations], dtype=np.float32)
    # The support features let the model distinguish strong evidence from a
    # location relation observed only once. A small denominator prevents 0/0.
    extra = np.column_stack([
        count / max(total, 1), np.log1p(count),
        np.full(len(count), np.log1p(total), dtype=np.float32),
        (count > 0).astype(np.float32),
    ])
    return candidates, np.column_stack([base, extra]).astype(np.float32)
