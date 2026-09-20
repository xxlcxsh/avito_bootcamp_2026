"""Frozen multilingual E5-base retrieval with resumable CPU embedding caches.

The upstream INT8 ONNX export changes inference precision, not the pretrained
weights through task training. All inference is local after --download.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from audit import normalize_text

ROOT = Path(__file__).resolve().parent
MODEL_ID = 'intfloat/multilingual-e5-base'
REVISION = 'd128750597153bb5987e10b1c3493a34e5a4502a'
MODEL_DIR = ROOT / 'models' / 'multilingual-e5-base'
MODEL_FILE = 'onnx/model_qint8_avx512_vnni.onnx'
FORMAT_VERSION = 'title40-params56-description88-query48-filter24-v1'


def download_model():
    for name in (MODEL_FILE, 'tokenizer.json', 'config.json'):
        target = MODEL_DIR / name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}'
        temporary = target.with_suffix(target.suffix + '.part')
        print(f'Downloading {name}', flush=True)
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(target)


class E5Encoder:
    def __init__(self, threads=6, description_tokens=88):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(MODEL_DIR / 'tokenizer.json'))
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(MODEL_DIR / MODEL_FILE), options,
                                           providers=['CPUExecutionProvider'])
        self.input_names = {x.name for x in self.session.get_inputs()}
        self.separator = self.tokenizer.encode(' | ', add_special_tokens=False).ids
        self.description_tokens = description_tokens

    def tokens(self, fields, kind):
        budgets = (40, 56, self.description_tokens) if kind == 'passage' else (48, 24)
        result = [0] + self.tokenizer.encode(kind + ': ', add_special_tokens=False).ids
        for i, (field, budget) in enumerate(zip(fields, budgets)):
            if i:
                result.extend(self.separator)
            text = normalize_text(field)
            if text:
                result.extend(self.tokenizer.encode(text, add_special_tokens=False).ids[:budget])
        result.append(2)
        assert len(result) <= 512
        return result

    def encode_tokens(self, rows):
        width = max(map(len, rows))
        ids = np.full((len(rows), width), 1, dtype=np.int64)
        mask = np.zeros_like(ids)
        for i, row in enumerate(rows):
            ids[i, :len(row)] = row
            mask[i, :len(row)] = 1
        feed = {'input_ids': ids, 'attention_mask': mask,
                'token_type_ids': np.zeros_like(ids)}
        hidden = self.session.run(None, {k: v for k, v in feed.items() if k in self.input_names})[0]
        if hidden.ndim != 3:
            raise ValueError(f'Expected token embeddings, got {hidden.shape}')
        masked = hidden * mask[..., None]
        pooled = masked.sum(axis=1) / mask.sum(axis=1, keepdims=True)
        pooled = pooled.astype(np.float32)
        return pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)


def embedding_cache(frame, kind, stem, batch_size=16, threads=6, description_tokens=0):
    """Content-addressed, crash-resumable memmap. Cache includes model and format."""
    columns = (['item_title_raw', 'item_infm_params_text', 'item_description_raw']
               if kind == 'passage' else ['search_query', 'search_infm_params_text'])
    data = frame[columns].fillna('').astype(str)
    digest = hashlib.sha256(pd.util.hash_pandas_object(data, index=False).values.tobytes()
                            + (REVISION + MODEL_FILE + FORMAT_VERSION
                               + str(description_tokens)).encode()).hexdigest()
    path = ROOT / 'cache' / f'{stem}.npy'
    meta_path = path.with_suffix('.json')
    path.parent.mkdir(exist_ok=True)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get('digest') == digest and path.exists():
        done = meta['done']
        output = np.lib.format.open_memmap(path, mode='r+')
        if done == len(data):
            print(f'Cached {stem}: {done} embeddings', flush=True)
            return output
    else:
        done = 0
        output = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(data), 768))
    encoder = E5Encoder(threads, description_tokens)
    started, initial = time.perf_counter(), done
    last_log = started
    tuples = list(data.itertuples(index=False, name=None))
    while done < len(data):
        # Sort within checkpoint blocks to reduce padding, then restore order.
        end = min(done + 512, len(data))
        token_rows = [encoder.tokens(row, kind) for row in tuples[done:end]]
        order = sorted(range(len(token_rows)), key=lambda i: len(token_rows[i]))
        for offset in range(0, len(order), batch_size):
            selected = order[offset:offset + batch_size]
            output[np.asarray(selected) + done] = encoder.encode_tokens([token_rows[j] for j in selected])
        output.flush()
        done = end
        meta = {'digest': digest, 'done': done, 'rows': len(data), 'revision': REVISION,
                'model_file': MODEL_FILE, 'format': FORMAT_VERSION}
        temporary = meta_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(meta, indent=2), encoding='utf-8')
        temporary.replace(meta_path)
        now = time.perf_counter()
        if now - last_log >= 30 or done == len(data):
            speed = (done - initial) / (now - started)
            print(f'{stem}: {done}/{len(data)}, {speed:.1f} texts/s, '
                  f'ETA {(len(data)-done)/max(speed, .01)/60:.1f} min', flush=True)
            last_log = now
    return output


def topk_dense(queries, items, k=200, batch_size=32):
    ranks, similarities = [], []
    for start in range(0, len(queries), batch_size):
        scores = np.asarray(queries[start:start + batch_size]) @ np.asarray(items).T
        size = min(k, len(items))
        indices = np.argpartition(-scores, size - 1, axis=1)[:, :size]
        for row, selected in zip(scores, indices):
            selected = selected[np.lexsort((selected, -row[selected]))]
            ranks.append(selected.tolist())
            similarities.append(row[selected].tolist())
    return ranks, similarities


def scoped_dense(queries, items, scopes, k=200):
    """Exact dense retrieval inside the geographic scopes inferred from train."""
    locations = items.item_location_id.to_numpy()
    groups = {}
    for query_idx, scope in enumerate(scopes):
        if scope:
            groups.setdefault(tuple(scope), []).append(query_idx)
    rankings = [[] for _ in range(len(queries))]
    similarities = [[] for _ in range(len(queries))]
    for scope, query_rows in groups.items():
        item_rows = np.flatnonzero(np.isin(locations, scope))
        if not len(item_rows):
            continue
        item_vectors = np.stack(items.dense_vector.iloc[item_rows].to_numpy())
        for query_idx in query_rows:
            score = np.asarray(queries.dense_vector.iat[query_idx]) @ item_vectors.T
            size = min(k, len(item_rows))
            selected = np.argpartition(-score, size - 1)[:size]
            selected = selected[np.lexsort((item_rows[selected], -score[selected]))]
            rankings[query_idx] = item_rows[selected].tolist()
            similarities[query_idx] = score[selected].tolist()
    return rankings, similarities


def parameter_sets(queries, items):
    """Prepare cheap lexical filter signals without changing retrieval or E5."""
    def tokens(values):
        return [set(re.findall(r'\w+', normalize_text(value))) for value in values]

    def numbers(values):
        return [set(re.findall(r'\d+(?:[.,]\d+)?', normalize_text(value))) for value in values]

    return {
        'query_params': tokens(queries.search_infm_params_text),
        'item_params': tokens(items.item_infm_params_text),
        'query_numbers': numbers(queries.search_infm_params_text),
        'item_numbers': numbers(items.item_infm_params_text),
    }


def hybrid_features(query_idx, query_tokens, queries, items, item_tokens, channels, prior,
                    params=None):
    """Pool lexical+dense candidates and build selector features without score mixing."""
    row = queries.iloc[query_idx]
    global_rows = [part[0][query_idx] for part in channels]
    local_rows = [part[1][query_idx] for part in channels]
    candidates = list(dict.fromkeys(idx for ranking in local_rows for idx in ranking))
    seen = set(candidates)
    for ranking in global_rows:
        added = 0
        for idx in ranking:
            if idx not in seen and items.item_location_id.iat[idx] != row.search_location_id:
                candidates.append(idx); seen.add(idx); added += 1
            if added >= 50:
                break
    rank_maps = [{idx: rank for rank, idx in enumerate(ranking, 1)}
                 for ranking in local_rows + global_rows]
    score_maps = [{idx: score for idx, score in zip(ranking[query_idx], scores[query_idx])}
                  for ranking, scores in [(part[0], part[2]) for part in channels]
                  + [(part[1], part[3]) for part in channels]]
    locations = items.item_location_id.to_numpy(); ratings = items.item_rating.to_numpy()
    reviews = items.item_rating_reviews_count.to_numpy(); prices = items.price_numeric.to_numpy()
    phones = items.item_is_phone_hidden.to_numpy(); messages = items.item_is_message_forbidden.to_numpy()
    counts = prior.counts.get(int(row.search_location_id), {})
    total = prior.totals.get(int(row.search_location_id), 0)
    features = np.empty((len(candidates), 30), dtype=np.float32)
    query_params = params['query_params'][query_idx] if params else set()
    query_numbers = params['query_numbers'][query_idx] if params else set()
    for feature_idx, item_idx in enumerate(candidates):
        ranks = [1 / (60 + ranks[item_idx]) if item_idx in ranks else 0. for ranks in rank_maps]
        words = item_tokens[item_idx]; overlap = len(query_tokens & words)
        item_params = params['item_params'][item_idx] if params else set()
        item_numbers = params['item_numbers'][item_idx] if params else set()
        param_overlap = len(query_params & item_params)
        number_overlap = len(query_numbers & item_numbers)
        count = counts.get(int(locations[item_idx]), 0)
        dense_scores = [score_maps[len(channels) - 1].get(item_idx, 0.),
                        score_maps[2 * len(channels) - 1].get(item_idx, 0.)]
        features[feature_idx] = [
            *ranks, sum(ranks[:len(channels)]), sum(ranks[len(channels):]), *dense_scores,
            float(locations[item_idx] == row.search_location_id),
            overlap / max(len(query_tokens), 1), overlap / max(len(words), 1),
            np.nan_to_num(ratings[item_idx], nan=0.),
            np.log1p(max(np.nan_to_num(reviews[item_idx], nan=0.), 0.)),
            np.log1p(max(np.nan_to_num(prices[item_idx], nan=0.), 0.)),
            float(phones[item_idx]), float(messages[item_idx]),
            count / max(total, 1), np.log1p(count), np.log1p(total), float(count > 0),
            param_overlap / max(len(query_params), 1),
            param_overlap / max(len(item_params), 1),
            np.log1p(param_overlap), float(number_overlap > 0),
            number_overlap / max(len(query_numbers), 1), float(bool(query_params) and bool(item_params)),
        ]
    return candidates, features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--bench', action='store_true')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--description-tokens', type=int, default=88)
    args = parser.parse_args()
    if args.download:
        download_model()
    if args.bench:
        items = pd.read_parquet(ROOT / 'train.parquet', columns=[
            'item_id', 'item_title_raw', 'item_infm_params_text', 'item_description_raw'])
        items = items.drop_duplicates('item_id', keep='last').sample(256, random_state=67)
        encoder = E5Encoder(args.threads, args.description_tokens)
        rows = [encoder.tokens(row, 'passage') for row in items[[
            'item_title_raw', 'item_infm_params_text', 'item_description_raw']].itertuples(index=False, name=None)]
        encoder.encode_tokens(rows[:4])
        start = time.perf_counter()
        for i in range(0, len(rows), args.batch_size):
            values = encoder.encode_tokens(rows[i:i+args.batch_size])
            assert np.isfinite(values).all()
            assert np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5)
        elapsed = time.perf_counter() - start
        report = {'texts': len(rows), 'seconds': elapsed, 'texts_per_second': len(rows)/elapsed,
                  'estimated_train_minutes': 344825/len(rows)*elapsed/60,
                  'estimated_benchmark_minutes': 189212/len(rows)*elapsed/60,
                  'mean_tokens': float(np.mean(list(map(len, rows)))), 'threads': args.threads,
                  'batch_size': args.batch_size,
                  'description_tokens': args.description_tokens,
                  'model': MODEL_ID, 'revision': REVISION, 'precision': 'upstream INT8 ONNX'}
        (ROOT / 'dense_speed.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
