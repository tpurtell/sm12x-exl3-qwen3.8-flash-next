#!/usr/bin/env python3
"""Compare matched synthetic PLE runs without treating SSE chunks as tokens."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def index_cases(report):
    indexed = {}
    for case in report['cases']:
        key = (case['case'], case['repeat'])
        if key in indexed:
            raise ValueError(f'duplicate case/repeat: {key}')
        if not case.get('valid'):
            raise ValueError(f'invalid case: {key}: {case.get("invalid_reason")}')
        requests = {}
        for request in case['requests']:
            label = request['label']
            if label in requests:
                raise ValueError(f'duplicate request label: {key}/{label}')
            if request.get('status') != 'completed' or request.get('error') or not request.get('done_received'):
                raise ValueError(f'incomplete request: {key}/{label}')
            usage = request.get('usage') or {}
            for field in ('prompt_tokens', 'completion_tokens'):
                if not isinstance(usage.get(field), int) or isinstance(usage[field], bool) or usage[field] <= 0:
                    raise ValueError(f'missing/invalid usage.{field}: {key}/{label}')
            if request.get('wall_s', 0) <= 0:
                raise ValueError(f'invalid wall time: {key}/{label}')
            requests[label] = request
        expected = {'c1': {'decode-0'}, 'c2': {'decode-0', 'decode-1'},
                    'mixed': {'ongoing-decode', 'long-prefill'}}.get(case['case'])
        if expected is None or set(requests) != expected:
            raise ValueError(f'unexpected request labels: {key}: {set(requests)}')
        indexed[key] = requests
    if not indexed:
        raise ValueError('no benchmark cases')
    return indexed


def compare(baseline, candidate):
    if baseline.get('schema_version') != 1 or candidate.get('schema_version') != 1:
        raise ValueError('only benchmark schema_version 1 is supported')
    for key in ('model', 'run_id', 'seed', 'short_words', 'long_words', 'output_tokens',
                'thinking', 'mixed_delay', 'timeout'):
        if key not in baseline['config'] or key not in candidate['config']:
            raise ValueError(f'missing workload config: {key}')
        if baseline['config'][key] != candidate['config'][key]:
            raise ValueError(f'workload config mismatch: {key}')
    before = index_cases(baseline)
    after = index_cases(candidate)
    if before.keys() != after.keys():
        raise ValueError('case/repeat sets differ')
    rows = []
    for key, requests in before.items():
        for label, old in requests.items():
            new = after[key][label]
            identity = f'{key[0]} repeat={key[1]} {label}'
            if not old.get('prompt_sha256') or old['prompt_sha256'] != new.get('prompt_sha256'):
                raise ValueError(f'prompt hash mismatch: {identity}')
            for field in ('prompt_tokens', 'completion_tokens'):
                if old['usage'][field] != new['usage'][field]:
                    raise ValueError(f'usage.{field} mismatch: {identity}')
            expected = baseline['config']['output_tokens']
            if old['usage']['completion_tokens'] != expected:
                raise ValueError(f'completion length {old["usage"]["completion_tokens"]} != requested {expected}: {identity}')
            if old.get('finish_reason') != new.get('finish_reason') or old.get('finish_reason') != 'length':
                raise ValueError(f'finish reason is not matched length completion: {identity}')
            metrics = {'wall_s': (old['wall_s'], new['wall_s']),
                       'max_gap_s': (old.get('max_inter_chunk_gap_s'), new.get('max_inter_chunk_gap_s')),
                       'p95_gap_s': (old.get('p95_inter_chunk_gap_s'), new.get('p95_inter_chunk_gap_s')),
                       'ttft_s': (old.get('ttft_s'), new.get('ttft_s'))}
            if key[0] in ('c1', 'c2'):
                metrics['usage_completion_tok_per_wall_s'] = tuple(r['usage']['completion_tokens'] / r['wall_s'] for r in (old, new))
            rows.append({'case': key[0], 'repeat': key[1], 'request': label,
                         'prompt_tokens': old['usage']['prompt_tokens'],
                         'completion_tokens': old['usage']['completion_tokens'], 'metrics': metrics})
    return rows


def number(value):
    return 'unavailable' if value is None else f'{value:.6f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--json', action='store_true', help='output validated comparisons as JSON')
    args = parser.parse_args()
    try:
        before = json.loads(args.baseline.read_text(encoding='utf-8-sig'))
        after = json.loads(args.candidate.read_text(encoding='utf-8-sig'))
        rows = compare(before, after)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f'Comparison rejected: {exc}', file=sys.stderr)
        raise SystemExit(1)
    notes = [
        'Each metric is baseline -> candidate; no averages across workloads.',
        'Gaps measure SSE chunk arrivals, not per-token latency.',
        'Usage/wall throughput includes prefill; it is not pure decode throughput.',
        'Matching hashes/lengths do not establish equivalent OS/KV cache state or prove speedup.',
    ]
    if args.json:
        print(json.dumps({'notes': notes, 'comparisons': rows}, indent=2))
        return
    print('\n'.join(notes))
    for row in rows:
        print(f'\n{row["case"]} repeat={row["repeat"]} {row["request"]} '
              f'prompt_tokens={row["prompt_tokens"]} completion_tokens={row["completion_tokens"]}')
        for metric, (old, new) in row['metrics'].items():
            print(f'  {metric}: {number(old)} -> {number(new)}')


if __name__ == '__main__':
    main()
