"""Read-only decoder for local W&B v0 datastore records.

The seven-byte W&B header precedes 32 KiB blocks with CRC32/type/length
fragments. Complete records are protobufs; an incomplete live tail is ignored.
No authentication, network access, syncing, or source-run mutation occurs.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import struct
import zlib


def read_history(path):
    from wandb.proto import wandb_internal_pb2 as pb
    blob = Path(path).read_bytes()
    if blob[:7] != b':W&B\xe1\xbe\x00':
        raise ValueError('Unsupported W&B datastore header')
    offset, fragment, rows = 7, None, []
    while offset + 7 <= len(blob):
        remaining = 32768 - offset % 32768
        if remaining < 7 or blob[offset:offset + 7] == bytes(7):
            offset += remaining
            continue
        crc, length, kind = struct.unpack_from('<IHB', blob, offset)
        if length > remaining - 7:
            raise ValueError('W&B fragment crosses a block boundary')
        offset += 7
        if offset + length > len(blob):
            break
        chunk = blob[offset:offset + length]
        offset += length
        if zlib.crc32(bytes([kind]) + chunk) != crc:
            raise ValueError('W&B fragment CRC mismatch')
        if kind == 1:
            if fragment is not None:
                raise ValueError('Unexpected full W&B record within fragmented record')
            payload = chunk
        elif kind == 2:
            fragment = chunk
            continue
        elif kind in (3, 4):
            if fragment is None:
                raise ValueError('W&B continuation without first fragment')
            fragment += chunk
            if kind == 3:
                continue
            payload, fragment = fragment, None
        else:
            raise ValueError(f'Unknown W&B record kind {kind}')
        record = pb.Record()
        record.ParseFromString(payload)
        if record.HasField('history'):
            rows.append({'/'.join(i.nested_key) if i.nested_key else i.key: json.loads(i.value_json)
                         for i in record.history.item})
    return rows


def paired_histories(root, study):
    runs, errors = [], []
    for run in (study or {}).get('runs', []):
        run_id = run.get('run_id')
        if not run_id:
            continue
        paths = sorted((Path(root) / 'wandb').glob(f'run-*-{run_id}/run-{run_id}.wandb'))
        if not paths:
            continue
        try:
            history = read_history(paths[-1])
        except ValueError as exc:
            errors.append(f'{run_id}: {exc}')
            continue
        points = []
        for row in history:
            step = row.get('optimizer_step', row.get('train/global_step', row.get('_step')))
            if step is None:
                continue
            selected = {k: row[k] for k in ('valid/mse_loss', 'valid/sigreg_loss',
                        'representation/effective_rank', 'representation/token_effective_rank',
                        'representation/mean_pairwise_cosine') if k in row}
            if selected:
                points.append({'step': step, **selected})
        latest = next((r for r in reversed(history) if 'train/global_step' in r), {})
        runs.append({'run_id': run_id, 'patches': run['patches'], 'seed': run['seed'], 'history': points,
                     'latest_step': latest.get('train/global_step'), 'latest_timestamp': latest.get('_timestamp'),
                     'source': str(paths[-1].relative_to(root))})
    return runs, errors


def timestamp_text(timestamp, timezone='America/Blanc-Sablon'):
    return datetime.fromtimestamp(timestamp, ZoneInfo(timezone)).strftime('%Y-%m-%d %H:%M %Z (UTC%z)')
