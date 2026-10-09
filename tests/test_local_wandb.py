import json
import struct
import zlib

import pytest
from wandb.proto import wandb_internal_pb2 as pb

from scripts.local_wandb import read_history


def frame(payload, kind=1):
    return struct.pack('<IHB', zlib.crc32(bytes([kind]) + payload), len(payload), kind) + payload


def test_reads_complete_history_and_ignores_incomplete_live_tail(tmp_path):
    record = pb.Record()
    for key, val in [('optimizer_step', 25000), ('representation/effective_rank', 75.2)]:
        item = record.history.item.add()
        item.key, item.value_json = key, json.dumps(val)
    p = tmp_path / 'live.wandb'
    p.write_bytes(b':W&B\xe1\xbe\x00' + frame(record.SerializeToString()) + b'\x01\x02')
    assert read_history(p) == [{'optimizer_step': 25000, 'representation/effective_rank': 75.2}]


def test_corrupt_history_is_rejected(tmp_path):
    p = tmp_path / 'bad.wandb'
    p.write_bytes(b':W&B\xe1\xbe\x00' + struct.pack('<IHB', 0, 2, 1) + b'xx')
    with pytest.raises(ValueError, match='CRC'):
        read_history(p)


def test_reassembles_a_history_record_crossing_block_boundaries(tmp_path):
    record = pb.Record()
    item = record.history.item.add()
    item.nested_key.extend(['train', 'large'])
    item.value_json = json.dumps('a' * 40000)
    payload = record.SerializeToString()
    first_length = 32768 - 7 - 7
    p = tmp_path / 'fragmented.wandb'
    p.write_bytes(b':W&B\xe1\xbe\x00' + frame(payload[:first_length], 2) + frame(payload[first_length:], 4))
    assert read_history(p) == [{'train/large': 'a' * 40000}]
