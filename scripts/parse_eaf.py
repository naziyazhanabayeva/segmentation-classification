#!/usr/bin/env python3
"""parse_eaf.py

Parse an ELAN .eaf file, extract SIGN and SENTENCE tiers
(ALIGNABLE_ANNOTATION),
resolve TIME_SLOT_REF* to milliseconds and write segments.json.

Usage:
    python parse_eaf.py --input kazakh_output.eaf --output kazakh_segments.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

TARGET_TIERS = {"SIGN", "SENTENCE"}


def _local_name(tag: str) -> str:
    """Return the local name of an XML tag, ignoring namespace."""
    return tag.split('}')[-1] if '}' in tag else tag


def parse_time_slots(root: ET.Element) -> dict:
    """Parse TIME_ORDER/TIME_SLOT elements into a dict TIME_SLOT_ID ->
    ms (int).

    ELAN stores time values in milliseconds in the TIME_VALUE attribute.
    """
    times: dict[str, int | None] = {}
    # find TIME_ORDER element (namespace-agnostic)
    time_orders = [
        el for el in root.iter() if _local_name(el.tag) == 'TIME_ORDER'
    ]
    for to in time_orders:
        for ts in to:
            if _local_name(ts.tag) != 'TIME_SLOT':
                continue
            ts_id = ts.attrib.get('TIME_SLOT_ID')
            tv = ts.attrib.get('TIME_VALUE')
            if ts_id is None:
                continue
            if tv is None or tv == '':
                times[ts_id] = None
                continue
            try:
                # TIME_VALUE is typically an integer milliseconds string
                times[ts_id] = int(float(tv))
            except Exception:
                times[ts_id] = None
    return times


def extract_tier_segments(root: ET.Element, times: dict) -> dict:
    segments = {tier: [] for tier in TARGET_TIERS}

    for tier in root.iter():
        if _local_name(tier.tag) != 'TIER':
            continue
        tier_id = tier.attrib.get('TIER_ID')
        if tier_id not in TARGET_TIERS:
            continue

        # find ALIGNABLE_ANNOTATION elements inside this tier
        for ann in tier.iter():
            if _local_name(ann.tag) != 'ALIGNABLE_ANNOTATION':
                continue
            ts1 = ann.attrib.get('TIME_SLOT_REF1')
            ts2 = ann.attrib.get('TIME_SLOT_REF2')
            if not ts1 or not ts2:
                continue
            start = times.get(ts1)
            end = times.get(ts2)
            if start is None or end is None:
                continue
            segments[tier_id].append(
                {"start_ms": int(start), "end_ms": int(end)}
            )

    # sort segments by start time
    for tier in segments:
        segments[tier].sort(key=lambda x: x["start_ms"])
    return segments


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description='Parse ELAN .eaf and extract SIGN/SENTENCE segments'
    )
    parser.add_argument(
        '--input',
        default='output.eaf',
        help='input .eaf file (default: output.eaf)',
    )
    parser.add_argument(
        '--output',
        default='segments.json',
        help='output JSON file (default: segments.json)',
    )
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        parser.error(f'Input file not found: {in_path}')

    out_path.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(in_path)
    root = tree.getroot()

    times = parse_time_slots(root)
    segments = extract_tier_segments(root, times)

    out_path.write_text(json.dumps(segments, indent=2, sort_keys=False))
    print(f'Wrote {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
