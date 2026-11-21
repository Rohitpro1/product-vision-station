#!/usr/bin/env python3
"""Compare a candidate bundle's replay results against the active bundle (shadow before promotion).

  pvs --config configs/A.yaml replay clip.mp4 --out out/active
  pvs --config configs/B.yaml replay clip.mp4 --out out/candidate
  python scripts/shadow_compare.py out/active/results.jsonl out/candidate/results.jsonl
"""
from __future__ import annotations

import json
import sys

from pvs.geometry import periodic_error


def key(inst):
    return inst["track_id"] or inst["instance_id"]


def main(a: str, b: str) -> int:
    A = [json.loads(x) for x in open(a)]
    B = [json.loads(x) for x in open(b)]
    if [r["frame_id"] for r in A] != [r["frame_id"] for r in B]:
        print("frame sequences differ: inputs are not identical replays")
        return 2
    diff = {"frames": len(A), "identity_changed": 0, "axis_max_deg": 0.0, "heading_max_deg": 0.0,
            "heading_availability_changed": 0, "count_changed": 0}
    for ra, rb in zip(A, B):
        if len(ra["instances"]) != len(rb["instances"]):
            diff["count_changed"] += 1
        ib = {key(i): i for i in rb["instances"]}
        for ia in ra["instances"]:
            j = ib.get(key(ia))
            if not j:
                continue
            if (ia["identity"]["level"], ia["identity"]["sku"]) != (j["identity"]["level"], j["identity"]["sku"]):
                diff["identity_changed"] += 1
            pa, pb = ia["pose"], j["pose"]
            if pa["axis"]["available"] and pb["axis"]["available"]:
                diff["axis_max_deg"] = max(diff["axis_max_deg"], periodic_error(pa["axis"]["value_deg"], pb["axis"]["value_deg"], 180))
            if pa["heading"]["available"] != pb["heading"]["available"]:
                diff["heading_availability_changed"] += 1
            elif pa["heading"]["available"]:
                diff["heading_max_deg"] = max(diff["heading_max_deg"], periodic_error(pa["heading"]["value_deg"], pb["heading"]["value_deg"], 360))
    print(json.dumps(diff, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
