#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the primary dynamic dependency graph timeline"
    )
    parser.add_argument(
        "timeline",
        type=Path,
        help="Path to diagnostics/.../wavefront_probe/graph_timeline.json",
    )
    args = parser.parse_args()
    snapshots = json.loads(args.timeline.read_text(encoding="utf-8"))
    selected_names = {
        "dapd",
        "mean_field_t0.90",
        "combined_union_t0.90",
        "combined_intersection_t0.90",
    }
    print(
        "step  signal                         edges density nonadj components "
        "added removed jaccard"
    )
    for snapshot in snapshots:
        metadata = snapshot.get("metadata", {}).get("mean_field", {})
        if metadata:
            print(
                f"\nstep {snapshot['iteration']}: mean-field top-k mass "
                f"mean={metadata.get('mean_topk_probability_mass')} "
                f"min={metadata.get('minimum_topk_probability_mass')} "
                f"exact={metadata.get('paper_exact')}"
            )
        for name, values in snapshot.get("signals", {}).items():
            if name not in selected_names:
                continue
            print(
                f"{snapshot['iteration']:>4}  {name:<30} "
                f"{values['edge_count']:>5} "
                f"{values['active_edge_density']:>7.3f} "
                f"{values['nonadjacent_edge_count']:>6} "
                f"{values['active_component_count']:>10} "
                f"{len(values['edges_added_since_previous']):>5} "
                f"{len(values['edges_removed_since_previous']):>7} "
                f"{values['edge_jaccard_with_previous']:>7.3f}"
            )


if __name__ == "__main__":
    main()
