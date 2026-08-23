import torch

from dream_region_pilot.dependencies import (
    aggregate_region_dependencies,
    threshold_region_graph,
)
from dream_region_pilot.regions import build_fixed_regions
from dream_region_pilot.scheduler import RegionScheduler


def test_fixed_regions_exclude_prompt_by_construction():
    regions = build_fixed_regions(70, 32)
    assert [region.token_indices for region in regions] == [
        tuple(range(0, 32)),
        tuple(range(32, 64)),
        tuple(range(64, 70)),
    ]


def test_region_aggregators_and_positional_orientation():
    regions = build_fixed_regions(4, 2)
    dependency = torch.zeros(4, 4)
    dependency[:2, 2:] = 0.75
    dependency[2:, :2] = 0.75
    matrix = aggregate_region_dependencies(dependency, regions, method="mean")
    parents, edges = threshold_region_graph(matrix, threshold=0.5)
    assert parents == {0: set(), 1: {0}}
    assert edges == [{"left": 0, "right": 1, "score": 0.75}]


def test_lag_one_pipeline_and_terminal_release():
    regions = build_fixed_regions(6, 2)
    scheduler = RegionScheduler(
        regions, mode="fixed_lag", lag=1, release_completed_parents=True
    )
    scheduler.set_parents({0: set(), 1: {0}, 2: {1}})
    assert [region.index for region in scheduler.regions_allowed_to_advance(2)] == [0]
    advanced = scheduler.apply_updates([regions[0]], {0: []})
    assert advanced == []
    assert regions[0].schedule_step == 1
    assert regions[0].clock == 0
    assert [region.index for region in scheduler.regions_allowed_to_advance(2)] == [0]
    advanced = scheduler.apply_updates([regions[0]], {0: [0]})
    assert advanced == [regions[0]]
    assert regions[0].schedule_step == 2
    assert regions[0].clock == 1
    regions[0].remaining_mask_indices = ()
    assert [region.index for region in scheduler.regions_allowed_to_advance(2)] == [1]
    scheduler.apply_updates([regions[1]], {1: [2]})
    assert [region.index for region in scheduler.regions_allowed_to_advance(2)] == [1, 2]
