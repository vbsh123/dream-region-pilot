import torch

from dream_region_pilot.dependencies import (
    aggregate_region_dependencies,
    threshold_region_graph,
)
from dream_region_pilot.regions import build_fixed_regions
from dream_region_pilot.scheduler import RegionScheduler
from dream_region_pilot.mean_field import topk_tail_jsd_dependency


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


def test_wavefront_admits_only_one_region_per_iteration():
    regions = build_fixed_regions(8, 2)
    scheduler = RegionScheduler(
        regions,
        mode="wavefront_probe",
        max_active_regions=4,
        spawn_readiness=0.15,
    )
    assert [region.index for region in scheduler.regions_allowed_to_advance(2)] == [0]
    assert scheduler.maybe_admit_next({0: 0.14}) == []
    assert scheduler.maybe_admit_next({0: 0.15}) == [1]
    assert scheduler.maybe_admit_next({1: 1.0}) == [2]


def test_mean_field_jsd_signal_is_symmetric_and_zero_diagonal():
    logits = torch.tensor(
        [[[8.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 8.0, 0.0]]]
    )
    raw_matrix, matrix, metadata = topk_tail_jsd_dependency(
        logits,
        torch.tensor([[True, True, True]]),
        topk=3,
        pair_chunk_size=2,
    )
    assert torch.allclose(matrix, matrix.T)
    assert torch.equal(matrix.diag(), torch.zeros(3))
    assert matrix[0, 1] > matrix[0, 2]
    assert raw_matrix[0, 1] > raw_matrix[0, 2]
    assert metadata["paper_exact"] is True
