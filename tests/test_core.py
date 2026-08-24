import torch

from dream_region_pilot.benchmarks import (
    _humaneval_solution,
    prepare_example,
    score_generation,
)
from dream_region_pilot.dependencies import (
    aggregate_region_dependencies,
    threshold_region_graph,
)
from dream_region_pilot.evaluation import summarize
from dream_region_pilot.regions import build_fixed_regions
from dream_region_pilot.scheduler import RegionScheduler, parse_strategy
from dream_region_pilot.mean_field import (
    exact_jsd_interaction,
    mean_field_commit_indices,
    topk_tail_jsd_dependency,
)


def test_fixed_regions_exclude_prompt_by_construction():
    regions = build_fixed_regions(70, 32)
    assert [region.token_indices for region in regions] == [
        tuple(range(0, 32)),
        tuple(range(32, 64)),
        tuple(range(64, 70)),
    ]


def test_asdiv_combines_body_and_question_and_scores_text_or_number():
    data = {"task": "asdiv"}
    numeric = prepare_example(
        data,
        {"body": "There are 7 red and 2 green apples.", "question": "How many?", "answer": "9 (apples)"},
    )
    assert numeric.question == "There are 7 red and 2 green apples.\nHow many?"
    assert score_generation(data, "#### 9", numeric.reference_answer)[1]
    categorical = prepare_example(
        data,
        {"body": "Ann has more.", "question": "Who?", "answer": "Ann"},
    )
    assert score_generation(data, "#### Ann", categorical.reference_answer)[1]


def test_humaneval_full_function_or_body_becomes_complete_solution():
    source = {
        "task_id": "HumanEval/0",
        "prompt": "from typing import List\n\ndef add_one(x: int) -> int:\n    \"\"\"Add one.\"\"\"\n",
        "entry_point": "add_one",
    }
    full = _humaneval_solution(
        "```python\ndef add_one(x: int) -> int:\n    return x + 1\n```", source
    )
    assert full.startswith("from typing import List")
    assert "def add_one" in full
    assert "    return x + 1" in full
    body = _humaneval_solution("return x + 1", source)
    assert body.endswith("    return x + 1\n")


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


def test_flowblock_proxy_uses_a_two_region_active_window():
    regions = build_fixed_regions(8, 2)
    scheduler = RegionScheduler(
        regions,
        mode="flowblock_proxy",
        max_active_regions=2,
        spawn_readiness=0.60,
    )
    assert scheduler.maybe_admit_next({0: 0.59}) == []
    assert scheduler.maybe_admit_next({0: 0.60}) == [1]
    assert scheduler.maybe_admit_next({1: 1.0}) == []
    regions[0].remaining_mask_indices = ()
    assert scheduler.maybe_admit_next({1: 0.60}) == [2]


def test_new_comparison_strategy_names_parse_without_changing_mode():
    assert parse_strategy("flowblock_proxy") == ("flowblock_proxy", 0)
    assert parse_strategy("loose_wavefront") == ("loose_wavefront", 0)
    assert parse_strategy("controlled_position") == ("controlled_position", 0)


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


def test_exact_mean_field_algorithm_selects_and_has_progress_fallback():
    logits = torch.tensor([[8.0, 0.0, 0.0], [7.0, 0.0, 0.0]])
    interaction = exact_jsd_interaction(logits, pair_chunk_size=1)
    assert torch.allclose(interaction, interaction.T)
    assert torch.equal(interaction.diag(), torch.zeros(2))
    selected, intensity, fallback = mean_field_commit_indices(
        logits, threshold=0.99, iterations=2, pair_chunk_size=1
    )
    assert selected.numel() >= 1
    assert intensity.shape == (2,)
    assert isinstance(fallback, bool)


def test_loose_wavefront_has_no_positional_backpressure_edges():
    regions = build_fixed_regions(8, 2)
    scheduler = RegionScheduler(
        regions,
        mode="loose_wavefront",
        max_active_regions=4,
        spawn_readiness=0.15,
    )
    scheduler.admitted_count = 4
    assert [
        region.index for region in scheduler.regions_allowed_to_advance(2)
    ] == [0, 1, 2, 3]
    assert scheduler.last_control_edges == set()


def test_controlled_scheduler_prioritizes_a_lagging_child_without_equalizing():
    regions = build_fixed_regions(8, 4)
    scheduler = RegionScheduler(
        regions,
        mode="controlled_dapd",
        max_active_regions=2,
        max_progress_gap=2,
        edge_persistence=2,
    )
    scheduler.admitted_count = 2
    regions[0].remaining_mask_indices = (3,)
    regions[1].remaining_mask_indices = (4, 5, 6, 7)
    allowed = scheduler.regions_allowed_to_advance(4)
    assert [region.index for region in allowed] == [1]
    assert scheduler.last_blocked_regions == {0}
    assert scheduler.last_urgent_regions == {1}


def test_position_control_uses_only_adjacent_edges():
    regions = build_fixed_regions(12, 4)
    scheduler = RegionScheduler(
        regions,
        mode="controlled_position",
        max_active_regions=3,
        max_progress_gap=8,
    )
    scheduler.admitted_count = 3
    assert [
        region.index for region in scheduler.regions_allowed_to_advance(4)
    ] == [0, 1, 2]
    assert scheduler.last_control_edges == {(0, 1), (1, 2)}
    assert scheduler.active_dependency_edges == set()


def test_dependency_edge_requires_real_progress_and_two_observations():
    regions = build_fixed_regions(8, 4)
    scheduler = RegionScheduler(
        regions,
        mode="controlled_dapd",
        max_active_regions=2,
        edge_persistence=2,
    )
    scheduler.admitted_count = 2
    edge = [{"left": 0, "right": 1, "score": 1.0}]
    scheduler.observe_dependency_edges(edge)
    assert scheduler.active_dependency_edges == set()
    regions[0].remaining_mask_indices = (1, 2, 3)
    regions[1].remaining_mask_indices = (5, 6, 7)
    scheduler.observe_dependency_edges(edge)
    assert scheduler.active_dependency_edges == set()
    scheduler.observe_dependency_edges(edge)
    assert scheduler.active_dependency_edges == {(0, 1)}


def test_summary_reports_measured_throughput_and_vanilla_speedups():
    common = {
        "correct": True,
        "average_tokens_committed_per_forward": 1.0,
        "canvas_tokens": 256,
        "dependency_seconds": 0.0,
    }
    summary = summarize(
        [
            {
                **common,
                "strategy": "vanilla",
                "nfe": 100,
                "effective_generated_tokens": 100,
                "wall_clock_seconds": 10.0,
            },
            {
                **common,
                "strategy": "controlled_position",
                "nfe": 50,
                "effective_generated_tokens": 80,
                "wall_clock_seconds": 5.0,
            },
        ]
    )
    by_strategy = {item["strategy"]: item for item in summary}
    controlled = by_strategy["controlled_position"]
    assert controlled["mean_tokens_committed_per_forward"] == 256 / 50
    assert controlled["canvas_tokens_per_second"] == 256 / 5
    assert controlled["effective_tokens_per_second"] == 80 / 5
    assert controlled["nfe_speedup_vs_vanilla"] == 2.0
    assert controlled["wall_clock_speedup_vs_vanilla"] == 2.0
    assert controlled["canvas_tps_speedup_vs_vanilla"] == 2.0
