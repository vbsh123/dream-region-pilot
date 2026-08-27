import torch

from dream_region_pilot.benchmarks import (
    _humaneval_solution,
    gsm8k_cot_score_details,
    prepare_example,
    score_generation,
)
from dream_region_pilot.commit import (
    commit_active_regions,
    describe_commitments,
    sample_tokens,
)
from dream_region_pilot.config import load_config
from dream_region_pilot.dependencies import (
    aggregate_region_dependencies,
    threshold_region_graph,
)
from dream_region_pilot.evaluation import summarize
from dream_region_pilot.decoding import predicted_tail_region
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


def test_config_rejects_invalid_bounded_deferral_settings(tmp_path):
    config = tmp_path / "invalid.yaml"
    config.write_text(
        """
model: {}
data: {task: gsm8k}
generation: {region_size: 32, steps: 32, max_new_tokens: 32}
dependency: {}
experiment: {}
probe: {deferral_confidence_threshold: 1.1, max_region_deferrals: -1}
""",
        encoding="utf-8",
    )
    try:
        load_config(config)
    except ValueError as error:
        assert "deferral_confidence_threshold" in str(error)
    else:
        raise AssertionError("invalid bounded deferral settings were accepted")


def test_config_accepts_intermediate_region_sizes(tmp_path):
    config = tmp_path / "region-25.yaml"
    config.write_text(
        """
model: {}
data: {task: gsm8k}
generation: {region_size: 25, steps: 25, max_new_tokens: 256}
dependency: {}
experiment: {}
""",
        encoding="utf-8",
    )
    loaded = load_config(config)
    assert loaded["generation"]["region_size"] == 25


def test_config_rejects_negative_deferral_reveal_cutoff(tmp_path):
    config = tmp_path / "invalid-cutoff.yaml"
    config.write_text(
        """
model: {}
data: {task: gsm8k}
generation: {region_size: 32, steps: 32, max_new_tokens: 32}
dependency: {}
experiment: {}
probe: {deferral_until_revealed_tokens: -1}
""",
        encoding="utf-8",
    )
    try:
        load_config(config)
    except ValueError as error:
        assert "deferral_until_revealed_tokens" in str(error)
    else:
        raise AssertionError("negative deferral reveal cutoff was accepted")


def test_temperature_sampling_uses_multinomial_and_greedy_numeric_fallback(
    monkeypatch,
):
    logits = torch.tensor([[0.0, 3.0, 1.0]])
    calls = []

    def failing_multinomial(probabilities, num_samples):
        calls.append((probabilities, num_samples))
        raise RuntimeError("invalid multinomial distribution")

    monkeypatch.setattr(torch, "multinomial", failing_multinomial)
    confidence, predictions = sample_tokens(
        logits,
        temperature=0.1,
        top_p=0.9,
        top_k=None,
        policy="entropy",
    )

    assert len(calls) == 1
    assert calls[0][1] == 1
    assert predictions.tolist() == [1]
    assert torch.isfinite(confidence).all()


def test_commitment_diagnostics_report_raw_and_sampling_top_two():
    mask = 99
    tokens = torch.tensor([[mask, mask]])
    logits = torch.tensor(
        [[[1.0, 5.0, 0.0], [4.0, 1.0, 0.0]]], dtype=torch.float32
    )
    regions = build_fixed_regions(2, 1)
    committed = commit_active_regions(
        tokens,
        logits,
        prompt_length=0,
        mask_token_id=mask,
        regions=regions,
        local_steps=1,
        eps=1e-3,
        temperature=0.0,
        top_p=None,
        top_k=None,
        policy="entropy",
        alg_temp=None,
    )
    details = describe_commitments(
        logits,
        tokens,
        prompt_length=0,
        regions=regions,
        committed=committed,
        temperature=0.0,
        top_p=None,
        top_k=None,
        policy="entropy",
    )

    assert [item["token_id"] for item in details] == [1, 0]
    assert [item["raw_top1_token_id"] for item in details] == [1, 0]
    assert [item["raw_top2_token_id"] for item in details] == [0, 1]
    assert all(item["raw_chosen_probability"] > 0.9 for item in details)
    assert all(item["raw_region_confidence_rank"] == 1 for item in details)


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


def test_lm_eval_gsm8k_cot_strict_and_flexible_filters():
    colon = gsm8k_cot_score_details("The answer is: 3", "3")
    assert not colon["strict_match_correct"]
    assert colon["flexible_extract_correct"]

    official = gsm8k_cot_score_details("The answer is 3.", "3")
    assert official["strict_match_correct"]
    assert official["flexible_extract_correct"]

    data = {"task": "gsm8k", "protocol": "lm_eval_gsm8k_cot"}
    prediction, correct, method = score_generation(
        data, "First 2, finally 3.", "3"
    )
    assert prediction == "3."
    assert correct
    assert method == "lm_eval_gsm8k_cot_flexible_extract"


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
    assert parse_strategy("always_on_tail_guard") == (
        "always_on_tail_guard",
        0,
    )
    assert parse_strategy("flowblock_proxy") == ("flowblock_proxy", 0)
    assert parse_strategy("loose_wavefront") == ("loose_wavefront", 0)
    assert parse_strategy("controlled_position") == ("controlled_position", 0)
    assert parse_strategy("controlled_position_tail_guard") == (
        "controlled_position_tail_guard",
        0,
    )
    assert parse_strategy("controlled_dapd_dynamic") == (
        "controlled_dapd_dynamic",
        0,
    )
    assert parse_strategy("always_on_bounded_defer") == (
        "always_on_bounded_defer",
        0,
    )
    assert parse_strategy("always_on_bounded_defer_tail_guard") == (
        "always_on_bounded_defer_tail_guard",
        0,
    )
    assert parse_strategy("always_on_coupled_defer") == (
        "always_on_coupled_defer",
        0,
    )
    assert parse_strategy("always_on_coupled_defer_tail_guard") == (
        "always_on_coupled_defer_tail_guard",
        0,
    )


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


def test_tail_guard_can_exclude_provisional_tail_without_blocking_parent():
    regions = build_fixed_regions(12, 4)
    scheduler = RegionScheduler(
        regions,
        mode="controlled_position_tail_guard",
        max_active_regions=3,
        max_progress_gap=1,
    )
    scheduler.admitted_count = 3
    regions[0].remaining_mask_indices = ()
    regions[1].remaining_mask_indices = (7,)
    assert [
        region.index
        for region in scheduler.regions_allowed_to_advance(
            4, max_region_exclusive=2
        )
    ] == [1]


def test_always_on_tail_guard_has_no_admission_or_backpressure():
    regions = build_fixed_regions(12, 4)
    scheduler = RegionScheduler(regions, mode="always_on_tail_guard")
    assert scheduler.admitted_count == 3
    assert [
        region.index
        for region in scheduler.regions_allowed_to_advance(
            4, max_region_exclusive=2
        )
    ] == [0, 1]
    assert scheduler.last_control_edges == set()
    assert scheduler.last_blocked_regions == set()


def test_bounded_deferral_modes_start_every_region_without_backpressure():
    regions = build_fixed_regions(12, 4)
    scheduler = RegionScheduler(regions, mode="always_on_bounded_defer")
    assert scheduler.admitted_count == 3
    assert [
        region.index for region in scheduler.regions_allowed_to_advance(4)
    ] == [0, 1, 2]
    assert scheduler.last_control_edges == set()


def test_coupled_deferral_forces_the_lagging_endpoint_at_gap():
    regions = build_fixed_regions(16, 8)
    scheduler = RegionScheduler(
        regions,
        mode="always_on_coupled_defer",
        max_progress_gap=4,
    )
    # R0 has revealed four tokens while R1 has revealed none. R0 is paused and
    # R1 becomes urgent; all regions were nevertheless admitted from t=0.
    regions[0].remaining_mask_indices = (4, 5, 6, 7)
    regions[1].remaining_mask_indices = tuple(range(8, 16))
    allowed = scheduler.regions_allowed_to_advance(8)
    assert [region.index for region in allowed] == [1]
    assert scheduler.last_blocked_regions == {0}
    assert scheduler.last_urgent_regions == {1}
    assert scheduler.admitted_count == 2


def test_coupled_deferral_forced_reason_bypasses_low_confidence():
    regions = build_fixed_regions(4, 4)
    regions[0].schedule_step = 1
    mask_id = 2
    tokens = torch.full((1, 4), mask_id, dtype=torch.long)
    logits = torch.tensor(
        [[[0.0, 0.0, -10.0]] * 4], dtype=torch.float32
    )
    decisions = []
    committed = commit_active_regions(
        tokens,
        logits,
        prompt_length=0,
        mask_token_id=mask_id,
        regions=regions,
        local_steps=4,
        eps=0.001,
        temperature=0.0,
        top_p=None,
        top_k=None,
        policy="entropy",
        alg_temp=0.0,
        deferral_confidence_threshold=0.8,
        max_region_deferrals=None,
        region_deferral_counts={0: 17},
        deferral_decisions=decisions,
        force_region_reasons={0: "gap"},
    )
    assert len(committed[0]) == 1
    assert decisions[0]["action"] == "gap_forced"


def test_bounded_deferral_skips_then_forces_the_ordinary_update():
    regions = build_fixed_regions(4, 4)
    regions[0].schedule_step = 1
    mask_id = 2
    tokens = torch.full((1, 4), mask_id, dtype=torch.long)
    # All candidates have raw top-1 probability about 0.5, below the gate.
    logits = torch.tensor(
        [[[0.0, 0.0, -10.0]] * 4], dtype=torch.float32
    )
    deferral_counts = {0: 0}

    for expected_count in (1, 2):
        decisions = []
        committed = commit_active_regions(
            tokens,
            logits,
            prompt_length=0,
            mask_token_id=mask_id,
            regions=regions,
            local_steps=4,
            eps=0.001,
            temperature=0.0,
            top_p=None,
            top_k=None,
            policy="entropy",
            alg_temp=0.0,
            deferral_confidence_threshold=0.8,
            max_region_deferrals=2,
            region_deferral_counts=deferral_counts,
            deferral_decisions=decisions,
        )
        assert committed == {0: []}
        assert decisions[0]["action"] == "deferred"
        assert deferral_counts[0] == expected_count
        # The caller excludes deferred regions from apply_updates, so the same
        # local schedule point remains pending.
        assert regions[0].schedule_step == 1

    decisions = []
    committed = commit_active_regions(
        tokens,
        logits,
        prompt_length=0,
        mask_token_id=mask_id,
        regions=regions,
        local_steps=4,
        eps=0.001,
        temperature=0.0,
        top_p=None,
        top_k=None,
        policy="entropy",
        alg_temp=0.0,
        deferral_confidence_threshold=0.8,
        max_region_deferrals=2,
        region_deferral_counts=deferral_counts,
        deferral_decisions=decisions,
    )
    assert len(committed[0]) == 1
    assert decisions[0]["action"] == "forced"
    assert deferral_counts[0] == 0


def test_bounded_deferral_does_not_count_natural_zero_quota():
    regions = build_fixed_regions(4, 4)
    mask_id = 2
    tokens = torch.full((1, 4), mask_id, dtype=torch.long)
    logits = torch.tensor(
        [[[8.0, 0.0, -10.0]] * 4], dtype=torch.float32
    )
    deferral_counts = {0: 0}
    decisions = []
    committed = commit_active_regions(
        tokens,
        logits,
        prompt_length=0,
        mask_token_id=mask_id,
        regions=regions,
        local_steps=4,
        eps=0.001,
        temperature=0.0,
        top_p=None,
        top_k=None,
        policy="entropy",
        alg_temp=0.0,
        deferral_confidence_threshold=0.8,
        max_region_deferrals=2,
        region_deferral_counts=deferral_counts,
        deferral_decisions=decisions,
    )
    assert committed == {0: []}
    assert decisions == [
        {
            "region": 0,
            "action": "natural_zero_quota",
            "scheduled_quota": 0,
            "minimum_scheduled_raw_top1_probability": None,
            "consecutive_deferrals": 0,
        }
    ]
    assert deferral_counts == {0: 0}


def test_predicted_tail_region_uses_earliest_masked_stop():
    regions = build_fixed_regions(8, 4)
    logits = torch.zeros((1, 10, 5))
    logits[0, 6, 4] = 3.0  # response position 4, in R1
    logits[0, 3, 4] = 4.0  # response position 1, but already revealed
    response_mask = torch.tensor(
        [[True, False, True, True, True, True, True, True]]
    )
    assert predicted_tail_region(
        logits,
        prompt_length=2,
        response_mask=response_mask,
        regions=regions,
        stop_ids={4},
    ) == 1


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


def test_dynamic_commit_groups_follow_current_dependency_components():
    regions = build_fixed_regions(8, 2)
    scheduler = RegionScheduler(regions, mode="controlled_dapd_dynamic")
    scheduler.active_dependency_edges = {(0, 1), (1, 2)}
    assert [
        [region.index for region in group]
        for group in scheduler.commitment_groups(regions)
    ] == [[0, 1, 2], [3]]

    scheduler.active_dependency_edges = {(2, 3)}
    assert [
        [region.index for region in group]
        for group in scheduler.commitment_groups(regions)
    ] == [[0], [1], [2, 3]]


def test_pooled_component_selects_jointly_instead_of_per_region():
    regions = build_fixed_regions(8, 4)
    for region in regions:
        region.schedule_step = 1
    mask_id = 2
    tokens = torch.full((1, 8), mask_id, dtype=torch.long)
    logits = torch.tensor(
        [
            [
                [8.0, 0.0, -8.0],
                [7.0, 0.0, -8.0],
                [2.0, 1.0, -8.0],
                [2.0, 1.0, -8.0],
                [1.1, 1.0, -8.0],
                [1.1, 1.0, -8.0],
                [1.1, 1.0, -8.0],
                [1.1, 1.0, -8.0],
            ]
        ]
    )
    committed = commit_active_regions(
        tokens,
        logits,
        prompt_length=0,
        mask_token_id=mask_id,
        regions=regions,
        local_steps=4,
        eps=0.001,
        temperature=0.0,
        top_p=None,
        top_k=None,
        policy="entropy",
        alg_temp=0.0,
        region_groups=[regions],
    )
    assert len(committed[0]) == 2
    assert committed[1] == []


def test_pooled_component_cannot_spend_a_regions_terminal_budget_elsewhere():
    regions = build_fixed_regions(8, 4)
    regions[0].schedule_step = 3
    regions[1].schedule_step = 1
    mask_id = 2
    tokens = torch.full((1, 8), mask_id, dtype=torch.long)
    logits = torch.tensor(
        [
            [
                [1.1, 1.0, -8.0],
                [1.1, 1.0, -8.0],
                [1.1, 1.0, -8.0],
                [1.1, 1.0, -8.0],
                [8.0, 0.0, -8.0],
                [7.0, 0.0, -8.0],
                [6.0, 0.0, -8.0],
                [5.0, 0.0, -8.0],
            ]
        ]
    )
    committed = commit_active_regions(
        tokens,
        logits,
        prompt_length=0,
        mask_token_id=mask_id,
        regions=regions,
        local_steps=4,
        eps=0.001,
        temperature=0.0,
        top_p=None,
        top_k=None,
        policy="entropy",
        alg_temp=0.0,
        region_groups=[regions],
    )

    assert committed[0] == [0, 1, 2, 3]
    assert committed[1] == [4]
    RegionScheduler.apply_updates(regions, committed)
    assert regions[0].schedule_step == 4
    assert regions[0].clock == 1
    assert regions[1].schedule_step == 2
    assert regions[1].clock == 1


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
    assert controlled["mean_effective_generated_tokens"] == 80
    assert controlled["nfe_speedup_vs_vanilla"] == 2.0
    assert controlled["wall_clock_speedup_vs_vanilla"] == 2.0
    assert controlled["canvas_tps_speedup_vs_vanilla"] == 2.0
