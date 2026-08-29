import torch

from dream_region_pilot.benchmarks import (
    _humaneval_solution,
    build_fewshot_contexts,
    gsm8k_cot_score_details,
    prepare_example,
    score_generation,
)
from dream_region_pilot.commit import (
    commit_active_regions,
    commit_active_regions_dawn,
    dawn_region_transfer_mask,
    describe_commitments,
    local_transfer_count,
    sample_tokens,
)
from dream_region_pilot.config import load_config
from dream_region_pilot.evaluation import summarize
from dream_region_pilot.decoding import decode_official_dawn, predicted_tail_region
from dream_region_pilot.regions import build_fixed_regions
from dream_region_pilot.scheduler import RegionScheduler, parse_strategy
from dream_region_pilot.mean_field import (
    exact_jsd_interaction,
    mean_field_commit_indices,
)
from dream_region_pilot.model_adapter import shift_logits_dream


def test_fixed_regions_exclude_prompt_by_construction():
    regions = build_fixed_regions(70, 32)
    assert [region.token_indices for region in regions] == [
        tuple(range(0, 32)),
        tuple(range(32, 64)),
        tuple(range(64, 70)),
    ]


def test_fewshot_contexts_match_lm_eval_continuous_sampler(monkeypatch):
    documents = [
        {"question": f"demo-{index}", "answer": f"work-{index} #### {index}"}
        for index in range(10)
    ]

    def fake_load_dataset(*args, **kwargs):
        assert kwargs["split"] == "train"
        return documents

    monkeypatch.setattr(
        "dream_region_pilot.benchmarks.load_dataset", fake_load_dataset
    )
    contexts = build_fewshot_contexts(
        {
            "task": "gsm8k",
            "dataset": "ignored",
            "subset": "main",
            "num_fewshot": 2,
            "fewshot_split": "train",
            "fewshot_seed": 1234,
            "fewshot_prompt_template": "Question: {question}\nAnswer:",
            "target_delimiter": " ",
            "fewshot_delimiter": "\n\n",
        },
        [0, 2],
    )

    assert contexts[0].startswith("Question: demo-7\nAnswer: work-7")
    assert "Question: demo-1\nAnswer: work-1" in contexts[0]
    # Index one is not requested, but its random draw must still be consumed.
    assert contexts[2].startswith("Question: demo-9\nAnswer: work-9")
    assert "Question: demo-0\nAnswer: work-0" in contexts[2]


def test_local_dream_logit_shift_matches_position_alignment():
    logits = torch.tensor([[[1.0], [2.0], [3.0]]])
    shifted = shift_logits_dream(logits)
    assert torch.equal(shifted, torch.tensor([[[1.0], [1.0], [2.0]]]))


def test_32_token_local_dream_schedule_has_expected_transfer_quotas():
    remaining = 32
    quotas = []
    for schedule_step in range(32):
        quota = local_transfer_count(
            remaining,
            schedule_step=schedule_step,
            local_steps=32,
            eps=0.001,
        )
        quotas.append(quota)
        remaining -= quota
    assert quotas == [0] + [1] * 30 + [2]
    assert remaining == 0


def test_config_rejects_invalid_bounded_deferral_settings(tmp_path):
    config = tmp_path / "invalid.yaml"
    config.write_text(
        """
model: {}
data: {task: gsm8k}
generation: {region_size: 32, steps: 32, max_new_tokens: 32}
sources: {}
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
    for region_size in (25, 40):
        config = tmp_path / f"region-{region_size}.yaml"
        config.write_text(
            f"""
model: {{}}
data: {{task: gsm8k}}
generation: {{region_size: {region_size}, steps: 32, max_new_tokens: 256}}
sources: {{}}
experiment: {{}}
""",
            encoding="utf-8",
        )
        loaded = load_config(config)
        assert loaded["generation"]["region_size"] == region_size


def test_config_accepts_explicit_dawn_model_implementation(tmp_path):
    config = tmp_path / "dawn-backend.yaml"
    config.write_text(
        """
model: {implementation: dawn_release}
data: {task: gsm8k}
generation: {region_size: 32, steps: 256, max_new_tokens: 256}
sources: {}
experiment: {}
""",
        encoding="utf-8",
    )
    loaded = load_config(config)
    assert loaded["model"]["implementation"] == "dawn_release"


def test_config_rejects_unknown_model_implementation(tmp_path):
    config = tmp_path / "bad-backend.yaml"
    config.write_text(
        """
model: {implementation: accidental_backend}
data: {task: gsm8k}
generation: {region_size: 32, steps: 256, max_new_tokens: 256}
sources: {}
experiment: {}
""",
        encoding="utf-8",
    )
    try:
        load_config(config)
    except ValueError as error:
        assert "model.implementation" in str(error)
    else:
        raise AssertionError("unknown model implementation was accepted")


def test_config_rejects_negative_deferral_reveal_cutoff(tmp_path):
    config = tmp_path / "invalid-cutoff.yaml"
    config.write_text(
        """
model: {}
data: {task: gsm8k}
generation: {region_size: 32, steps: 32, max_new_tokens: 32}
sources: {}
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


def test_loose_wavefront_admits_only_one_region_per_iteration():
    regions = build_fixed_regions(8, 2)
    scheduler = RegionScheduler(
        regions,
        mode="loose_wavefront",
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
    names = (
        "always_on_tail_guard",
        "flowblock_proxy",
        "loose_wavefront",
        "controlled_position",
        "controlled_position_tail_guard",
        "always_on_bounded_defer",
        "always_on_bounded_defer_tail_guard",
        "always_on_coupled_defer",
        "always_on_coupled_defer_tail_guard",
        "always_on_coupled_defer_stop_filter",
        "always_on_coupled_defer_stop_defer",
        "always_on_dawn_tail_guard",
        "always_on_coupled_defer_dawn_tail_guard",
    )
    assert [parse_strategy(name) for name in names] == list(names)


def test_dawn_conflict_selection_stops_after_suppressing_neighbors():
    attention = torch.zeros((4, 4), dtype=torch.float32)
    attention[1, 0] = 0.8
    mask = torch.ones(4, dtype=torch.bool)
    confidence = torch.tensor([0.85, 0.84, 0.83, 0.82])
    selected, stats = dawn_region_transfer_mask(
        attention,
        mask,
        confidence,
        torch.ones(4, dtype=torch.bool),
        sink_threshold=1.0,
        edge_threshold=0.5,
        high_confidence_threshold=0.9,
        induce_threshold=0.75,
        candidate_confidence_threshold=0.8,
    )
    # Positions zero and one conflict. Greedy confidence keeps zero, then the
    # two independent positions; it must not reselect the suppressed neighbor.
    assert torch.equal(selected, torch.tensor([True, False, True, True]))
    assert stats["conflict_mis"] == 3
    assert stats["fallback"] is False


def test_regional_dawn_fallback_is_scoped_to_active_region():
    regions = build_fixed_regions(4, 2)
    mask_id = 2
    tokens = torch.full((1, 4), mask_id, dtype=torch.long)
    logits = torch.tensor(
        [[[2.0, 0.0, -8.0], [1.0, 0.0, -8.0],
          [8.0, 0.0, -8.0], [7.0, 0.0, -8.0]]]
    )
    stats = []
    committed = commit_active_regions_dawn(
        tokens,
        logits,
        torch.zeros((1, 4, 4)),
        prompt_length=0,
        mask_token_id=mask_id,
        regions=[regions[0]],
        temperature=0.0,
        top_p=None,
        top_k=None,
        dawn_config={
            "high_confidence_threshold": 1.0,
            "candidate_confidence_threshold": 1.0,
        },
        selector_stats=stats,
    )
    assert committed == {0: [0]}
    assert tokens[0, 0].item() == 0
    assert torch.equal(tokens[0, 1:], torch.tensor([2, 2, 2]))
    assert stats[0]["fallback"] is True


def test_regional_dawn_thresholds_use_untempered_confidence():
    region = build_fixed_regions(1, 1)[0]
    mask_id = 2
    tokens = torch.tensor([[mask_id]])
    # Raw top-1 probability is about 0.73, whereas temperature 0.1 would make
    # it almost one. DAWN's 0.9 threshold must therefore miss and use fallback.
    logits = torch.tensor([[[1.0, 0.0, -10.0]]])
    stats = []
    commit_active_regions_dawn(
        tokens,
        logits,
        torch.zeros((1, 1, 1)),
        prompt_length=0,
        mask_token_id=mask_id,
        regions=[region],
        temperature=0.1,
        top_p=0.9,
        top_k=None,
        dawn_config={},
        selector_stats=stats,
    )
    assert stats[0]["confident"] == 0
    assert stats[0]["fallback"] is True


def test_official_dawn_wrapper_uses_released_decoder_settings():
    class Output:
        sequences = torch.tensor([[7, 8, 1, 1, 1, 1]])

    class Model:
        def __init__(self):
            self.kwargs = None

        def diffusion_generate(self, prompt, **kwargs):
            self.kwargs = kwargs
            return Output(), 11

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 9

        @staticmethod
        def get_vocab():
            return {}

        @staticmethod
        def decode(values, skip_special_tokens=True):
            return "answer"

    model = Model()
    result = decode_official_dawn(
        model,
        Tokenizer(),
        torch.tensor([[7, 8]]),
        {
            "max_new_tokens": 4,
            "until": [],
        },
        {
            "block_length": 2,
            "candidate_confidence_threshold": 0.8,
            "induce_threshold": 0.75,
            "sink_threshold": 0.03,
            "edge_threshold": 0.10,
        },
    )
    assert result["nfe"] == 11
    assert result["configured_steps"] == 2
    assert model.kwargs["alg"] == "dawn"
    assert model.kwargs["temperature"] == 0.0
    assert model.kwargs["top_p"] is None
    assert model.kwargs["block_length"] == 2
    assert model.kwargs["conf_threshold"] == 0.8


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
        mode="controlled_position",
        max_active_regions=2,
        max_progress_gap=2,
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


def test_stop_stalled_child_does_not_pause_its_parent_at_gap():
    regions = build_fixed_regions(16, 8)
    scheduler = RegionScheduler(
        regions,
        mode="always_on_coupled_defer_stop_defer",
        max_progress_gap=4,
    )
    regions[0].remaining_mask_indices = (4, 5, 6, 7)
    regions[1].remaining_mask_indices = tuple(range(8, 16))
    allowed = scheduler.regions_allowed_to_advance(
        8, progress_gap_exempt_children={1}
    )
    assert [region.index for region in allowed] == [0, 1]
    assert scheduler.last_blocked_regions == set()
    assert scheduler.last_urgent_regions == set()


def test_coupled_tail_guard_starts_all_regions_but_excludes_tail_suffix():
    regions = build_fixed_regions(16, 4)
    scheduler = RegionScheduler(
        regions,
        mode="always_on_coupled_defer_tail_guard",
        max_progress_gap=4,
    )
    assert scheduler.admitted_count == 4
    assert [
        region.index
        for region in scheduler.regions_allowed_to_advance(
            4, max_region_exclusive=2
        )
    ] == [0, 1]
    assert scheduler.last_control_edges == {(0, 1)}


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


def test_closed_startup_window_bypasses_low_confidence():
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
        region_deferral_counts={0: 0},
        deferral_decisions=decisions,
        force_region_reasons={0: "deferral_window_closed"},
    )
    assert len(committed[0]) == 1
    assert decisions[0]["action"] == "deferral_window_closed_forced"


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


def _stop_protection_fixture():
    regions = build_fixed_regions(4, 4)
    regions[0].schedule_step = 1
    mask_id = 4
    tokens = torch.full((1, 4), mask_id, dtype=torch.long)
    # R0's ordinary entropy-ranked position is position 0 and proposes EOS=3.
    # Position 1 is the strongest non-stop alternative.
    logits = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 8.0, -10.0],
                [7.0, 0.0, 0.0, 0.0, -10.0],
                [1.0, 0.0, 0.0, 0.0, -10.0],
                [0.0, 0.0, 0.0, 0.0, -10.0],
            ]
        ],
        dtype=torch.float32,
    )
    return regions, mask_id, tokens, logits


def test_stop_filter_backfills_with_best_non_stop_candidate():
    regions, mask_id, tokens, logits = _stop_protection_fixture()
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
        force_region_reasons={0: "gap"},
        stop_protection_mode="filter",
        stop_token_ids={3},
        stop_protected_regions={0},
        stop_protection_decisions=decisions,
    )
    assert committed == {0: [1]}
    assert tokens.tolist() == [[4, 0, 4, 4]]
    assert decisions == [
        {
            "region": 0,
            "action": "stop_filtered",
            "scheduled_quota": 1,
            "committed_quota": 1,
            "stop_candidates": 1,
            "low_confidence_candidates": 0,
            "confidence_threshold": 0.4,
            "hold_schedule": False,
        }
    ]


def test_stop_filter_holds_when_non_stop_candidates_miss_threshold():
    regions, mask_id, tokens, logits = _stop_protection_fixture()
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
        stop_protection_mode="filter",
        stop_token_ids={3},
        stop_protected_regions={0},
        stop_protection_decisions=decisions,
        stop_filter_confidence_threshold=1.0,
    )
    assert committed == {0: []}
    assert tokens.tolist() == [[4, 4, 4, 4]]
    assert decisions[0]["action"] == "stop_filtered_empty"
    assert decisions[0]["hold_schedule"] is True
    assert decisions[0]["low_confidence_candidates"] == 4


def test_stop_defer_beats_gap_force_and_does_not_backfill():
    regions, mask_id, tokens, logits = _stop_protection_fixture()
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
        force_region_reasons={0: "gap"},
        stop_protection_mode="defer",
        stop_token_ids={3},
        stop_protected_regions={0},
        stop_protection_decisions=decisions,
    )
    assert committed == {0: []}
    assert tokens.tolist() == [[4, 4, 4, 4]]
    assert decisions[0]["action"] == "stop_deferred"
    assert decisions[0]["hold_schedule"] is True


def test_stop_defer_allows_stop_after_left_prefix_finishes():
    regions, mask_id, tokens, logits = _stop_protection_fixture()
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
        stop_protection_mode="defer",
        stop_token_ids={3},
        stop_protected_regions=set(),
    )
    assert committed == {0: [0]}
    assert tokens.tolist() == [[3, 4, 4, 4]]


def test_latched_endpoint_prevents_suffix_commitment_inside_same_region():
    regions, mask_id, tokens, logits = _stop_protection_fixture()
    regions[0].schedule_step = 3
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
        max_response_position_exclusive=2,
    )
    assert committed == {0: [0, 1]}
    assert tokens.tolist() == [[3, 0, 4, 4]]


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


def test_summary_reports_measured_throughput_and_vanilla_speedups():
    common = {
        "correct": True,
        "average_tokens_committed_per_forward": 1.0,
        "canvas_tokens": 256,
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
