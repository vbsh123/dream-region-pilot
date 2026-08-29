from __future__ import annotations

from dataclasses import dataclass

from .regions import Region


@dataclass
class RegionScheduler:
    regions: list[Region]
    mode: str
    max_active_regions: int | None = None
    spawn_readiness: float = 0.15
    max_progress_gap: int = 8

    def __post_init__(self) -> None:
        valid = {
            "fixed_sequential",
            "always_on",
            "always_on_tail_guard",
            "always_on_dawn_tail_guard",
            "always_on_bounded_defer",
            "always_on_bounded_defer_tail_guard",
            "always_on_coupled_defer",
            "always_on_coupled_defer_tail_guard",
            "always_on_coupled_defer_stop_filter",
            "always_on_coupled_defer_stop_defer",
            "always_on_coupled_defer_dawn_tail_guard",
            "loose_wavefront",
            "flowblock_proxy",
            "controlled_position",
            "controlled_position_tail_guard",
        }
        if self.mode not in valid:
            raise ValueError(f"mode must be one of {sorted(valid)}")
        if not 0.0 <= self.spawn_readiness <= 1.0:
            raise ValueError("spawn_readiness must be in [0, 1]")
        if self.max_active_regions is None:
            self.max_active_regions = len(self.regions)
        if self.max_active_regions <= 0:
            raise ValueError("max_active_regions must be positive")
        if self.max_progress_gap < 0:
            raise ValueError("max_progress_gap must be non-negative")
        self.admitted_count = 1 if self.is_wavefront else len(self.regions)
        self.last_control_edges: set[tuple[int, int]] = set()
        self.last_blocked_regions: set[int] = set()
        self.last_urgent_regions: set[int] = set()

    @property
    def is_wavefront(self) -> bool:
        return self.mode in {
            "loose_wavefront",
            "flowblock_proxy",
            "controlled_position",
            "controlled_position_tail_guard",
        }

    @property
    def is_controlled(self) -> bool:
        return self.mode.startswith("controlled_") or self.mode in {
            "always_on_coupled_defer",
            "always_on_coupled_defer_tail_guard",
            "always_on_coupled_defer_stop_filter",
            "always_on_coupled_defer_stop_defer",
            "always_on_coupled_defer_dawn_tail_guard",
        }

    def revealed_tokens(self, region: Region) -> int:
        return len(region.token_indices) - len(region.remaining_mask_indices)

    def regions_allowed_to_advance(
        self,
        local_steps: int,
        *,
        max_region_exclusive: int | None = None,
        progress_gap_exempt_children: set[int] | None = None,
    ) -> list[Region]:
        unfinished = [
            region
            for region in self.regions
            if not region.done and region.schedule_step < local_steps
            and (
                max_region_exclusive is None
                or region.index < max_region_exclusive
            )
        ]
        if self.mode in {
            "always_on",
            "always_on_tail_guard",
            "always_on_dawn_tail_guard",
            "always_on_bounded_defer",
            "always_on_bounded_defer_tail_guard",
        }:
            return unfinished
        if self.mode in {
            "always_on_coupled_defer",
            "always_on_coupled_defer_tail_guard",
            "always_on_coupled_defer_stop_filter",
            "always_on_coupled_defer_stop_defer",
            "always_on_coupled_defer_dawn_tail_guard",
        }:
            return self._controlled_regions(
                unfinished,
                progress_gap_exempt_children=progress_gap_exempt_children,
            )
        if self.mode == "fixed_sequential":
            return unfinished[:1]
        if self.is_wavefront:
            admitted = [
                region for region in unfinished if region.index < self.admitted_count
            ]
            if not self.is_controlled:
                return admitted
            return self._controlled_regions(
                admitted,
                progress_gap_exempt_children=progress_gap_exempt_children,
            )

        raise AssertionError(f"Unhandled scheduler mode {self.mode!r}")

    def _controlled_regions(
        self,
        admitted: list[Region],
        *,
        progress_gap_exempt_children: set[int] | None = None,
    ) -> list[Region]:
        by_index = {region.index: region for region in self.regions}
        admitted_indices = {region.index for region in admitted}
        # Immediate positional neighbors provide the initial loose pipeline.
        positional_edges = {
            (index, index + 1)
            for index in range(self.admitted_count - 1)
            if index in admitted_indices and index + 1 in admitted_indices
        }
        control_edges = positional_edges
        blocked: set[int] = set()
        urgent: set[int] = set()
        for parent_index, child_index in sorted(control_edges):
            parent = by_index[parent_index]
            child = by_index[child_index]
            if parent.done or child.done:
                continue
            parent_progress = self.revealed_tokens(parent)
            child_progress = self.revealed_tokens(child)
            if child_progress > parent_progress:
                blocked.add(child_index)
                urgent.add(parent_index)
            else:
                parent_lead = parent_progress - child_progress
                coupled_mode = self.mode in {
                    "always_on_coupled_defer",
                    "always_on_coupled_defer_tail_guard",
                    "always_on_coupled_defer_stop_filter",
                    "always_on_coupled_defer_stop_defer",
                    "always_on_coupled_defer_dawn_tail_guard",
                }
                lead_limit_reached = (
                    parent_lead >= max(1, self.max_progress_gap)
                    if coupled_mode
                    else parent_lead > self.max_progress_gap
                )
                stop_guard_exempt = child_index in (
                    progress_gap_exempt_children or set()
                )
                if lead_limit_reached and not stop_guard_exempt:
                    blocked.add(parent_index)
                    urgent.add(child_index)
        self.last_control_edges = control_edges
        self.last_blocked_regions = blocked
        self.last_urgent_regions = urgent - blocked
        return [region for region in admitted if region.index not in blocked]

    def maybe_admit_next(self, readiness_by_region: dict[int, float]) -> list[int]:
        """Admit at most one new positional region for the next forward.

        The gate is evaluated on the rightmost already-admitted region. Limiting
        admission to one region per global iteration preserves the intended
        staggered A, AB, ABC wavefront even though ordinary Dream returns logits
        for the entire masked canvas in every forward.
        """
        if not self.is_wavefront or self.admitted_count >= len(self.regions):
            return []
        active = sum(
            not region.done for region in self.regions[: self.admitted_count]
        )
        if active >= int(self.max_active_regions):
            return []
        frontier = self.regions[self.admitted_count - 1]
        readiness = 1.0 if frontier.done else readiness_by_region.get(frontier.index, 0.0)
        if readiness < self.spawn_readiness:
            return []
        admitted = self.admitted_count
        self.admitted_count += 1
        return [admitted]

    @staticmethod
    def apply_updates(
        regions: list[Region], committed: dict[int, list[int]]
    ) -> list[Region]:
        """Advance schedule cursors, but only clock real commitment progress."""
        advanced = []
        for region in regions:
            region.schedule_step += 1
            if committed.get(region.index):
                region.clock += 1
                advanced.append(region)
        return advanced


def parse_strategy(name: str) -> str:
    if name in {
        "fixed_sequential",
        "always_on",
        "always_on_tail_guard",
        "always_on_dawn_tail_guard",
        "always_on_bounded_defer",
        "always_on_bounded_defer_tail_guard",
        "always_on_coupled_defer",
        "always_on_coupled_defer_tail_guard",
        "always_on_coupled_defer_stop_filter",
        "always_on_coupled_defer_stop_defer",
        "always_on_coupled_defer_dawn_tail_guard",
        "loose_wavefront",
        "flowblock_proxy",
        "controlled_position",
        "controlled_position_tail_guard",
    }:
        return name
    raise ValueError(f"Unknown regional strategy {name!r}")
