from __future__ import annotations

from dataclasses import dataclass

from .regions import Region


@dataclass
class RegionScheduler:
    regions: list[Region]
    mode: str
    lag: int = 0
    release_completed_parents: bool = True
    max_active_regions: int | None = None
    spawn_readiness: float = 0.15
    max_progress_gap: int = 8
    edge_persistence: int = 2

    def __post_init__(self) -> None:
        if self.lag < 0:
            raise ValueError("lag must be non-negative")
        valid = {
            "fixed_sequential",
            "always_on",
            "fixed_lag",
            "wavefront_probe",
            "loose_wavefront",
            "flowblock_proxy",
            "controlled_position",
            "controlled_dapd",
            "controlled_jsd",
            "controlled_combo",
            "controlled_dapd_dynamic",
            "controlled_jsd_dynamic",
            "controlled_combo_dynamic",
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
        if self.edge_persistence <= 0:
            raise ValueError("edge_persistence must be positive")
        self.admitted_count = 1 if self.is_wavefront else len(self.regions)
        self._present_streak: dict[tuple[int, int], int] = {}
        self._missing_streak: dict[tuple[int, int], int] = {}
        self.active_dependency_edges: set[tuple[int, int]] = set()
        self.last_control_edges: set[tuple[int, int]] = set()
        self.last_blocked_regions: set[int] = set()
        self.last_urgent_regions: set[int] = set()

    @property
    def is_wavefront(self) -> bool:
        return self.mode in {
            "wavefront_probe",
            "loose_wavefront",
            "flowblock_proxy",
            "controlled_position",
            "controlled_dapd",
            "controlled_jsd",
            "controlled_combo",
            "controlled_dapd_dynamic",
            "controlled_jsd_dynamic",
            "controlled_combo_dynamic",
        }

    @property
    def is_controlled(self) -> bool:
        return self.mode.startswith("controlled_")

    @property
    def uses_dynamic_commit_groups(self) -> bool:
        return self.mode in {
            "controlled_dapd_dynamic",
            "controlled_jsd_dynamic",
            "controlled_combo_dynamic",
        }

    def revealed_tokens(self, region: Region) -> int:
        return len(region.token_indices) - len(region.remaining_mask_indices)

    def observe_dependency_edges(self, edges: list[dict]) -> None:
        """Apply two-sided persistence after both endpoints have real progress."""
        if not self.is_controlled:
            return
        observed = {
            (min(int(edge["left"]), int(edge["right"])),
             max(int(edge["left"]), int(edge["right"])))
            for edge in edges
        }
        eligible = {
            (left.index, right.index)
            for left in self.regions
            for right in self.regions
            if left.index < right.index
            and left.index < self.admitted_count
            and right.index < self.admitted_count
            and self.revealed_tokens(left) > 0
            and self.revealed_tokens(right) > 0
        }
        for pair in eligible:
            if pair in observed:
                self._present_streak[pair] = self._present_streak.get(pair, 0) + 1
                self._missing_streak[pair] = 0
                if self._present_streak[pair] >= self.edge_persistence:
                    self.active_dependency_edges.add(pair)
            else:
                self._present_streak[pair] = 0
                if pair in self.active_dependency_edges:
                    self._missing_streak[pair] = self._missing_streak.get(pair, 0) + 1
                    if self._missing_streak[pair] >= self.edge_persistence:
                        self.active_dependency_edges.remove(pair)
                        self._missing_streak[pair] = 0

    def set_parents(self, parents: dict[int, set[int]]) -> None:
        for region in self.regions:
            region.parents = set(parents.get(region.index, set()))

    def regions_allowed_to_advance(self, local_steps: int) -> list[Region]:
        unfinished = [
            region
            for region in self.regions
            if not region.done and region.schedule_step < local_steps
        ]
        if self.mode == "always_on":
            return unfinished
        if self.mode == "fixed_sequential":
            return unfinished[:1]
        if self.is_wavefront:
            admitted = [
                region for region in unfinished if region.index < self.admitted_count
            ]
            if not self.is_controlled:
                return admitted
            return self._controlled_regions(admitted)

        allowed = []
        by_index = {region.index: region for region in self.regions}
        for region in unfinished:
            parents = [by_index[parent] for parent in sorted(region.parents)]
            if all(
                (self.release_completed_parents and parent.done)
                or parent.clock >= region.clock + self.lag
                for parent in parents
            ):
                allowed.append(region)
        return allowed

    def _controlled_regions(self, admitted: list[Region]) -> list[Region]:
        by_index = {region.index: region for region in self.regions}
        admitted_indices = {region.index for region in admitted}
        # Immediate positional neighbors provide the initial loose pipeline.
        positional_edges = {
            (index, index + 1)
            for index in range(self.admitted_count - 1)
            if index in admitted_indices and index + 1 in admitted_indices
        }
        dynamic_edges = {
            pair
            for pair in self.active_dependency_edges
            if pair[0] in admitted_indices and pair[1] in admitted_indices
        }
        control_edges = positional_edges | dynamic_edges
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
            elif parent_progress - child_progress > self.max_progress_gap:
                blocked.add(parent_index)
                urgent.add(child_index)
        self.last_control_edges = control_edges
        self.last_blocked_regions = blocked
        self.last_urgent_regions = urgent - blocked
        return [region for region in admitted if region.index not in blocked]

    def commitment_groups(self, active: list[Region]) -> list[list[Region]]:
        """Return dynamic dependency components over this iteration's regions.

        Existing strategies retain one independent commitment pool per region.
        Dynamic strategies pool confidence selection within each connected
        dependency component. Components are rebuilt from the currently active
        persistent graph, so merges and splits do not rewrite committed tokens
        or fabricate clock progress.
        """
        if not self.uses_dynamic_commit_groups:
            return [[region] for region in active]

        by_index = {region.index: region for region in active}
        adjacency = {index: set() for index in by_index}
        for left, right in self.active_dependency_edges:
            if left in adjacency and right in adjacency:
                adjacency[left].add(right)
                adjacency[right].add(left)

        groups: list[list[Region]] = []
        remaining = set(by_index)
        while remaining:
            root = min(remaining)
            stack = [root]
            component: set[int] = set()
            while stack:
                index = stack.pop()
                if index in component:
                    continue
                component.add(index)
                stack.extend(adjacency[index] - component)
            remaining -= component
            groups.append([by_index[index] for index in sorted(component)])
        return groups

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


def parse_strategy(name: str) -> tuple[str, int]:
    if name in {
        "fixed_sequential",
        "always_on",
        "wavefront_probe",
        "loose_wavefront",
        "flowblock_proxy",
        "controlled_position",
        "controlled_dapd",
        "controlled_jsd",
        "controlled_combo",
        "controlled_dapd_dynamic",
        "controlled_jsd_dynamic",
        "controlled_combo_dynamic",
    }:
        return name, 0
    prefix = "async_lag"
    if name.startswith(prefix):
        suffix = name[len(prefix) :]
        if not suffix.isdigit():
            raise ValueError(f"Invalid asynchronous strategy {name!r}")
        return "fixed_lag", int(suffix)
    raise ValueError(f"Unknown regional strategy {name!r}")
