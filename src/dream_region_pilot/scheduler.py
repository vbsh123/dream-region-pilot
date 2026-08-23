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

    def __post_init__(self) -> None:
        if self.lag < 0:
            raise ValueError("lag must be non-negative")
        valid = {"fixed_sequential", "always_on", "fixed_lag", "wavefront_probe"}
        if self.mode not in valid:
            raise ValueError(f"mode must be one of {sorted(valid)}")
        if not 0.0 <= self.spawn_readiness <= 1.0:
            raise ValueError("spawn_readiness must be in [0, 1]")
        if self.max_active_regions is None:
            self.max_active_regions = len(self.regions)
        if self.max_active_regions <= 0:
            raise ValueError("max_active_regions must be positive")
        self.admitted_count = 1 if self.mode == "wavefront_probe" else len(self.regions)

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
        if self.mode == "wavefront_probe":
            return [
                region for region in unfinished if region.index < self.admitted_count
            ]

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

    def maybe_admit_next(self, readiness_by_region: dict[int, float]) -> list[int]:
        """Admit at most one new positional region for the next forward.

        The gate is evaluated on the rightmost already-admitted region. Limiting
        admission to one region per global iteration preserves the intended
        staggered A, AB, ABC wavefront even though ordinary Dream returns logits
        for the entire masked canvas in every forward.
        """
        if self.mode != "wavefront_probe" or self.admitted_count >= len(self.regions):
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
    if name in {"fixed_sequential", "always_on", "wavefront_probe"}:
        return name, 0
    prefix = "async_lag"
    if name.startswith(prefix):
        suffix = name[len(prefix) :]
        if not suffix.isdigit():
            raise ValueError(f"Invalid asynchronous strategy {name!r}")
        return "fixed_lag", int(suffix)
    raise ValueError(f"Unknown regional strategy {name!r}")
