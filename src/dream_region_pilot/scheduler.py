from __future__ import annotations

from dataclasses import dataclass

from .regions import Region


@dataclass
class RegionScheduler:
    regions: list[Region]
    mode: str
    lag: int = 0
    release_completed_parents: bool = True

    def __post_init__(self) -> None:
        if self.lag < 0:
            raise ValueError("lag must be non-negative")
        valid = {"fixed_sequential", "always_on", "fixed_lag"}
        if self.mode not in valid:
            raise ValueError(f"mode must be one of {sorted(valid)}")

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
    if name in {"fixed_sequential", "always_on"}:
        return name, 0
    prefix = "async_lag"
    if name.startswith(prefix):
        suffix = name[len(prefix) :]
        if not suffix.isdigit():
            raise ValueError(f"Invalid asynchronous strategy {name!r}")
        return "fixed_lag", int(suffix)
    raise ValueError(f"Unknown regional strategy {name!r}")
