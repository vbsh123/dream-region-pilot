from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Region:
    """Mutable state for one generated-token region.

    Token positions are response-relative; prompt positions never enter a Region.
    """

    index: int
    token_indices: tuple[int, ...]
    parents: set[int] = field(default_factory=set)
    clock: int = 0
    schedule_step: int = 0
    remaining_mask_indices: tuple[int, ...] = field(default_factory=tuple)

    @property
    def done(self) -> bool:
        return not self.remaining_mask_indices


def build_fixed_regions(generation_length: int, region_size: int) -> list[Region]:
    if generation_length <= 0 or region_size <= 0:
        raise ValueError("generation_length and region_size must be positive")
    regions = []
    for start in range(0, generation_length, region_size):
        positions = tuple(range(start, min(start + region_size, generation_length)))
        regions.append(
            Region(
                index=len(regions),
                token_indices=positions,
                remaining_mask_indices=positions,
            )
        )
    return regions


def refresh_remaining_masks(
    regions: list[Region], response_mask: list[bool]
) -> None:
    if len(response_mask) != sum(len(region.token_indices) for region in regions):
        raise ValueError("response_mask length does not match the region canvas")
    for region in regions:
        region.remaining_mask_indices = tuple(
            position for position in region.token_indices if response_mask[position]
        )
