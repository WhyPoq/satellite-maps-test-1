"""Validated, serializable experiment settings. No web dependencies."""
from dataclasses import asdict, dataclass, fields
import math


@dataclass(frozen=True)
class PipelineConfig:
    n_orient: int = 8
    n_scale: int = 5
    ul: float = 0.05
    uh: float = 0.4
    sigma: float = 12.0
    n_angular: int = 8
    max_dimension: int = 768
    segmentation_enabled: bool = False
    segmentation_orientation: bool = True
    seg_n_angles: int = 8
    seg_smoothness: float = 0.65
    seg_orientation_weight: float = 1.0
    seg_direction_boost: float = 1.5
    seg_color_sigma: float = 0.15
    seg_variance_floor: float = 0.015
    seg_max_cycles: int = 4
    seg_capacity_scale: int = 10000
    extraction_enabled: bool = False
    net_min_width: int = 10
    net_max_width: int = 60
    net_width_step: int = 4
    net_n_orientations: int = 12
    net_sigma_across: float = 1.5
    net_sigma_along: float = 6.0
    net_curve_source: str = 'tensor'
    net_keep_longest: bool = True

    def __post_init__(self):
        for name in ('segmentation_enabled', 'segmentation_orientation', 'extraction_enabled', 'net_keep_longest'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be true or false.')
        if self.extraction_enabled and not self.segmentation_enabled:
            raise ValueError('Network extraction requires road segmentation to be enabled.')
        if self.net_curve_source not in ('tensor', 'mask'):
            raise ValueError('Road-side input must be tensor or mask.')
        for name, low, high in (('net_min_width', 2, 128), ('net_max_width', 2, 128),
                                ('net_width_step', 1, 32), ('net_n_orientations', 2, 36)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{name} must be an integer between {low} and {high}.')
        if self.net_min_width > self.net_max_width:
            raise ValueError('Minimum road width cannot exceed maximum road width.')
        if len(self.extraction_widths()) > 32:
            raise ValueError('Choose at most 32 road widths by increasing the width step.')
        for name in ('net_sigma_across', 'net_sigma_along'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.25 <= value <= 30:
                raise ValueError(f'{name} must be a finite number between 0.25 and 30 pixels.')
        # Also validate disabled-stage settings so saved experiments stay reproducible.
        self.segmentation_config()
        for name, low, high in (('seg_smoothness', 0, 10), ('seg_orientation_weight', 0, 20),
                                ('seg_direction_boost', 0, 20), ('seg_color_sigma', 0.001, 2),
                                ('seg_variance_floor', 0.00001, 1)):
            if not low <= getattr(self, name) <= high:
                raise ValueError(f'{name} must be between {low} and {high}.')
        for name, low, high in (("n_orient", 3, 16), ("n_scale", 2, 8),
                                ("n_angular", 1, 16), ("max_dimension", 128, 2048)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer between {low} and {high}.")
        for name in ("ul", "uh", "sigma"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
        if not 0.02 <= self.ul < self.uh <= 0.49:
            raise ValueError("Frequencies must satisfy 0.02 <= lower < upper <= 0.49 cycles/pixel.")
        if self.uh / self.ul < 1.25:
            raise ValueError("Upper frequency must be at least 1.25 times the lower frequency.")
        if not 1 <= self.sigma <= 40:
            raise ValueError("Voting sigma must be between 1 and 40 processed-image pixels.")

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError("Settings must be a JSON object.")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
        return cls(**values)

    def to_dict(self):
        return asdict(self)

    def segmentation_config(self):
        from segmentation import Config
        return Config(n_angles=self.seg_n_angles, smoothness=self.seg_smoothness,
                      orientation_weight=self.seg_orientation_weight,
                      direction_boost=self.seg_direction_boost, color_sigma=self.seg_color_sigma,
                      variance_floor=self.seg_variance_floor, max_cycles=self.seg_max_cycles,
                      capacity_scale=self.seg_capacity_scale)

    def extraction_widths(self):
        # Inclusive upper bound, exactly as the supplied script's CLI.
        return list(range(self.net_min_width, self.net_max_width + 1, self.net_width_step))
