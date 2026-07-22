"""Configuration dataclass with all pipeline parameters and TOML loading.

Loading priority: CLI args > book.toml > profile defaults > built-in defaults
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Stages that each profile SKIPS (everything not listed runs).
_PROFILE_SKIPS: dict[str, set[str]] = {
    "full": set(),
    "book-only": {"content_area", "staff_extract", "omr"},
    "scores-only": {"ocr"},
    "geometry": {
        "content_area", "staff_extract", "dewarp", "deskew",
        "enhance", "normalize", "ocr", "omr",
    },
    "clean": {"ocr", "omr"},
    "quick": {
        "content_area", "staff_extract", "dewarp", "deskew",
        "normalize", "ocr", "omr",
    },
}

_MANDATORY_STAGES = {"orientation", "page_detect", "perspective", "pdf_assembly"}


@dataclass
class Config:
    """Pipeline configuration with all stage parameters.

    Construct directly for programmatic use, or via ``from_toml()`` to load
    from a ``book.toml`` file with optional CLI overrides.
    """

    input_dir: Path
    output_dir: Path | None = None
    profile: str = "full"
    book_only: bool = False
    scores_only: bool = False
    preview: int = 0
    use_gpu: bool = True
    ai_dewarp: bool = False
    binarize: bool = False

    # Stage skip overrides (optional stages only)
    skip_content_area: bool = False
    skip_dewarp: bool = False
    skip_deskew: bool = False
    skip_enhance: bool = False
    skip_normalize: bool = False
    skip_ocr: bool = False

    # Error handling, cleanup, logging
    on_error: str = "skip"
    cleanup: bool = False
    keep_stages: list[str] | None = None
    verbose: bool = False
    quiet: bool = False
    minimize_diskspace: bool = False

    # Book characteristics (from book.toml via analyze)
    staff_color_hue: int = 5
    staff_color_range: int = 15
    staff_saturation_min: int = 40
    staff_value_min: int = 80
    channel_diff_rg: int = 30
    channel_diff_rb: int = 30
    has_border_frame: bool = True
    border_ink_matches_staff: bool = True
    page_number_position: str = "top-right"
    expected_staff_lines: int = 16
    has_illustrations: bool = False
    illustration_frequency: str = "none"
    median_aspect_ratio: float = 0.0

    # Stitch parameters (Stage 1)
    stitch_min_matches: int = 30
    stitch_ratio_threshold: float = 0.75
    stitch_min_overlap_frac: float = 0.15
    stitch_inlier_ratio: float = 0.5
    retake_overlap_threshold: float = 0.9

    # Page overrides (manual stitch control)
    stitch_groups: list[list[str]] | None = None
    exclude_images: list[str] | None = None
    no_stitch_images: list[str] | None = None
    include_covers: bool = False

    # Page detection (Stage 4)
    page_detect_method: str = "auto"
    page_detect_morph_kernel: int = 50
    page_detect_epsilon: float = 0.02
    page_detect_min_area_frac: float = 0.30
    page_detect_padding: int = 10
    page_detect_expand_frac: float = 0.03

    # Content area detection (Stage 6)
    content_detect_inset_fallback: float = 0.05
    content_margin_padding: float = 0.0
    content_feather_sigma: int = 20

    # Perspective validation (Stage 5)
    perspective_max_skew_deg: float = 5.0
    perspective_max_crop_frac: float = 0.30
    perspective_output_padding_frac: float = 0.02

    # Deskew (Stage 8)
    deskew_max_angle: float = 5.0
    deskew_angle_step: float = 0.1
    deskew_skip_threshold: float = 0.1

    # PDF assembly (Stage 15)
    pdf_compression: str = "jpeg"
    pdf_jpeg_quality: int = 90
    pdf_dpi: int = 300

    # Flipbook
    flipbook_max_width: int = 1600
    flipbook_jpeg_quality: int = 85
    flipbook_title: str = ""

    # Photography / condition
    has_flash_hotspots: bool = False
    fingers_detected: bool = False
    lens_distortion_k1: float = 0.0
    lens_distortion_k2: float = 0.0
    color_cast_detected: str = "none"
    background_contrast: str = "dark_on_light"
    shadow_severity: str = "none"
    coarse_rotation_offset: int = 0

    # Physical condition (severity: "none", "mild"/"slight", "moderate", "severe")
    stain_severity: str = "none"
    ink_fading: str = "none"
    show_through_severity: str = "none"
    foxing_severity: str = "none"
    iron_gall_halos: str = "none"
    salt_deposits: str = "none"

    # OCR
    ocr_engine: str = "tesseract"
    ocr_lang: str = "lat"

    # OMR (Stage 14)
    omr_model_dir: str = ""
    omr_beam_width: int = 1
    omr_device: str = "AUTO"
    skip_omr: bool = False

    # Branch-specific overrides (private; populated by from_toml)
    _branch_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Enhance sub-step toggles
    enhance_color_cast: bool = True
    enhance_illumination: bool = True
    enhance_shadow: bool = True
    enhance_stain: bool = True
    enhance_halo: bool = True
    enhance_show_through: bool = True
    enhance_white_balance: bool = True
    enhance_clahe: bool = True
    enhance_salt: bool = True
    enhance_denoise: bool = True
    enhance_sharpen: bool = True

    def __post_init__(self):
        self.input_dir = Path(self.input_dir)
        if self.output_dir is None:
            self.output_dir = self.input_dir.parent / f"{self.input_dir.name}_output"
        else:
            self.output_dir = Path(self.output_dir)

    def for_branch(self, branch_name: str) -> Config:
        """Return a copy with branch-specific overrides merged in.

        Branch overrides come from TOML sections like [book.deskew] or
        [score.enhance], flattened to attribute names like deskew_max_angle.
        """
        import dataclasses

        overrides = self._branch_overrides.get(branch_name, {})
        copy = dataclasses.replace(self)
        for attr_name, value in overrides.items():
            if hasattr(copy, attr_name):
                setattr(copy, attr_name, value)
            else:
                logger.warning("Unknown branch override: %s.%s", branch_name, attr_name)
        return copy

    def should_skip_stage(self, stage_name: str) -> bool:
        """Return True if a stage should be skipped based on profile + explicit flags.

        Mandatory stages can never be skipped.
        """
        if stage_name in _MANDATORY_STAGES:
            return False

        # Explicit skip flag takes precedence
        skip_attr = f"skip_{stage_name}"
        if hasattr(self, skip_attr) and getattr(self, skip_attr):
            return True

        # Profile-based skipping
        profile_skips = _PROFILE_SKIPS.get(self.profile, set())
        return stage_name in profile_skips

    @classmethod
    def from_toml(
        cls,
        input_dir: str | Path,
        toml_path: str | Path | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> Config:
        """Load config from a book.toml file, with optional CLI overrides.

        Missing file is silently ignored (all defaults used).
        """
        toml_data: dict[str, Any] = {}
        toml_path = Path(toml_path) if toml_path else None

        if toml_path and toml_path.exists():
            with open(toml_path, "rb") as f:
                toml_data = tomllib.load(f)

        kwargs: dict[str, Any] = {"input_dir": Path(input_dir)}

        # [ink] section
        ink = toml_data.get("ink", {})
        _map_if_present(kwargs, ink, "staff_color_hue", "staff_color_hue")
        _map_if_present(kwargs, ink, "staff_color_range", "staff_color_range")
        _map_if_present(kwargs, ink, "staff_saturation_min", "staff_saturation_min")
        _map_if_present(kwargs, ink, "staff_value_min", "staff_value_min")
        _map_if_present(kwargs, ink, "channel_diff_rg", "channel_diff_rg")
        _map_if_present(kwargs, ink, "channel_diff_rb", "channel_diff_rb")

        # [layout] section
        layout = toml_data.get("layout", {})
        _map_if_present(kwargs, layout, "has_border_frame", "has_border_frame")
        _map_if_present(kwargs, layout, "border_ink_matches_staff", "border_ink_matches_staff")
        _map_if_present(kwargs, layout, "page_number_position", "page_number_position")
        _map_if_present(kwargs, layout, "expected_staff_lines_per_page", "expected_staff_lines")
        _map_if_present(kwargs, layout, "has_illustrations", "has_illustrations")
        _map_if_present(kwargs, layout, "illustration_frequency", "illustration_frequency")
        _map_if_present(kwargs, layout, "median_aspect_ratio", "median_aspect_ratio")

        # [pipeline] section
        pipeline = toml_data.get("pipeline", {})
        _map_if_present(kwargs, pipeline, "profile", "profile")
        _map_if_present(kwargs, pipeline, "skip_content_area", "skip_content_area")
        _map_if_present(kwargs, pipeline, "skip_dewarp", "skip_dewarp")
        _map_if_present(kwargs, pipeline, "skip_deskew", "skip_deskew")
        _map_if_present(kwargs, pipeline, "skip_enhance", "skip_enhance")
        _map_if_present(kwargs, pipeline, "skip_normalize", "skip_normalize")
        _map_if_present(kwargs, pipeline, "skip_ocr", "skip_ocr")
        _map_if_present(kwargs, pipeline, "minimize_diskspace", "minimize_diskspace")

        # [enhance] section
        enhance = toml_data.get("enhance", {})
        _ENHANCE_MAP = {
            "color_cast_correction": "enhance_color_cast",
            "illumination_normalization": "enhance_illumination",
            "shadow_correction": "enhance_shadow",
            "stain_correction": "enhance_stain",
            "halo_reduction": "enhance_halo",
            "show_through_removal": "enhance_show_through",
            "white_balance": "enhance_white_balance",
            "clahe": "enhance_clahe",
            "salt_correction": "enhance_salt",
            "denoise": "enhance_denoise",
            "sharpen": "enhance_sharpen",
        }
        for toml_key, attr_name in _ENHANCE_MAP.items():
            _map_if_present(kwargs, enhance, toml_key, attr_name)

        # [ocr] section
        ocr = toml_data.get("ocr", {})
        _map_if_present(kwargs, ocr, "language", "ocr_lang")
        _map_if_present(kwargs, ocr, "engine", "ocr_engine")

        # [omr] section
        omr = toml_data.get("omr", {})
        _map_if_present(kwargs, omr, "model_dir", "omr_model_dir")
        _map_if_present(kwargs, omr, "beam_width", "omr_beam_width")
        _map_if_present(kwargs, omr, "device", "omr_device")

        # [stitch] section
        stitch = toml_data.get("stitch", {})
        _map_if_present(kwargs, stitch, "min_matches", "stitch_min_matches")
        _map_if_present(kwargs, stitch, "ratio_threshold", "stitch_ratio_threshold")
        _map_if_present(kwargs, stitch, "min_overlap_frac", "stitch_min_overlap_frac")
        _map_if_present(kwargs, stitch, "inlier_ratio", "stitch_inlier_ratio")
        _map_if_present(kwargs, stitch, "retake_overlap_threshold", "retake_overlap_threshold")

        # [page_overrides] section
        overrides_section = toml_data.get("page_overrides", {})
        _map_if_present(kwargs, overrides_section, "stitch_groups", "stitch_groups")
        _map_if_present(kwargs, overrides_section, "exclude", "exclude_images")
        _map_if_present(kwargs, overrides_section, "no_stitch", "no_stitch_images")
        _map_if_present(kwargs, overrides_section, "include_covers", "include_covers")

        # [page_detect] section
        page_detect = toml_data.get("page_detect", {})
        _map_if_present(kwargs, page_detect, "method", "page_detect_method")
        _map_if_present(kwargs, page_detect, "morph_kernel", "page_detect_morph_kernel")
        _map_if_present(kwargs, page_detect, "epsilon", "page_detect_epsilon")
        _map_if_present(kwargs, page_detect, "min_area_frac", "page_detect_min_area_frac")
        _map_if_present(kwargs, page_detect, "padding", "page_detect_padding")
        _map_if_present(kwargs, page_detect, "expand_frac", "page_detect_expand_frac")

        # [content_area] section
        content = toml_data.get("content_area", {})
        _map_if_present(kwargs, content, "inset_fallback", "content_detect_inset_fallback")
        _map_if_present(kwargs, content, "margin_padding", "content_margin_padding")
        _map_if_present(kwargs, content, "feather_sigma", "content_feather_sigma")

        # [perspective] section
        perspective = toml_data.get("perspective", {})
        _map_if_present(kwargs, perspective, "max_skew_deg", "perspective_max_skew_deg")
        _map_if_present(kwargs, perspective, "max_crop_frac", "perspective_max_crop_frac")

        # [deskew] section
        deskew = toml_data.get("deskew", {})
        _map_if_present(kwargs, deskew, "max_angle", "deskew_max_angle")
        _map_if_present(kwargs, deskew, "angle_step", "deskew_angle_step")
        _map_if_present(kwargs, deskew, "skip_threshold", "deskew_skip_threshold")

        # [pdf] section
        pdf = toml_data.get("pdf", {})
        _map_if_present(kwargs, pdf, "compression", "pdf_compression")
        _map_if_present(kwargs, pdf, "jpeg_quality", "pdf_jpeg_quality")
        _map_if_present(kwargs, pdf, "dpi", "pdf_dpi")
        if "pdf_compression" in kwargs:
            kwargs["pdf_compression"] = str(kwargs["pdf_compression"]).lower()

        # [flipbook] section
        flipbook = toml_data.get("flipbook", {})
        _map_if_present(kwargs, flipbook, "max_width", "flipbook_max_width")
        _map_if_present(kwargs, flipbook, "jpeg_quality", "flipbook_jpeg_quality")
        _map_if_present(kwargs, flipbook, "title", "flipbook_title")

        # [photography] section
        photography = toml_data.get("photography", {})
        _map_if_present(kwargs, photography, "has_flash_hotspots", "has_flash_hotspots")
        _map_if_present(kwargs, photography, "fingers_detected", "fingers_detected")
        _map_if_present(kwargs, photography, "lens_distortion_k1", "lens_distortion_k1")
        _map_if_present(kwargs, photography, "lens_distortion_k2", "lens_distortion_k2")
        _map_if_present(kwargs, photography, "color_cast_detected", "color_cast_detected")
        _map_if_present(kwargs, photography, "background_contrast", "background_contrast")
        _map_if_present(kwargs, photography, "shadow_severity", "shadow_severity")
        _map_if_present(kwargs, photography, "coarse_rotation_offset", "coarse_rotation_offset")

        # [condition] section
        condition = toml_data.get("condition", {})
        _map_if_present(kwargs, condition, "stain_severity", "stain_severity")
        _map_if_present(kwargs, condition, "ink_fading", "ink_fading")
        _map_if_present(kwargs, condition, "show_through_severity", "show_through_severity")
        _map_if_present(kwargs, condition, "foxing_severity", "foxing_severity")
        _map_if_present(kwargs, condition, "iron_gall_halos", "iron_gall_halos")
        _map_if_present(kwargs, condition, "salt_deposits", "salt_deposits")

        # Branch-specific overrides: [book.*] and [score.*]
        branch_overrides: dict[str, dict[str, Any]] = {}
        for branch_name in ("book", "score"):
            branch_data = toml_data.get(branch_name, {})
            if isinstance(branch_data, dict):
                flat: dict[str, Any] = {}
                for section_name, section_vals in branch_data.items():
                    if isinstance(section_vals, dict):
                        for key, val in section_vals.items():
                            flat[f"{section_name}_{key}"] = val
                if flat:
                    branch_overrides[branch_name] = flat
        if branch_overrides:
            kwargs["_branch_overrides"] = branch_overrides

        # CLI overrides take top priority
        if overrides:
            kwargs.update(overrides)

        return cls(**kwargs)


def _map_if_present(
    target: dict[str, Any],
    source: dict[str, Any],
    source_key: str,
    target_key: str,
) -> None:
    """Copy a value from source dict to target dict if the key exists."""
    if source_key in source:
        target[target_key] = source[source_key]
