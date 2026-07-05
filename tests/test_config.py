"""TDD tests for ghh.config -- Config dataclass and TOML loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from ghh.config import Config


class TestConfigDefaults:
    """Tests for Config with default values."""

    def test_creates_with_input_dir_only(self, tmp_path):
        cfg = Config(input_dir=tmp_path)
        assert cfg.input_dir == tmp_path

    def test_default_output_dir(self, tmp_path):
        input_dir = tmp_path / "LPA 1"
        input_dir.mkdir()
        cfg = Config(input_dir=input_dir)
        expected = tmp_path / "LPA 1_output"
        assert cfg.output_dir == expected

    def test_explicit_output_dir(self, tmp_path):
        cfg = Config(input_dir=tmp_path, output_dir=tmp_path / "custom_output")
        assert cfg.output_dir == tmp_path / "custom_output"

    def test_default_profile_is_full(self, tmp_path):
        cfg = Config(input_dir=tmp_path)
        assert cfg.profile == "full"

    def test_default_skip_flags_are_false(self, tmp_path):
        cfg = Config(input_dir=tmp_path)
        assert cfg.skip_dewarp is False
        assert cfg.skip_deskew is False
        assert cfg.skip_enhance is False
        assert cfg.skip_normalize is False
        assert cfg.skip_ocr is False
        assert cfg.skip_content_area is False

    def test_default_ink_parameters(self, tmp_path):
        cfg = Config(input_dir=tmp_path)
        assert cfg.staff_color_hue == 5
        assert cfg.staff_color_range == 15
        assert cfg.staff_saturation_min == 40
        assert cfg.staff_value_min == 80

    def test_default_ocr_engine(self, tmp_path):
        cfg = Config(input_dir=tmp_path)
        assert cfg.ocr_engine == "tesseract"
        assert cfg.ocr_lang == "lat"

    def test_default_on_error(self, tmp_path):
        cfg = Config(input_dir=tmp_path)
        assert cfg.on_error == "skip"


class TestConfigFromTOML:
    """Tests for Config.from_toml() loading."""

    def test_loads_ink_settings(self, tmp_path):
        toml_path = tmp_path / "book.toml"
        toml_path.write_text("""\
[ink]
staff_color_hue = 120
staff_color_range = 20
staff_saturation_min = 50
staff_value_min = 90
channel_diff_rg = 40
channel_diff_rb = 40
""")
        cfg = Config.from_toml(input_dir=tmp_path, toml_path=toml_path)
        assert cfg.staff_color_hue == 120
        assert cfg.staff_color_range == 20
        assert cfg.staff_saturation_min == 50

    def test_loads_pipeline_profile(self, tmp_path):
        toml_path = tmp_path / "book.toml"
        toml_path.write_text("""\
[pipeline]
profile = "geometry"
""")
        cfg = Config.from_toml(input_dir=tmp_path, toml_path=toml_path)
        assert cfg.profile == "geometry"

    def test_loads_skip_flags(self, tmp_path):
        toml_path = tmp_path / "book.toml"
        toml_path.write_text("""\
[pipeline]
skip_dewarp = true
skip_ocr = true
""")
        cfg = Config.from_toml(input_dir=tmp_path, toml_path=toml_path)
        assert cfg.skip_dewarp is True
        assert cfg.skip_ocr is True
        assert cfg.skip_deskew is False  # not set, should be default

    def test_loads_enhance_sub_steps(self, tmp_path):
        toml_path = tmp_path / "book.toml"
        toml_path.write_text("""\
[enhance]
color_cast_correction = false
denoise = false
""")
        cfg = Config.from_toml(input_dir=tmp_path, toml_path=toml_path)
        assert cfg.enhance_color_cast is False
        assert cfg.enhance_denoise is False
        assert cfg.enhance_sharpen is True  # default

    def test_loads_ocr_settings(self, tmp_path):
        toml_path = tmp_path / "book.toml"
        toml_path.write_text("""\
[ocr]
language = "deu"
""")
        cfg = Config.from_toml(input_dir=tmp_path, toml_path=toml_path)
        assert cfg.ocr_lang == "deu"

    def test_cli_overrides_toml(self, tmp_path):
        toml_path = tmp_path / "book.toml"
        toml_path.write_text("""\
[pipeline]
profile = "full"
skip_dewarp = false
""")
        cfg = Config.from_toml(
            input_dir=tmp_path,
            toml_path=toml_path,
            overrides={"profile": "geometry", "skip_dewarp": True},
        )
        assert cfg.profile == "geometry"
        assert cfg.skip_dewarp is True

    def test_missing_toml_uses_defaults(self, tmp_path):
        cfg = Config.from_toml(
            input_dir=tmp_path,
            toml_path=tmp_path / "nonexistent.toml",
        )
        assert cfg.profile == "full"
        assert cfg.staff_color_hue == 5


class TestConfigProfiles:
    """Tests for profile-based stage skipping."""

    def test_geometry_profile_skips_enhance(self, tmp_path):
        cfg = Config(input_dir=tmp_path, profile="geometry")
        assert cfg.should_skip_stage("enhance") is True
        assert cfg.should_skip_stage("normalize") is True
        assert cfg.should_skip_stage("ocr") is True

    def test_geometry_profile_keeps_mandatory(self, tmp_path):
        cfg = Config(input_dir=tmp_path, profile="geometry")
        assert cfg.should_skip_stage("orientation") is False
        assert cfg.should_skip_stage("page_detect") is False
        assert cfg.should_skip_stage("perspective") is False
        assert cfg.should_skip_stage("pdf_assembly") is False

    def test_clean_profile_skips_ocr(self, tmp_path):
        cfg = Config(input_dir=tmp_path, profile="clean")
        assert cfg.should_skip_stage("ocr") is True
        assert cfg.should_skip_stage("enhance") is False

    def test_quick_profile_skips_dewarp_and_ocr(self, tmp_path):
        cfg = Config(input_dir=tmp_path, profile="quick")
        assert cfg.should_skip_stage("dewarp") is True
        assert cfg.should_skip_stage("ocr") is True

    def test_full_profile_skips_nothing_optional(self, tmp_path):
        cfg = Config(input_dir=tmp_path, profile="full")
        for stage in ("content_area", "dewarp", "deskew", "enhance", "normalize", "ocr"):
            assert cfg.should_skip_stage(stage) is False

    def test_explicit_skip_overrides_profile(self, tmp_path):
        cfg = Config(input_dir=tmp_path, profile="full", skip_dewarp=True)
        assert cfg.should_skip_stage("dewarp") is True


class TestConfigLayoutFields:
    """Tests for [layout] section fields in Config."""

    def test_default_layout_values(self, tmp_path):
        cfg = Config(input_dir=tmp_path)
        assert cfg.border_ink_matches_staff is True
        assert cfg.has_illustrations is False
        assert cfg.illustration_frequency == "none"
        assert cfg.median_aspect_ratio == 0.0

    def test_loads_layout_from_toml(self, tmp_path):
        toml_path = tmp_path / "book.toml"
        toml_path.write_text("""\
[layout]
has_border_frame = false
border_ink_matches_staff = false
page_number_position = "bottom"
expected_staff_lines_per_page = 20
has_illustrations = true
illustration_frequency = "frequent"
median_aspect_ratio = 1.5
""")
        cfg = Config.from_toml(input_dir=tmp_path, toml_path=toml_path)
        assert cfg.has_border_frame is False
        assert cfg.border_ink_matches_staff is False
        assert cfg.page_number_position == "bottom"
        assert cfg.expected_staff_lines == 20
        assert cfg.has_illustrations is True
        assert cfg.illustration_frequency == "frequent"
        assert cfg.median_aspect_ratio == 1.5


class TestConfigConditionFields:
    """Tests for [condition] section fields in Config."""

    def test_default_condition_values(self, tmp_path):
        cfg = Config(input_dir=tmp_path)
        assert cfg.stain_severity == "none"
        assert cfg.ink_fading == "none"
        assert cfg.show_through_severity == "none"
        assert cfg.foxing_severity == "none"
        assert cfg.iron_gall_halos == "none"
        assert cfg.salt_deposits == "none"

    def test_loads_condition_from_toml(self, tmp_path):
        toml_path = tmp_path / "book.toml"
        toml_path.write_text("""\
[condition]
stain_severity = "moderate"
ink_fading = "slight"
show_through_severity = "mild"
foxing_severity = "severe"
iron_gall_halos = "moderate"
salt_deposits = "mild"
""")
        cfg = Config.from_toml(input_dir=tmp_path, toml_path=toml_path)
        assert cfg.stain_severity == "moderate"
        assert cfg.ink_fading == "slight"
        assert cfg.show_through_severity == "mild"
        assert cfg.foxing_severity == "severe"
        assert cfg.iron_gall_halos == "moderate"
        assert cfg.salt_deposits == "mild"


class TestConfigPhotographyExtended:
    """Tests for extended [photography] fields in Config."""

    def test_default_extended_photography_values(self, tmp_path):
        cfg = Config(input_dir=tmp_path)
        assert cfg.color_cast_detected == "none"
        assert cfg.background_contrast == "dark_on_light"
        assert cfg.shadow_severity == "none"
        assert cfg.coarse_rotation_offset == 0

    def test_loads_extended_photography_from_toml(self, tmp_path):
        toml_path = tmp_path / "book.toml"
        toml_path.write_text("""\
[photography]
color_cast_detected = "slight_warm"
background_contrast = "light_on_dark"
shadow_severity = "moderate"
coarse_rotation_offset = 90
""")
        cfg = Config.from_toml(input_dir=tmp_path, toml_path=toml_path)
        assert cfg.color_cast_detected == "slight_warm"
        assert cfg.background_contrast == "light_on_dark"
        assert cfg.shadow_severity == "moderate"
        assert cfg.coarse_rotation_offset == 90
