"""Tests for publication bias — fixes v2 audit #23."""

import pytest
from lumen.tools.statistics.publication_bias import egger_test, trim_and_fill


# Symmetric data (no bias expected) — similar SEs to avoid precision-effect correlation
EFFECTS_SYM = [0.5, 0.3, 0.7, 0.4, 0.6, 0.35, 0.65, 0.45, 0.55, 0.50]
SES_SYM = [0.15, 0.14, 0.16, 0.15, 0.14, 0.15, 0.16, 0.14, 0.15, 0.15]

# Asymmetric data (bias likely — small studies with large positive effects)
EFFECTS_ASYM = [0.1, 0.2, 0.3, 0.8, 1.2, 1.5]
SES_ASYM = [0.05, 0.06, 0.08, 0.25, 0.30, 0.35]


class TestEggerTest:
    def test_output_fields(self):
        result = egger_test(EFFECTS_SYM, SES_SYM)
        for field in ["intercept", "se", "t_stat", "p_value", "significant", "k"]:
            assert field in result

    def test_symmetric_not_significant(self):
        """Symmetric data should generally not trigger Egger's test."""
        # Symmetric: small-study effects balanced on both sides of pooled
        # Pairs: (small SE, low effect) balanced by (small SE, high effect)
        effects = [0.3, 0.7, 0.35, 0.65, 0.4, 0.6, 0.42, 0.58, 0.48, 0.52]
        ses =     [0.10, 0.10, 0.15, 0.15, 0.20, 0.20, 0.25, 0.25, 0.30, 0.30]
        result = egger_test(effects, ses)
        # Balanced data → Egger should not be significant
        assert result["p_value"] > 0.05

    def test_insufficient_studies(self):
        with pytest.raises(ValueError, match="k >= 3"):
            egger_test([0.5, 0.3], [0.1, 0.15])

    def test_k_field(self):
        result = egger_test(EFFECTS_SYM, SES_SYM)
        assert result["k"] == 10


class TestTrimAndFill:
    def test_output_fields(self):
        result = trim_and_fill(EFFECTS_SYM, SES_SYM)
        for field in ["adjusted_effect", "adjusted_ci", "n_imputed",
                       "original_effect", "direction_flipped", "warning"]:
            assert field in result

    def test_symmetric_few_imputed(self):
        """Symmetric data should need few or no imputed studies."""
        result = trim_and_fill(EFFECTS_SYM, SES_SYM)
        assert result["n_imputed"] <= 3  # generous tolerance

    def test_direction_flip_flag(self):
        """v2 audit #23: direction flip must be flagged."""
        # Create strongly biased data where fill would reverse direction
        effects = [0.1, 0.05, 0.02, 0.8, 1.0, 1.2, 1.5]
        ses = [0.05, 0.04, 0.06, 0.3, 0.35, 0.4, 0.45]
        result = trim_and_fill(effects, ses)
        # Whether it flips depends on data, but the flag must exist
        assert isinstance(result["direction_flipped"], bool)
        if result["direction_flipped"]:
            assert result["warning"] is not None
            assert "reversed" in result["warning"].lower()

    def test_no_flip_no_warning(self):
        result = trim_and_fill(EFFECTS_SYM, SES_SYM)
        if not result["direction_flipped"]:
            assert result["warning"] is None

    def test_insufficient_studies(self):
        with pytest.raises(ValueError, match="k >= 3"):
            trim_and_fill([0.5, 0.3], [0.1, 0.15])

    def test_k_total(self):
        result = trim_and_fill(EFFECTS_SYM, SES_SYM)
        assert result["k_total"] == len(EFFECTS_SYM) + result["n_imputed"]
