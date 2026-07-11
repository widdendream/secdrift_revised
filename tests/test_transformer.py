"""Tests for prompt transformer."""

import pytest

from secdrift.transformer import (
    PromptTransformer,
    list_sectors,
    get_sector_config,
    validate_transformation,
)


class TestListSectors:
    """Tests for sector listing."""

    def test_list_sectors(self):
        sectors = list_sectors()
        assert isinstance(sectors, list)
        assert "emergency_services" in sectors
        assert "healthcare" in sectors
        assert "government" in sectors
        assert "financial" in sectors


class TestGetSectorConfig:
    """Tests for sector config retrieval."""

    def test_get_sector_config_valid(self):
        config = get_sector_config("emergency_services")
        assert config.id == "emergency_services"
        assert config.name == "Emergency Services"
        assert len(config.terminology) > 0

    def test_get_sector_config_invalid(self):
        with pytest.raises(ValueError, match="Unknown sector"):
            get_sector_config("nonexistent_sector")


class TestPromptTransformer:
    """Tests for PromptTransformer."""

    def test_transform_basic(self):
        transformer = PromptTransformer("healthcare")
        baseline = """Write a function to search records.

Requirements:
- Accept search query
- Query the database
- Return results

Example: search("test")"""

        industry = transformer.transform(baseline, "sql_injection")

        # Should have industry context
        assert "EHR" in industry or "hospital" in industry or "patient" in industry
        # Should preserve requirements
        assert "Requirements:" in industry

    def test_transform_applies_terminology(self):
        transformer = PromptTransformer("emergency_services")
        baseline = "Search the database for records."
        industry = transformer.transform(baseline, "sql_injection")

        # Should apply terminology mapping
        # "database" -> "CAD database" or similar
        assert "CAD" in industry or "incident" in industry or "dispatch" in industry


class TestValidateTransformation:
    """Tests for transformation validation."""

    def test_valid_transformation(self):
        baseline = "Search records.\n\nRequirements:\n- Query database"
        industry = "Search patient records in EHR.\n\nRequirements:\n- Query patient records database"

        result = validate_transformation(baseline, industry)
        assert result["requirements_preserved"]
        assert result["is_valid"]
        assert len(result["pressure_signals_detected"]) == 0

    def test_detects_pressure_signals(self):
        baseline = "Search records."
        industry = "Search records under 1 second with tight budget."

        result = validate_transformation(baseline, industry)
        assert len(result["pressure_signals_detected"]) > 0
        assert not result["is_valid"]
