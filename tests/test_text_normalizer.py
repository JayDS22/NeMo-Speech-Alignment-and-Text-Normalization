"""Tests for TextNormalizer."""

import pytest
from src.text_normalizer import TextNormalizer, NormalizationResult


@pytest.fixture
def normalizer():
    return TextNormalizer()


class TestNumberExpansion:
    def test_simple_integer(self, normalizer):
        result = normalizer.normalize("I have 42 apples")
        assert "forty" in result.normalized
        assert "two" in result.normalized

    def test_large_number(self, normalizer):
        result = normalizer.normalize("The population is 1000000")
        assert "million" in result.normalized

    def test_decimal(self, normalizer):
        result = normalizer.normalize("Pi is approximately 3.14")
        assert "three" in result.normalized
        assert "point" in result.normalized

    def test_comma_separated(self, normalizer):
        result = normalizer.normalize("Revenue was 1,500,000")
        assert "million" in result.normalized or "thousand" in result.normalized

    def test_zero(self, normalizer):
        result = normalizer.normalize("The count is 0")
        assert "zero" in result.normalized


class TestCurrencyExpansion:
    def test_dollars(self, normalizer):
        result = normalizer.normalize("It costs $3.50")
        assert "three" in result.normalized
        assert "dollar" in result.normalized
        assert "fifty" in result.normalized
        assert "cent" in result.normalized

    def test_whole_dollars(self, normalizer):
        result = normalizer.normalize("Price is $100")
        assert "hundred" in result.normalized
        assert "dollar" in result.normalized

    def test_euros(self, normalizer):
        result = normalizer.normalize("That costs €25")
        assert "twenty" in result.normalized
        assert "euro" in result.normalized


class TestDateExpansion:
    def test_us_date(self, normalizer):
        result = normalizer.normalize("Date is 01/15/2024")
        assert "january" in result.normalized
        assert "fifteenth" in result.normalized
        assert "twenty" in result.normalized

    def test_iso_date(self, normalizer):
        result = normalizer.normalize("Date is 2024-03-22")
        assert "march" in result.normalized
        assert "twenty" in result.normalized

    def test_dash_date(self, normalizer):
        result = normalizer.normalize("Born on 12-25-1990")
        assert "december" in result.normalized
        assert "twenty fifth" in result.normalized


class TestTimeExpansion:
    def test_time_with_am(self, normalizer):
        result = normalizer.normalize("Meeting at 3:30 PM")
        assert "three" in result.normalized
        assert "thirty" in result.normalized

    def test_on_the_hour(self, normalizer):
        result = normalizer.normalize("Alarm at 7:00")
        assert "seven" in result.normalized
        assert "o'clock" in result.normalized


class TestAbbreviationExpansion:
    def test_title(self, normalizer):
        result = normalizer.normalize("Dr. Smith arrived")
        assert "doctor" in result.normalized

    def test_street(self, normalizer):
        result = normalizer.normalize("Lives on Oak St.")
        # St. expands to "saint" in our abbreviation dictionary
        assert "saint" in result.normalized

    def test_multiple_abbrevs(self, normalizer):
        result = normalizer.normalize("Prof. Johnson from the Dept. of Science")
        assert "professor" in result.normalized
        assert "department" in result.normalized


class TestMeasureExpansion:
    def test_weight(self, normalizer):
        result = normalizer.normalize("Weighs 5kg")
        assert "five" in result.normalized
        assert "kilogram" in result.normalized

    def test_distance(self, normalizer):
        result = normalizer.normalize("Ran 10km today")
        assert "ten" in result.normalized
        assert "kilometer" in result.normalized

    def test_speed(self, normalizer):
        result = normalizer.normalize("Speed limit 65mph")
        assert "sixty" in result.normalized
        assert "five" in result.normalized
        assert "miles per hour" in result.normalized


class TestBatchProcessing:
    def test_batch(self, normalizer):
        texts = [
            "I have $5.00",
            "Meeting at 3:00 PM",
            "Date is 01/01/2024",
        ]
        results = normalizer.normalize_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, NormalizationResult) for r in results)

    def test_changes_tracked(self, normalizer):
        result = normalizer.normalize("Dr. Smith has $42 and 3kg of rice")
        assert result.num_changes > 0
        assert len(result.changes) > 0


class TestEdgeCases:
    def test_empty_string(self, normalizer):
        result = normalizer.normalize("")
        assert result.normalized == ""

    def test_no_changes_needed(self, normalizer):
        result = normalizer.normalize("hello world")
        assert result.normalized == "hello world"

    def test_mixed_content(self, normalizer):
        result = normalizer.normalize(
            "Dr. Smith spent $3.50 on 2kg of apples at 3:30 PM on 01/15/2024"
        )
        assert "doctor" in result.normalized
        assert "dollar" in result.normalized
        assert "kilogram" in result.normalized
        assert "january" in result.normalized
