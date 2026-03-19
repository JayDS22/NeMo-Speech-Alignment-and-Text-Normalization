"""
Text Normalization (TN) & Inverse Text Normalization (ITN) Engine.

Converts written-form text to spoken-form for ASR dataset preparation.
Handles numbers, abbreviations, dates, currency, time, and measurements.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    from num2words import num2words
except ImportError:
    num2words = None


@dataclass
class NormalizationResult:
    """Result of text normalization."""
    original: str
    normalized: str
    changes: List[Dict[str, str]] = field(default_factory=list)
    num_changes: int = 0

    def __post_init__(self):
        self.num_changes = len(self.changes)


class TextNormalizer:
    """
    Production text normalization engine for ASR data preparation.

    Supports:
        - Cardinal/ordinal number expansion
        - Currency conversion
        - Date formatting
        - Abbreviation expansion
        - Time conversion
        - Measure/unit expansion
    """

    # Common abbreviations mapping
    ABBREVIATIONS: Dict[str, str] = {
        "dr.": "doctor", "dr": "doctor",
        "mr.": "mister", "mr": "mister",
        "mrs.": "missus", "mrs": "missus",
        "ms.": "miss", "ms": "miss",
        "prof.": "professor", "prof": "professor",
        "sr.": "senior", "sr": "senior",
        "jr.": "junior", "jr": "junior",
        "st.": "saint", "ave.": "avenue",
        "blvd.": "boulevard", "rd.": "road",
        "dept.": "department", "govt.": "government",
        "approx.": "approximately", "etc.": "et cetera",
        "vs.": "versus", "vs": "versus",
        "inc.": "incorporated", "corp.": "corporation",
        "ltd.": "limited", "co.": "company",
        "jan.": "january", "feb.": "february",
        "mar.": "march", "apr.": "april",
        "jun.": "june", "jul.": "july",
        "aug.": "august", "sep.": "september",
        "sept.": "september", "oct.": "october",
        "nov.": "november", "dec.": "december",
        "mon.": "monday", "tue.": "tuesday",
        "wed.": "wednesday", "thu.": "thursday",
        "fri.": "friday", "sat.": "saturday",
        "sun.": "sunday",
        "ft.": "feet", "in.": "inches",
        "lb.": "pounds", "oz.": "ounces",
        "min.": "minutes", "sec.": "seconds",
        "hr.": "hours", "hrs.": "hours",
        "no.": "number", "vol.": "volume",
        "fig.": "figure", "eq.": "equation",
        "approx": "approximately",
        "govt": "government", "dept": "department",
        "univ.": "university", "assn.": "association",
    }

    # Unit expansions
    UNITS: Dict[str, str] = {
        "kg": "kilograms", "g": "grams", "mg": "milligrams",
        "lb": "pounds", "oz": "ounces",
        "km": "kilometers", "m": "meters", "cm": "centimeters",
        "mm": "millimeters", "mi": "miles", "ft": "feet",
        "in": "inches", "yd": "yards",
        "l": "liters", "ml": "milliliters", "gal": "gallons",
        "mph": "miles per hour", "kmh": "kilometers per hour",
        "kph": "kilometers per hour",
        "hz": "hertz", "khz": "kilohertz", "mhz": "megahertz",
        "ghz": "gigahertz",
        "kb": "kilobytes", "mb": "megabytes", "gb": "gigabytes",
        "tb": "terabytes",
    }

    # Currency symbols
    CURRENCIES: Dict[str, str] = {
        "$": "dollars", "€": "euros", "£": "pounds",
        "¥": "yen", "₹": "rupees",
    }

    MONTH_NAMES = {
        1: "january", 2: "february", 3: "march", 4: "april",
        5: "may", 6: "june", 7: "july", 8: "august",
        9: "september", 10: "october", 11: "november", 12: "december",
    }

    ORDINAL_SUFFIXES = {
        1: "first", 2: "second", 3: "third", 4: "fourth",
        5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth",
        9: "ninth", 10: "tenth", 11: "eleventh", 12: "twelfth",
        13: "thirteenth", 14: "fourteenth", 15: "fifteenth",
        16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
        19: "nineteenth", 20: "twentieth", 21: "twenty first",
        22: "twenty second", 23: "twenty third", 24: "twenty fourth",
        25: "twenty fifth", 26: "twenty sixth", 27: "twenty seventh",
        28: "twenty eighth", 29: "twenty ninth", 30: "thirtieth",
        31: "thirty first",
    }

    def __init__(
        self,
        expand_numbers: bool = True,
        expand_abbreviations: bool = True,
        expand_dates: bool = True,
        expand_currency: bool = True,
        expand_time: bool = True,
        expand_measures: bool = True,
        lowercase: bool = True,
        language: str = "en",
    ):
        self.expand_numbers = expand_numbers
        self.expand_abbreviations = expand_abbreviations
        self.expand_dates = expand_dates
        self.expand_currency = expand_currency
        self.expand_time = expand_time
        self.expand_measures = expand_measures
        self.lowercase = lowercase
        self.language = language

    def normalize(self, text: str) -> NormalizationResult:
        """
        Apply full text normalization pipeline.

        Args:
            text: Raw input text.

        Returns:
            NormalizationResult with original, normalized text, and change log.
        """
        original = text
        changes = []

        if self.expand_currency:
            text, c = self._normalize_currency(text)
            changes.extend(c)

        if self.expand_dates:
            text, c = self._normalize_dates(text)
            changes.extend(c)

        if self.expand_time:
            text, c = self._normalize_time(text)
            changes.extend(c)

        if self.expand_measures:
            text, c = self._normalize_measures(text)
            changes.extend(c)

        if self.expand_numbers:
            text, c = self._normalize_numbers(text)
            changes.extend(c)

        if self.expand_abbreviations:
            text, c = self._normalize_abbreviations(text)
            changes.extend(c)

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        if self.lowercase:
            text = text.lower()

        return NormalizationResult(
            original=original,
            normalized=text,
            changes=changes,
        )

    def normalize_batch(self, texts: List[str]) -> List[NormalizationResult]:
        """Normalize a batch of texts."""
        return [self.normalize(t) for t in texts]

    # ── Number Expansion ──────────────────────────────────────────

    def _number_to_words(self, n: float) -> str:
        """Convert a number to its word representation."""
        if num2words is not None:
            try:
                if n == int(n):
                    return num2words(int(n), lang=self.language)
                else:
                    return num2words(n, lang=self.language)
            except Exception:
                pass
        return self._fallback_number_to_words(n)

    def _fallback_number_to_words(self, n: float) -> str:
        """Fallback number-to-words without num2words library."""
        ones = ["", "one", "two", "three", "four", "five", "six",
                "seven", "eight", "nine", "ten", "eleven", "twelve",
                "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                "eighteen", "nineteen"]
        tens = ["", "", "twenty", "thirty", "forty", "fifty",
                "sixty", "seventy", "eighty", "ninety"]

        if n < 0:
            return "negative " + self._fallback_number_to_words(-n)

        n_int = int(n)
        decimal_part = round(n - n_int, 10)

        if n_int == 0 and decimal_part == 0:
            return "zero"

        result = ""
        if n_int >= 1000000:
            millions = n_int // 1000000
            result += self._fallback_number_to_words(millions) + " million "
            n_int %= 1000000

        if n_int >= 1000:
            thousands = n_int // 1000
            result += self._fallback_number_to_words(thousands) + " thousand "
            n_int %= 1000

        if n_int >= 100:
            result += ones[n_int // 100] + " hundred "
            n_int %= 100

        if n_int >= 20:
            result += tens[n_int // 10] + " "
            n_int %= 10

        if n_int > 0:
            result += ones[n_int] + " "

        result = result.strip()

        if decimal_part > 0:
            decimal_str = str(round(decimal_part, 10)).split('.')[1]
            result += " point " + " ".join(ones[int(d)] if int(d) > 0 else "zero"
                                           for d in decimal_str)

        return result.strip()

    def _normalize_numbers(self, text: str) -> Tuple[str, List[Dict]]:
        """Expand standalone numbers to words."""
        changes = []

        def replace_number(match):
            num_str = match.group(0)
            try:
                num = float(num_str.replace(",", ""))
                word = self._number_to_words(num)
                changes.append({"type": "number", "original": num_str, "normalized": word})
                return word
            except ValueError:
                return num_str

        # Match numbers with optional commas and decimals
        text = re.sub(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|\b\d{4,}(?:\.\d+)?\b', replace_number, text)
        return text, changes

    # ── Currency Expansion ────────────────────────────────────────

    def _normalize_currency(self, text: str) -> Tuple[str, List[Dict]]:
        """Expand currency expressions."""
        changes = []

        for symbol, name in self.CURRENCIES.items():
            pattern = re.escape(symbol) + r'(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)'
            matches = re.finditer(pattern, text)
            for match in matches:
                original = match.group(0)
                amount_str = match.group(1).replace(",", "")
                amount = float(amount_str)

                dollars = int(amount)
                cents = round((amount - dollars) * 100)

                dollar_word = self._number_to_words(dollars)
                if cents > 0:
                    cent_word = self._number_to_words(cents)
                    singular = name.rstrip('s') if dollars == 1 else name
                    replacement = f"{dollar_word} {singular} and {cent_word} cents"
                else:
                    singular = name.rstrip('s') if dollars == 1 else name
                    replacement = f"{dollar_word} {singular}"

                text = text.replace(original, replacement, 1)
                changes.append({"type": "currency", "original": original, "normalized": replacement})

        return text, changes

    # ── Date Expansion ────────────────────────────────────────────

    def _normalize_dates(self, text: str) -> Tuple[str, List[Dict]]:
        """Expand date patterns to spoken form."""
        changes = []

        # MM/DD/YYYY or MM-DD-YYYY
        def replace_date_mdy(match):
            month = int(match.group(1))
            day = int(match.group(2))
            year = int(match.group(3))
            if 1 <= month <= 12 and 1 <= day <= 31:
                month_name = self.MONTH_NAMES.get(month, str(month))
                day_word = self.ORDINAL_SUFFIXES.get(day, self._number_to_words(day))
                year_word = self._expand_year(year)
                replacement = f"{month_name} {day_word} {year_word}"
                changes.append({"type": "date", "original": match.group(0), "normalized": replacement})
                return replacement
            return match.group(0)

        text = re.sub(r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})', replace_date_mdy, text)

        # YYYY-MM-DD (ISO)
        def replace_date_iso(match):
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            if 1 <= month <= 12 and 1 <= day <= 31:
                month_name = self.MONTH_NAMES.get(month, str(month))
                day_word = self.ORDINAL_SUFFIXES.get(day, self._number_to_words(day))
                year_word = self._expand_year(year)
                replacement = f"{month_name} {day_word} {year_word}"
                changes.append({"type": "date", "original": match.group(0), "normalized": replacement})
                return replacement
            return match.group(0)

        text = re.sub(r'(\d{4})-(\d{2})-(\d{2})', replace_date_iso, text)

        return text, changes

    def _expand_year(self, year: int) -> str:
        """Expand a year to spoken form."""
        if 2000 <= year <= 2009:
            return "two thousand " + (self._number_to_words(year - 2000) if year > 2000 else "")
        elif 2010 <= year <= 2099:
            return "twenty " + self._number_to_words(year - 2000)
        elif 1000 <= year <= 1999:
            first = year // 100
            second = year % 100
            first_word = self._number_to_words(first)
            if second == 0:
                return first_word + " hundred"
            else:
                return first_word + " " + self._number_to_words(second)
        else:
            return self._number_to_words(year)

    # ── Time Expansion ────────────────────────────────────────────

    def _normalize_time(self, text: str) -> Tuple[str, List[Dict]]:
        """Expand time expressions."""
        changes = []

        def replace_time(match):
            hour = int(match.group(1))
            minute = int(match.group(2))
            period = match.group(3) if match.group(3) else ""

            hour_word = self._number_to_words(hour)
            if minute == 0:
                time_word = hour_word + " o'clock"
            else:
                minute_word = self._number_to_words(minute)
                if minute < 10:
                    minute_word = "oh " + minute_word
                time_word = f"{hour_word} {minute_word}"

            if period:
                time_word += " " + " ".join(period.upper().replace(".", ""))

            changes.append({"type": "time", "original": match.group(0), "normalized": time_word})
            return time_word

        text = re.sub(r'(\d{1,2}):(\d{2})\s*((?:AM|PM|am|pm|a\.m\.|p\.m\.)?)', replace_time, text)
        return text, changes

    # ── Measure Expansion ─────────────────────────────────────────

    def _normalize_measures(self, text: str) -> Tuple[str, List[Dict]]:
        """Expand measurement units."""
        changes = []

        for unit, expansion in sorted(self.UNITS.items(), key=lambda x: -len(x[0])):
            pattern = r'(\d+(?:\.\d+)?)\s*' + re.escape(unit) + r'\b'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in reversed(list(matches)):
                original = match.group(0)
                number = float(match.group(1))
                num_word = self._number_to_words(number)
                unit_word = expansion if number != 1 else expansion.rstrip('s')
                replacement = f"{num_word} {unit_word}"
                text = text[:match.start()] + replacement + text[match.end():]
                changes.append({"type": "measure", "original": original, "normalized": replacement})

        return text, changes

    # ── Abbreviation Expansion ────────────────────────────────────

    def _normalize_abbreviations(self, text: str) -> Tuple[str, List[Dict]]:
        """Expand common abbreviations."""
        changes = []

        for abbrev, expansion in sorted(self.ABBREVIATIONS.items(), key=lambda x: -len(x[0])):
            # For abbreviations ending with '.', match with or without trailing period
            escaped = re.escape(abbrev)
            if abbrev.endswith('.'):
                pattern = r'(?<!\w)' + escaped + r'(?!\w)'
            else:
                pattern = r'\b' + escaped + r'\b'
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in reversed(matches):
                original = match.group(0)
                text = text[:match.start()] + expansion + text[match.end():]
                changes.append({"type": "abbreviation", "original": original, "normalized": expansion})

        return text, changes
