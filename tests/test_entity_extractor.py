"""Tests for the regex-based entity extractor."""

import pytest

from elo_memory.memory.entity_extractor import EntityExtractor


@pytest.fixture
def ex():
    return EntityExtractor()


# ── Emails ────────────────────────────────────────────────────────────

class TestEmails:
    def test_simple_email(self, ex):
        result = ex.extract("Contact alice@example.com for info.")
        assert result["emails"] == ["alice@example.com"]

    def test_multiple_emails(self, ex):
        text = "Send to bob@corp.io and carol+tag@sub.domain.org."
        result = ex.extract(text)
        assert "bob@corp.io" in result["emails"]
        assert "carol+tag@sub.domain.org" in result["emails"]

    def test_no_emails(self, ex):
        assert ex.extract("No email here.")["emails"] == []

    def test_dedup_emails(self, ex):
        text = "alice@example.com and alice@example.com again"
        assert len(ex.extract(text)["emails"]) == 1


# ── Names ─────────────────────────────────────────────────────────────

class TestNames:
    def test_two_word_name(self, ex):
        result = ex.extract("Met with Sarah Chen today.")
        assert "Sarah Chen" in result["names"]

    def test_three_word_name(self, ex):
        result = ex.extract("Talked to Mary Jane Watson.")
        assert "Mary Jane Watson" in result["names"]

    def test_skip_word_stripping(self, ex):
        # "The Sarah Chen" -> strip "The" -> "Sarah Chen"
        result = ex.extract("The Sarah Chen project is live.")
        assert "Sarah Chen" in result["names"]

    def test_skip_sentence_starters(self, ex):
        # "The Project" -- after stripping "The", only one word left -> skip
        result = ex.extract("The Project is done.")
        assert result["names"] == []

    def test_skip_gerund_starters(self, ex):
        result = ex.extract("Using Python for scripts.")
        assert result["names"] == []

    def test_role_word_single_name(self, ex):
        result = ex.extract("Engineer Maria built the API.")
        assert "Maria" in result["names"]

    def test_wife_name(self, ex):
        result = ex.extract("My wife Julia loved the trip.")
        assert "Julia" in result["names"]

    def test_hired_name(self, ex):
        result = ex.extract("Hired Jake last Monday.")
        assert "Jake" in result["names"]

    def test_promoted_name(self, ex):
        result = ex.extract("Promoted Sarah to lead.")
        assert "Sarah" in result["names"]

    def test_tech_product_rejected(self, ex):
        result = ex.extract("Deploy on Vercel with Redis caching.")
        assert result["names"] == []

    def test_multi_word_tech_rejected(self, ex):
        # "React Native" -- React is in _NOT_NAMES
        result = ex.extract("We use React Native for mobile.")
        assert "React Native" not in result["names"]

    def test_place_component_rejected(self, ex):
        result = ex.extract("Office in San Francisco.")
        assert result["names"] == []

    def test_new_york_rejected(self, ex):
        result = ex.extract("Flew to New York yesterday.")
        assert result["names"] == []

    def test_dedup_names(self, ex):
        text = "Sarah Chen met Sarah Chen at lunch."
        names = ex.extract(text)["names"]
        assert names.count("Sarah Chen") == 1

    def test_real_name_not_filtered(self, ex):
        result = ex.extract("David Kim joined the team.")
        assert "David Kim" in result["names"]


# ── Dates ─────────────────────────────────────────────────────────────

class TestDates:
    def test_iso_date(self, ex):
        result = ex.extract("Deadline is 2024-01-15.")
        assert "2024-01-15" in result["dates"]

    def test_long_date(self, ex):
        result = ex.extract("Meeting on January 15, 2024.")
        assert "January 15, 2024" in result["dates"]

    def test_long_date_no_year(self, ex):
        result = ex.extract("Party on March 8 at noon.")
        assert "March 8" in result["dates"]

    def test_short_date(self, ex):
        result = ex.extract("Due by Jan 20, 2025.")
        assert "Jan 20, 2025" in result["dates"]

    def test_relative_yesterday(self, ex):
        result = ex.extract("Finished it yesterday.")
        assert "yesterday" in result["dates"]

    def test_relative_next_monday(self, ex):
        result = ex.extract("Let's sync next Monday.")
        assert "next Monday" in result["dates"]

    def test_relative_last_week(self, ex):
        result = ex.extract("Shipped it last week.")
        assert "last week" in result["dates"]

    def test_no_dates(self, ex):
        assert ex.extract("No dates here.")["dates"] == []


# ── Numbers with units ────────────────────────────────────────────────

class TestNumbers:
    def test_dollar_amount(self, ex):
        result = ex.extract("Budget is $500k for Q1.")
        assert "$500k" in result["numbers"]

    def test_dollar_with_commas(self, ex):
        result = ex.extract("Revenue hit $1,200,000 last year.")
        nums = result["numbers"]
        assert any("1,200,000" in n for n in nums)

    def test_storage_size(self, ex):
        result = ex.extract("Disk usage at 42GB.")
        assert any("42GB" in n for n in result["numbers"])

    def test_user_count(self, ex):
        result = ex.extract("We have 100 users now.")
        assert "100 users" in result["numbers"]

    def test_time_unit(self, ex):
        result = ex.extract("Took 3 hours to fix.")
        assert "3 hours" in result["numbers"]

    def test_percentage(self, ex):
        result = ex.extract("Conversion rate is 50%.")
        assert any("50" in n and "%" in n for n in result["numbers"])

    def test_no_numbers(self, ex):
        assert ex.extract("No numbers here.")["numbers"] == []


# ── URLs ──────────────────────────────────────────────────────────────

class TestURLs:
    def test_https_url(self, ex):
        result = ex.extract("Visit https://example.com/page for docs.")
        assert "https://example.com/page" in result["urls"]

    def test_http_url(self, ex):
        result = ex.extract("See http://old-site.io/api/v2.")
        assert "http://old-site.io/api/v2" in result["urls"]

    def test_trailing_punctuation_stripped(self, ex):
        result = ex.extract("Go to https://example.com.")
        assert "https://example.com" in result["urls"]

    def test_no_urls(self, ex):
        assert ex.extract("No links here.")["urls"] == []


# ── Versions ──────────────────────────────────────────────────────────

class TestVersions:
    def test_short_version(self, ex):
        result = ex.extract("Upgraded to v2.1 today.")
        assert "v2.1" in result["versions"]

    def test_long_version(self, ex):
        result = ex.extract("Running version 3.0 in prod.")
        assert "version 3.0" in result["versions"]

    def test_semver(self, ex):
        result = ex.extract("Pinned at v1.2.3 for stability.")
        assert "v1.2.3" in result["versions"]

    def test_no_versions(self, ex):
        assert ex.extract("No versions.")["versions"] == []


# ── extract_flat ──────────────────────────────────────────────────────

class TestExtractFlat:
    def test_flat_dedup(self, ex):
        text = "Sarah Chen (sarah@example.com) deployed v2.1 on 2024-01-15."
        flat = ex.extract_flat(text)
        assert "sarah@example.com" in flat
        assert "Sarah Chen" in flat
        assert "v2.1" in flat
        assert "2024-01-15" in flat
        # No duplicates
        assert len(flat) == len(set(flat))

    def test_flat_empty(self, ex):
        assert ex.extract_flat("nothing special here") == []


# ── False positive prevention ─────────────────────────────────────────

class TestFalsePositives:
    def test_sentence_starters_not_names(self, ex):
        texts = [
            "The Quick Brown Fox jumps.",
            "This Morning Was great.",
            "We Should Leave now.",
        ]
        for text in texts:
            names = ex.extract(text)["names"]
            for n in names:
                first = n.split()[0]
                assert first not in {"The", "This", "We"}, f"False positive: {n}"

    def test_tech_words_not_names(self, ex):
        text = "Set up Docker Redis Prisma Vercel."
        assert ex.extract(text)["names"] == []

    def test_gerunds_not_names(self, ex):
        text = "Building Great Things every day."
        names = ex.extract(text)["names"]
        for n in names:
            assert not n.startswith("Building"), f"False positive: {n}"

    def test_place_components_not_names(self, ex):
        for place in ["San Diego", "Los Angeles", "New Orleans", "North Dakota"]:
            result = ex.extract(f"Visited {place} last year.")
            assert place not in result["names"], f"False positive: {place}"
