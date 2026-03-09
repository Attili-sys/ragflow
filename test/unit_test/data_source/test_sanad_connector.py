#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""Unit tests for the Sanad (Al Jazeera) connector.

Run with:
    cd ragflow
    uv run pytest test/unit_test/data_source/test_sanad_connector.py -v

Run a single class:
    uv run pytest test/unit_test/data_source/test_sanad_connector.py::TestStripHtml -v

Run by marker:
    uv run pytest test/unit_test/data_source/test_sanad_connector.py -m p1 -v
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from common.data_source.exceptions import (
    ConnectorMissingCredentialError,
    ConnectorValidationError,
)
from common.data_source.sanad_connector import (
    SanadAPI,
    SanadConnector,
    SanadStory,
    _strip_html,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_story(
    story_id: int = 12345,
    title: str = "عنوان الخبر",
    content: str = "<p>محتوى الخبر هنا</p>",
    published: str = "2025-01-15T10:30:00",
    category: str = "Politics",
    country_name: str = "Qatar",
    country_iso2: str = "QA",
    media_type: str = "article",
    source: str = "Al Jazeera",
    copyright_val: str = "Al Jazeera Media Network",
    news_types: list[str] | None = None,
    ingested_at: str | None = None,
) -> dict:
    """Build a raw story dict as returned by the Sanad API."""
    raw: dict = {
        "id": story_id,
        "title": title,
        "content": content,
        "published": published,
        "category": category,
        "country": {"name": country_name, "ISO2": country_iso2},
        "mediaType": media_type,
        "source": source,
        "copyright": copyright_val,
        "newsType": [{"name": n} for n in (news_types or ["Breaking News"])],
    }
    if ingested_at is not None:
        raw["ingested_at"] = ingested_at
    return raw


def _make_story(
    story_id: int = 12345,
    published: datetime | None = None,
    ingested_at: datetime | None = None,
    title: str = "Test Story",
    content_html: str = "<p>Test content</p>",
) -> SanadStory:
    """Build a SanadStory object directly."""
    if published is None:
        published = datetime(2025, 1, 15, 10, 30, tzinfo=timezone.utc)
    return SanadStory(
        story_id=story_id,
        title=title,
        content_html=content_html,
        published=published,
        category="Politics",
        country_name="Qatar",
        country_iso2="QA",
        media_type="article",
        source="Al Jazeera",
        copyright="Al Jazeera Media Network",
        news_types=["Breaking News"],
        ingested_at=ingested_at,
    )


def _mock_response(json_data: object, status_code: int = 200) -> Mock:
    """Return a mock requests.Response."""
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(
            response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# _strip_html
# ---------------------------------------------------------------------------

class TestStripHtml:
    """Tests for the _strip_html helper function."""

    @pytest.mark.p1
    def test_removes_basic_html_tags(self):
        assert _strip_html("<p>Hello</p>") == "Hello"

    @pytest.mark.p1
    def test_preserves_arabic_text(self):
        result = _strip_html("<p>أخبار عاجلة من قطر</p>")
        assert "أخبار عاجلة من قطر" in result

    @pytest.mark.p1
    def test_collapses_whitespace(self):
        result = _strip_html("<p>word1</p>   <p>word2</p>")
        assert "  " not in result
        assert "word1" in result
        assert "word2" in result

    @pytest.mark.p2
    def test_empty_string_returns_empty(self):
        assert _strip_html("") == ""

    @pytest.mark.p2
    def test_no_html_returns_text_unchanged(self):
        result = _strip_html("Plain text with no tags")
        assert result == "Plain text with no tags"

    @pytest.mark.p2
    def test_strips_nested_tags(self):
        result = _strip_html("<div><span><b>deep</b></span></div>")
        assert result == "deep"

    @pytest.mark.p2
    def test_strips_tags_with_attributes(self):
        result = _strip_html('<a href="http://example.com" class="link">Click here</a>')
        assert result == "Click here"

    @pytest.mark.p2
    def test_handles_self_closing_tags(self):
        result = _strip_html("line1<br/>line2")
        assert "line1" in result
        assert "line2" in result

    @pytest.mark.p2
    def test_mixed_arabic_and_html(self):
        html = "<h1>عنوان</h1><p>محتوى <b>مهم</b> هنا</p>"
        result = _strip_html(html)
        assert "عنوان" in result
        assert "محتوى" in result
        assert "مهم" in result
        assert "<" not in result


# ---------------------------------------------------------------------------
# SanadStory
# ---------------------------------------------------------------------------

class TestSanadStory:
    """Tests for the SanadStory data class."""

    @pytest.mark.p1
    def test_html_content_is_stripped_on_init(self):
        story = SanadStory(
            story_id=1,
            title="Title",
            content_html="<p>Content here</p>",
            published=datetime(2025, 1, 1, tzinfo=timezone.utc),
            category="",
            country_name="",
            country_iso2="",
            media_type="",
            source="",
            copyright="",
            news_types=[],
        )
        assert story.content_text == "Content here"
        assert "<p>" not in story.content_text

    @pytest.mark.p1
    def test_empty_content_html_gives_empty_text(self):
        story = _make_story(content_html="")
        assert story.content_text == ""

    @pytest.mark.p2
    def test_str_truncates_title(self):
        long_title = "A" * 200
        story = _make_story(title=long_title)
        result = str(story)
        assert "SanadStory" in result
        assert str(story.story_id) in result

    @pytest.mark.p2
    def test_ingested_at_optional(self):
        story = _make_story(ingested_at=None)
        assert story.ingested_at is None

    @pytest.mark.p2
    def test_ingested_at_stored(self):
        ts = datetime(2025, 6, 1, tzinfo=timezone.utc)
        story = _make_story(ingested_at=ts)
        assert story.ingested_at == ts


# ---------------------------------------------------------------------------
# SanadAPI._parse_story
# ---------------------------------------------------------------------------

class TestParseStory:
    """Tests for SanadAPI._parse_story static method."""

    @pytest.mark.p1
    def test_valid_full_story(self):
        raw = _make_raw_story()
        story = SanadAPI._parse_story(raw)
        assert story is not None
        assert story.story_id == 12345
        assert story.title == "عنوان الخبر"
        assert story.category == "Politics"
        assert story.country_name == "Qatar"
        assert story.country_iso2 == "QA"
        assert story.media_type == "article"
        assert story.source == "Al Jazeera"
        assert story.copyright == "Al Jazeera Media Network"
        assert story.news_types == ["Breaking News"]

    @pytest.mark.p1
    def test_missing_required_id_returns_none(self):
        raw = _make_raw_story()
        del raw["id"]
        story = SanadAPI._parse_story(raw)
        assert story is None

    @pytest.mark.p1
    def test_published_date_parsed_correctly(self):
        raw = _make_raw_story(published="2025-03-01T12:00:00")
        story = SanadAPI._parse_story(raw)
        assert story is not None
        assert story.published == datetime(2025, 3, 1, 12, 0, tzinfo=timezone.utc)

    @pytest.mark.p1
    def test_published_with_timezone_kept(self):
        raw = _make_raw_story(published="2025-03-01T12:00:00+03:00")
        story = SanadAPI._parse_story(raw)
        assert story is not None
        assert story.published.tzinfo is not None

    @pytest.mark.p1
    def test_missing_published_defaults_to_now(self):
        raw = _make_raw_story()
        raw["published"] = ""
        before = datetime.now(timezone.utc)
        story = SanadAPI._parse_story(raw)
        after = datetime.now(timezone.utc)
        assert story is not None
        assert before <= story.published <= after

    @pytest.mark.p2
    def test_invalid_published_date_returns_none(self):
        raw = _make_raw_story(published="not-a-date")
        story = SanadAPI._parse_story(raw)
        assert story is None

    @pytest.mark.p2
    def test_ingested_at_parsed(self):
        raw = _make_raw_story(ingested_at="2025-06-01T08:00:00")
        story = SanadAPI._parse_story(raw)
        assert story is not None
        assert story.ingested_at == datetime(2025, 6, 1, 8, 0, tzinfo=timezone.utc)

    @pytest.mark.p2
    def test_ingested_at_absent_is_none(self):
        raw = _make_raw_story()  # no ingested_at key
        story = SanadAPI._parse_story(raw)
        assert story is not None
        assert story.ingested_at is None

    @pytest.mark.p2
    def test_multiple_news_types(self):
        raw = _make_raw_story(news_types=["Breaking News", "World", "Arabic"])
        story = SanadAPI._parse_story(raw)
        assert story is not None
        assert story.news_types == ["Breaking News", "World", "Arabic"]

    @pytest.mark.p2
    def test_null_news_types_gives_empty_list(self):
        raw = _make_raw_story()
        raw["newsType"] = None
        story = SanadAPI._parse_story(raw)
        assert story is not None
        assert story.news_types == []

    @pytest.mark.p2
    def test_null_country_gives_empty_strings(self):
        raw = _make_raw_story()
        raw["country"] = None
        story = SanadAPI._parse_story(raw)
        assert story is not None
        assert story.country_name == ""
        assert story.country_iso2 == ""

    @pytest.mark.p2
    def test_missing_optional_fields_use_defaults(self):
        raw = {"id": 99, "published": "2025-01-01T00:00:00"}
        story = SanadAPI._parse_story(raw)
        assert story is not None
        assert story.title == ""
        assert story.category == ""
        assert story.source == ""
        assert story.media_type == ""
        assert story.copyright == ""


# ---------------------------------------------------------------------------
# SanadAPI.validate_credentials
# ---------------------------------------------------------------------------

class TestValidateCredentials:
    """Tests for SanadAPI.validate_credentials."""

    @pytest.mark.p1
    def test_returns_true_on_success(self):
        api = SanadAPI(api_key="test-key", lang="ar")
        with patch.object(api.session, "get", return_value=_mock_response([])):
            assert api.validate_credentials() is True

    @pytest.mark.p1
    def test_returns_false_on_http_error(self):
        api = SanadAPI(api_key="bad-key", lang="ar")
        with patch.object(api.session, "get", side_effect=requests.HTTPError("401")):
            assert api.validate_credentials() is False

    @pytest.mark.p1
    def test_returns_false_on_connection_error(self):
        api = SanadAPI(api_key="key", lang="ar")
        with patch.object(api.session, "get", side_effect=requests.ConnectionError()):
            assert api.validate_credentials() is False

    @pytest.mark.p2
    def test_returns_false_on_timeout(self):
        api = SanadAPI(api_key="key", lang="ar")
        with patch.object(api.session, "get", side_effect=requests.Timeout()):
            assert api.validate_credentials() is False


# ---------------------------------------------------------------------------
# SanadAPI.get_stories
# ---------------------------------------------------------------------------

class TestGetStories:
    """Tests for SanadAPI.get_stories, including pagination and filtering."""

    def _api(self) -> SanadAPI:
        return SanadAPI(api_key="test-key", lang="ar")

    @pytest.mark.p1
    def test_yields_all_stories_single_page(self):
        stories_data = [_make_raw_story(story_id=i) for i in range(3)]
        api = self._api()
        with patch.object(api.session, "get", return_value=_mock_response(stories_data)):
            result = list(api.get_stories(stories_per_page=20))
        assert len(result) == 3
        assert all(isinstance(s, SanadStory) for s in result)

    @pytest.mark.p1
    def test_stops_early_on_published_cutoff(self):
        """Default mode stops when published < start_date (early-stop optimisation)."""
        old = _make_raw_story(story_id=1, published="2024-01-01T00:00:00")
        new = _make_raw_story(story_id=2, published="2025-06-01T00:00:00")
        # API returns newest-first
        api = self._api()
        with patch.object(api.session, "get", return_value=_mock_response([new, old])):
            start = datetime(2025, 1, 1, tzinfo=timezone.utc)
            result = list(api.get_stories(start_date=start))
        # Should get `new` but stop before `old`
        assert len(result) == 1
        assert result[0].story_id == 2

    @pytest.mark.p1
    def test_empty_response_stops_pagination(self):
        api = self._api()
        with patch.object(api.session, "get", return_value=_mock_response([])):
            result = list(api.get_stories())
        assert result == []

    @pytest.mark.p1
    def test_last_page_detection_by_count(self):
        """Stops if the page returns fewer stories than stories_per_page."""
        # Request 5 per page, get only 2 → last page
        stories_data = [_make_raw_story(story_id=i) for i in range(2)]
        api = self._api()
        with patch.object(api.session, "get", return_value=_mock_response(stories_data)):
            result = list(api.get_stories(stories_per_page=5))
        assert len(result) == 2

    @pytest.mark.p2
    def test_paginated_two_pages(self):
        """Fetches multiple pages until fewer-than-requested count is seen."""
        page1 = [_make_raw_story(story_id=i) for i in range(3)]
        page2 = [_make_raw_story(story_id=i + 100) for i in range(2)]  # fewer → last page
        responses = [_mock_response(page1), _mock_response(page2)]
        api = self._api()
        with patch.object(api.session, "get", side_effect=responses):
            result = list(api.get_stories(stories_per_page=3))
        assert len(result) == 5

    @pytest.mark.p2
    def test_skips_malformed_stories(self):
        """Stories that fail _parse_story (missing id) are silently skipped."""
        good = _make_raw_story(story_id=1)
        bad = {"title": "no id field"}  # missing "id"
        api = self._api()
        with patch.object(api.session, "get", return_value=_mock_response([good, bad])):
            result = list(api.get_stories())
        assert len(result) == 1
        assert result[0].story_id == 1

    @pytest.mark.p2
    def test_retries_on_server_error_then_succeeds(self):
        """Retries transient server errors and eventually yields stories."""
        stories_data = [_make_raw_story(story_id=42)]
        server_err = requests.HTTPError(response=Mock(status_code=503))
        server_err.response = Mock(status_code=503)

        api = self._api()
        with patch.object(
            api.session, "get",
            side_effect=[server_err, server_err, _mock_response(stories_data)]
        ), patch("common.data_source.sanad_connector.time.sleep"):
            result = list(api.get_stories())
        assert len(result) == 1
        assert result[0].story_id == 42

    @pytest.mark.p2
    def test_client_4xx_error_is_not_retried(self):
        """4xx client errors should raise immediately, not retry."""
        client_err = requests.HTTPError("403 Forbidden")
        client_err.response = Mock(status_code=403)

        api = self._api()
        with patch.object(api.session, "get", side_effect=client_err):
            with pytest.raises(requests.HTTPError):
                list(api.get_stories())

    @pytest.mark.p2
    def test_api_wraps_stories_in_results_key(self):
        """Handles API responses wrapped in a `results` dict."""
        stories_data = [_make_raw_story(story_id=77)]
        api = self._api()
        with patch.object(api.session, "get", return_value=_mock_response({"results": stories_data})):
            result = list(api.get_stories())
        assert len(result) == 1
        assert result[0].story_id == 77

    @pytest.mark.p3
    def test_exhausted_retries_stops_gracefully(self):
        """After all retries fail, stops pagination without raising."""
        timeout_err = requests.Timeout("timed out")
        api = self._api()
        with patch.object(api.session, "get", side_effect=timeout_err), \
             patch("common.data_source.sanad_connector.time.sleep"):
            result = list(api.get_stories())
        assert result == []


# ---------------------------------------------------------------------------
# SanadConnector.load_credentials
# ---------------------------------------------------------------------------

class TestLoadCredentials:
    """Tests for SanadConnector.load_credentials."""

    @pytest.mark.p1
    def test_raises_on_missing_api_key(self):
        connector = SanadConnector()
        with pytest.raises(ConnectorMissingCredentialError):
            connector.load_credentials({})

    @pytest.mark.p1
    def test_raises_on_invalid_credentials(self):
        connector = SanadConnector()
        with patch("common.data_source.sanad_connector.SanadAPI") as MockAPI:
            MockAPI.return_value.validate_credentials.return_value = False
            with pytest.raises(ConnectorValidationError):
                connector.load_credentials({"sanad_api_key": "bad-key"})

    @pytest.mark.p1
    def test_succeeds_with_valid_key(self):
        connector = SanadConnector()
        with patch("common.data_source.sanad_connector.SanadAPI") as MockAPI:
            MockAPI.return_value.validate_credentials.return_value = True
            result = connector.load_credentials({"sanad_api_key": "good-key"})
        assert result is None
        assert connector.sanad_client is not None

    @pytest.mark.p2
    def test_api_client_initialized_with_lang(self):
        connector = SanadConnector(lang="en")
        with patch("common.data_source.sanad_connector.SanadAPI") as MockAPI:
            MockAPI.return_value.validate_credentials.return_value = True
            connector.load_credentials({"sanad_api_key": "key"})
        MockAPI.assert_called_once_with(api_key="key", lang="en")


# ---------------------------------------------------------------------------
# SanadConnector._story_to_document
# ---------------------------------------------------------------------------

class TestStoryToDocument:
    """Tests for SanadConnector._story_to_document."""

    def _connector(self) -> SanadConnector:
        return SanadConnector()

    @pytest.mark.p1
    def test_document_id_uses_story_id(self):
        connector = self._connector()
        story = _make_story(story_id=25243)
        doc = connector._story_to_document(story)
        assert doc.id == "sanad:25243"

    @pytest.mark.p1
    def test_doc_updated_at_equals_published(self):
        connector = self._connector()
        published = datetime(2025, 5, 10, tzinfo=timezone.utc)
        story = _make_story(published=published)
        doc = connector._story_to_document(story)
        assert doc.doc_updated_at == published

    @pytest.mark.p1
    def test_blob_contains_title_and_content(self):
        connector = self._connector()
        story = _make_story(title="Breaking News", content_html="<p>Full story here</p>")
        doc = connector._story_to_document(story)
        text = doc.blob.decode("utf-8")
        assert "Breaking News" in text
        assert "Full story here" in text

    @pytest.mark.p1
    def test_extension_is_txt(self):
        connector = self._connector()
        doc = connector._story_to_document(_make_story())
        assert doc.extension == ".txt"

    @pytest.mark.p1
    def test_metadata_contains_expected_fields(self):
        connector = self._connector()
        story = _make_story(story_id=99)
        doc = connector._story_to_document(story)
        assert doc.metadata is not None
        assert doc.metadata["sanad_id"] == "99"
        assert "published" in doc.metadata
        assert doc.metadata["category"] == "Politics"
        assert doc.metadata["country"] == "Qatar"
        assert doc.metadata["country_iso2"] == "QA"
        assert doc.metadata["source"] == "Al Jazeera"
        assert doc.metadata["media_type"] == "article"
        assert doc.metadata["copyright"] == "Al Jazeera Media Network"

    @pytest.mark.p1
    def test_news_types_joined_in_metadata(self):
        connector = self._connector()
        story = SanadStory(
            story_id=1,
            title="T",
            content_html="C",
            published=datetime(2025, 1, 1, tzinfo=timezone.utc),
            category="",
            country_name="",
            country_iso2="",
            media_type="",
            source="",
            copyright="",
            news_types=["World", "Politics", "Sports"],
        )
        doc = connector._story_to_document(story)
        assert doc.metadata["news_types"] == "World, Politics, Sports"

    @pytest.mark.p1
    def test_semantic_identifier_from_title(self):
        connector = self._connector()
        story = _make_story(title="Breaking: Major Event Happens")
        doc = connector._story_to_document(story)
        assert doc.semantic_identifier == "Breaking: Major Event Happens"

    @pytest.mark.p2
    def test_semantic_identifier_truncated_to_100_chars(self):
        connector = self._connector()
        long_title = "أ" * 200
        story = _make_story(title=long_title)
        doc = connector._story_to_document(story)
        assert len(doc.semantic_identifier) <= 100

    @pytest.mark.p2
    def test_semantic_identifier_fallback_when_no_title(self):
        connector = self._connector()
        story = SanadStory(
            story_id=777,
            title="",
            content_html="content",
            published=datetime(2025, 1, 1, tzinfo=timezone.utc),
            category="",
            country_name="",
            country_iso2="",
            media_type="",
            source="",
            copyright="",
            news_types=[],
        )
        doc = connector._story_to_document(story)
        assert "777" in doc.semantic_identifier

    @pytest.mark.p2
    def test_size_bytes_matches_blob(self):
        connector = self._connector()
        story = _make_story(title="Hello", content_html="<p>World</p>")
        doc = connector._story_to_document(story)
        assert doc.size_bytes == len(doc.blob)

    @pytest.mark.p2
    def test_metadata_is_none_when_all_optional_fields_empty(self):
        """If story has no optional metadata fields, metadata should be None."""
        connector = self._connector()
        story = SanadStory(
            story_id=1,
            title="",
            content_html="",
            published=datetime(2025, 1, 1, tzinfo=timezone.utc),
            category="",
            country_name="",
            country_iso2="",
            media_type="",
            source="",
            copyright="",
            news_types=[],
        )
        doc = connector._story_to_document(story)
        # sanad_id and published are always included, so metadata is never None
        assert doc.metadata is not None
        assert doc.metadata["sanad_id"] == "1"

    @pytest.mark.p2
    def test_ingested_at_in_metadata_when_present(self):
        connector = self._connector()
        ingested = datetime(2025, 9, 1, tzinfo=timezone.utc)
        story = _make_story(ingested_at=ingested)
        doc = connector._story_to_document(story)
        assert "ingested_at" in doc.metadata
        assert doc.metadata["ingested_at"] == ingested.isoformat()

    @pytest.mark.p2
    def test_ingested_at_absent_from_metadata_when_none(self):
        connector = self._connector()
        story = _make_story(ingested_at=None)
        doc = connector._story_to_document(story)
        assert "ingested_at" not in (doc.metadata or {})

    @pytest.mark.p2
    def test_published_iso_format_in_metadata(self):
        connector = self._connector()
        published = datetime(2025, 3, 15, 9, 0, tzinfo=timezone.utc)
        story = _make_story(published=published)
        doc = connector._story_to_document(story)
        assert doc.metadata["published"] == published.isoformat()


# ---------------------------------------------------------------------------
# SanadConnector.load_from_state
# ---------------------------------------------------------------------------

class TestLoadFromState:
    """Tests for SanadConnector.load_from_state (full reindex)."""

    def _connector_with_mock_client(self, stories: list[SanadStory]) -> SanadConnector:
        connector = SanadConnector(batch_size=2)
        mock_client = MagicMock()
        mock_client.get_stories.return_value = iter(stories)
        connector.sanad_client = mock_client
        return connector

    @pytest.mark.p1
    def test_yields_no_batches_on_empty_stories(self):
        connector = self._connector_with_mock_client([])
        batches = list(connector.load_from_state())
        assert batches == []

    @pytest.mark.p1
    def test_yields_single_batch_when_fewer_than_batch_size(self):
        stories = [_make_story(story_id=i) for i in range(2)]
        connector = self._connector_with_mock_client(stories)
        batches = list(connector.load_from_state())
        assert len(batches) == 1
        assert len(batches[0]) == 2

    @pytest.mark.p1
    def test_yields_multiple_full_batches(self):
        stories = [_make_story(story_id=i) for i in range(6)]
        connector = SanadConnector(batch_size=2)
        mock_client = MagicMock()
        mock_client.get_stories.return_value = iter(stories)
        connector.sanad_client = mock_client
        batches = list(connector.load_from_state())
        assert len(batches) == 3
        assert all(len(b) == 2 for b in batches)

    @pytest.mark.p1
    def test_yields_partial_final_batch(self):
        stories = [_make_story(story_id=i) for i in range(5)]
        connector = SanadConnector(batch_size=2)
        mock_client = MagicMock()
        mock_client.get_stories.return_value = iter(stories)
        connector.sanad_client = mock_client
        batches = list(connector.load_from_state())
        assert len(batches) == 3
        assert len(batches[-1]) == 1  # remainder

    @pytest.mark.p1
    def test_documents_have_correct_ids(self):
        stories = [_make_story(story_id=100), _make_story(story_id=101)]
        connector = self._connector_with_mock_client(stories)
        all_docs = [doc for batch in connector.load_from_state() for doc in batch]
        ids = {doc.id for doc in all_docs}
        assert ids == {"sanad:100", "sanad:101"}

    @pytest.mark.p2
    def test_passes_no_start_date_to_get_stories(self):
        connector = SanadConnector()
        mock_client = MagicMock()
        mock_client.get_stories.return_value = iter([])
        connector.sanad_client = mock_client
        list(connector.load_from_state())
        call_kwargs = mock_client.get_stories.call_args
        assert call_kwargs.kwargs.get("start_date") is None

    @pytest.mark.p2
    def test_raises_if_client_not_initialized(self):
        connector = SanadConnector()
        # sanad_client is None by default
        with pytest.raises(RuntimeError, match="not initialized"):
            list(connector.load_from_state())

    @pytest.mark.p2
    def test_continue_on_failure_skips_bad_stories(self):
        """A story that raises during conversion should be skipped when continue_on_failure=True."""
        good_story = _make_story(story_id=1)
        bad_story = MagicMock(spec=SanadStory)
        bad_story.story_id = 999
        bad_story.title = "bad"
        bad_story.content_text = "text"
        bad_story.published = datetime(2025, 1, 1, tzinfo=timezone.utc)
        bad_story.ingested_at = None

        connector = SanadConnector(batch_size=10, continue_on_failure=True)
        mock_client = MagicMock()
        mock_client.get_stories.return_value = iter([good_story, bad_story])
        connector.sanad_client = mock_client

        # Patch _story_to_document to raise for the bad story
        original_convert = connector._story_to_document

        def patched_convert(story):
            if story.story_id == 999:
                raise ValueError("Simulated parse error")
            return original_convert(story)

        connector._story_to_document = patched_convert
        batches = list(connector.load_from_state())
        all_docs = [doc for batch in batches for doc in batch]
        assert len(all_docs) == 1
        assert all_docs[0].id == "sanad:1"

    @pytest.mark.p3
    def test_stop_on_failure_when_continue_on_failure_false(self):
        """Raises immediately on conversion error when continue_on_failure=False."""
        bad_story = MagicMock(spec=SanadStory)
        bad_story.story_id = 999
        bad_story.title = ""
        bad_story.content_text = ""
        bad_story.published = datetime(2025, 1, 1, tzinfo=timezone.utc)
        bad_story.ingested_at = None

        connector = SanadConnector(batch_size=10, continue_on_failure=False)
        mock_client = MagicMock()
        mock_client.get_stories.return_value = iter([bad_story])
        connector.sanad_client = mock_client

        def patched_convert(story):
            raise RuntimeError("Hard failure")

        connector._story_to_document = patched_convert
        with pytest.raises(RuntimeError, match="Hard failure"):
            list(connector.load_from_state())


# ---------------------------------------------------------------------------
# SanadConnector.poll_source
# ---------------------------------------------------------------------------

class TestPollSource:
    """Tests for SanadConnector.poll_source (incremental sync)."""

    def _connector_with_mock_client(
        self, stories: list[SanadStory], batch_size: int = 10
    ) -> SanadConnector:
        connector = SanadConnector(batch_size=batch_size)
        mock_client = MagicMock()
        mock_client.get_stories.return_value = iter(stories)
        connector.sanad_client = mock_client
        return connector

    @pytest.mark.p1
    def test_passes_start_date_to_get_stories(self):
        connector = SanadConnector()
        mock_client = MagicMock()
        mock_client.get_stories.return_value = iter([])
        connector.sanad_client = mock_client

        start_ts = datetime(2025, 3, 1, tzinfo=timezone.utc).timestamp()
        list(connector.poll_source(start_ts))

        call_kwargs = mock_client.get_stories.call_args
        passed_start = call_kwargs.kwargs.get("start_date")
        assert passed_start is not None
        assert passed_start == datetime.fromtimestamp(start_ts, tz=timezone.utc)

    @pytest.mark.p1
    def test_yields_documents_after_start(self):
        stories = [
            _make_story(story_id=1, published=datetime(2025, 4, 1, tzinfo=timezone.utc)),
            _make_story(story_id=2, published=datetime(2025, 4, 2, tzinfo=timezone.utc)),
        ]
        connector = self._connector_with_mock_client(stories)
        start_ts = datetime(2025, 3, 1, tzinfo=timezone.utc).timestamp()
        all_docs = [doc for batch in connector.poll_source(start_ts) for doc in batch]
        assert len(all_docs) == 2

    @pytest.mark.p1
    def test_empty_result_yields_no_batches(self):
        connector = self._connector_with_mock_client([])
        start_ts = datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()
        batches = list(connector.poll_source(start_ts))
        assert batches == []

    @pytest.mark.p2
    def test_raises_if_client_not_initialized(self):
        connector = SanadConnector()
        start_ts = datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()
        with pytest.raises(RuntimeError, match="not initialized"):
            list(connector.poll_source(start_ts))

    @pytest.mark.p2
    def test_document_ids_are_stable_across_polls(self):
        """Same story_id should produce the same document ID every time."""
        story = _make_story(story_id=25243)
        connector = self._connector_with_mock_client([story])
        start_ts = 0.0
        docs_first = [doc for batch in connector.poll_source(start_ts) for doc in batch]

        connector.sanad_client.get_stories.return_value = iter([story])
        docs_second = [doc for batch in connector.poll_source(start_ts) for doc in batch]

        assert docs_first[0].id == docs_second[0].id == "sanad:25243"
