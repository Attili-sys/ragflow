"""
Sanad (Al Jazeera) connector for RAGFlow.

Fetches Arabic news stories from the Sanad API and converts them into
RAGFlow Document objects for indexing and retrieval.

API endpoint: GET https://capi.aljazeera.net/internal/sanad/v1/rest/stories-list
Auth: Header `apikey`
"""

import logging
import time
from collections.abc import Generator
from datetime import datetime, timezone
from typing import Any

import requests

from common.data_source.config import (
    CONTINUE_ON_CONNECTOR_FAILURE,
    INDEX_BATCH_SIZE,
    DocumentSource,
)
from common.data_source.interfaces import LoadConnector, PollConnector
from common.data_source.models import Document, SecondsSinceUnixEpoch

logger = logging.getLogger(__name__)

_SANAD_API_BASE = "https://capi.aljazeera.net/internal/sanad/v1/rest/stories-list"
_MAX_STORIES_PER_PAGE = 20
_REQUEST_DELAY_SECONDS = 0.5
_REQUEST_TIMEOUT_SECONDS = 30
_MAX_PAGES_SAFETY = 5000  # safety cap to prevent infinite loops


class SanadStory:
    """Represents a single Sanad news story."""

    def __init__(
        self,
        story_id: int,
        title: str,
        content: str,
        published: datetime,
        category: str,
        country_name: str,
        country_iso2: str,
        media_type: str,
        source: str,
        copyright: str,
        news_types: list[str],
        ingested_at: datetime | None = None,
    ) -> None:
        self.story_id = story_id
        self.title = title
        self.content = content
        self.published = published
        self.category = category
        self.country_name = country_name
        self.country_iso2 = country_iso2
        self.media_type = media_type
        self.source = source
        self.copyright = copyright
        self.news_types = news_types
        self.ingested_at = ingested_at

    def __str__(self) -> str:
        return (
            f"SanadStory(id={self.story_id}, title={self.title[:60]}..., "
            f"published={self.published.isoformat()})"
        )


class SanadAPI:
    """Client for the Sanad stories-list API."""

    def __init__(self, api_key: str, lang: str = "ar") -> None:
        self.api_key = api_key
        self.lang = lang
        self.session = requests.Session()
        self.session.headers.update({"apikey": self.api_key})

    def get_stories(
        self,
        start_date: datetime | None = None,
        stories_per_page: int = _MAX_STORIES_PER_PAGE,
    ) -> Generator[SanadStory, None, None]:
        """
        Fetch all stories from the API, paginating automatically.

        The API returns stories sorted by publish date descending (newest first).
        If `start_date` is provided, we stop fetching once we encounter a story
        published before that date (used for incremental sync).
        """
        page = 1
        total_fetched = 0

        while page <= _MAX_PAGES_SAFETY:
            params = {
                "lang": self.lang,
                "page": page,
                "story_per_page": min(stories_per_page, _MAX_STORIES_PER_PAGE),
            }

            try:
                resp = self.session.get(
                    _SANAD_API_BASE,
                    params=params,
                    timeout=_REQUEST_TIMEOUT_SECONDS,
                )
                resp.raise_for_status()
            except requests.RequestException as e:
                logger.error(f"Sanad API request failed on page {page}: {e}")
                raise

            data = resp.json()

            # The API returns either a list of stories directly or
            # wraps them in an object. Handle both cases.
            stories_list = data if isinstance(data, list) else data.get("results", data.get("stories", []))

            if not stories_list:
                logger.info(f"No more stories on page {page}. Stopping.")
                break

            reached_cutoff = False
            for raw_story in stories_list:
                story = self._parse_story(raw_story)
                if story is None:
                    continue

                # If we've passed the start_date cutoff, stop
                if start_date and story.published < start_date:
                    logger.info(
                        f"Reached story published at {story.published.isoformat()}, "
                        f"before cutoff {start_date.isoformat()}. Stopping."
                    )
                    reached_cutoff = True
                    break

                total_fetched += 1
                yield story

            if reached_cutoff:
                break

            # If the page returned fewer stories than requested, we've hit the end
            if len(stories_list) < stories_per_page:
                logger.info(f"Received {len(stories_list)} stories (less than {stories_per_page}). Last page.")
                break

            page += 1
            time.sleep(_REQUEST_DELAY_SECONDS)

        logger.info(f"Fetched {total_fetched} stories across {page} page(s)")

    def validate_credentials(self) -> bool:
        """Validate the API key by making a minimal request."""
        try:
            resp = self.session.get(
                _SANAD_API_BASE,
                params={"lang": self.lang, "page": 1, "story_per_page": 1},
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error(f"Sanad credential validation failed: {e}")
            return False

    @staticmethod
    def _parse_story(raw: dict) -> SanadStory | None:
        """Parse a raw API response dict into a SanadStory object."""
        try:
            story_id = raw["id"]
            title = raw.get("title", "")
            content = raw.get("content", "")
            published_str = raw.get("published", "")
            category = raw.get("category", "")

            # Country info
            country = raw.get("country") or {}
            country_name = country.get("name", "")
            country_iso2 = country.get("ISO2", "")

            media_type = raw.get("mediaType", "")
            source = raw.get("source", "")
            copyright_val = raw.get("copyright", "")

            # News types
            news_types = [nt.get("name", "") for nt in (raw.get("newsType") or [])]

            # Parse dates
            if published_str:
                published = datetime.fromisoformat(published_str)
                # Ensure timezone aware (convert to UTC)
                if published.tzinfo is None:
                    published = published.replace(tzinfo=timezone.utc)
            else:
                published = datetime.now(timezone.utc)

            ingested_at = None
            ingested_str = raw.get("ingested_at")
            if ingested_str:
                ingested_at = datetime.fromisoformat(ingested_str)
                if ingested_at.tzinfo is None:
                    ingested_at = ingested_at.replace(tzinfo=timezone.utc)

            return SanadStory(
                story_id=story_id,
                title=title,
                content=content,
                published=published,
                category=category,
                country_name=country_name,
                country_iso2=country_iso2,
                media_type=media_type,
                source=source,
                copyright=copyright_val,
                news_types=news_types,
                ingested_at=ingested_at,
            )
        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to parse Sanad story: {e}. Raw: {raw.get('id', 'unknown')}")
            return None


class SanadConnector(LoadConnector, PollConnector):
    """
    RAGFlow connector for the Al Jazeera Sanad API.

    Implements both LoadConnector (full reindex) and PollConnector (incremental sync).
    Each story is converted to an HTML document for RAGFlow to parse.
    """

    def __init__(
        self,
        lang: str = "ar",
        stories_per_page: int = _MAX_STORIES_PER_PAGE,
        batch_size: int = INDEX_BATCH_SIZE,
        continue_on_failure: bool = CONTINUE_ON_CONNECTOR_FAILURE,
    ) -> None:
        self.lang = lang
        self.stories_per_page = min(stories_per_page, _MAX_STORIES_PER_PAGE)
        self.batch_size = batch_size
        self.continue_on_failure = continue_on_failure
        self.sanad_client: SanadAPI | None = None
        logger.info(
            f"SanadConnector initialized (lang={lang}, "
            f"stories_per_page={self.stories_per_page})"
        )

    def load_credentials(self, credentials: dict[str, Any]) -> dict[str, Any] | None:
        """Load and validate Sanad API credentials."""
        api_key = credentials.get("sanad_api_key")
        if not api_key:
            from common.data_source.exceptions import ConnectorMissingCredentialError
            raise ConnectorMissingCredentialError("sanad")

        self.sanad_client = SanadAPI(api_key=api_key, lang=self.lang)

        if not self.sanad_client.validate_credentials():
            from common.data_source.exceptions import ConnectorValidationError
            raise ConnectorValidationError(
                "Failed to validate Sanad API key. "
                "Ensure the key is correct and you have network access to capi.aljazeera.net."
            )

        logger.info("Sanad credentials loaded and validated")
        return None

    def load_from_state(self) -> Generator[list[Document], None, None]:
        """Full reindex: load all available stories."""
        logger.info("Starting full Sanad reindex (load_from_state)")
        yield from self._fetch_and_batch(start_date=None)

    def poll_source(
        self,
        start: SecondsSinceUnixEpoch,
        end: SecondsSinceUnixEpoch | None = None,
    ) -> Generator[list[Document], None, None]:
        """Incremental sync: fetch stories published since `start`."""
        start_dt = datetime.fromtimestamp(start, tz=timezone.utc)
        logger.info(f"Starting Sanad incremental poll from {start_dt.isoformat()}")
        yield from self._fetch_and_batch(start_date=start_dt)

    def _fetch_and_batch(
        self, start_date: datetime | None
    ) -> Generator[list[Document], None, None]:
        """Core fetch loop: get stories, convert to Documents, yield in batches."""
        if self.sanad_client is None:
            raise RuntimeError("Sanad client not initialized. Call load_credentials() first.")

        docs_batch: list[Document] = []

        for story in self.sanad_client.get_stories(
            start_date=start_date,
            stories_per_page=self.stories_per_page,
        ):
            try:
                doc = self._story_to_document(story)
                docs_batch.append(doc)

                if len(docs_batch) >= self.batch_size:
                    logger.info(f"Yielding batch of {len(docs_batch)} documents")
                    yield docs_batch
                    docs_batch = []

            except Exception as e:
                logger.error(
                    f"Error converting story {story.story_id} to document: {e}",
                    exc_info=True,
                )
                if not self.continue_on_failure:
                    raise

        if docs_batch:
            logger.info(f"Yielding final batch of {len(docs_batch)} documents")
            yield docs_batch

    def _story_to_document(self, story: SanadStory) -> Document:
        """Convert a SanadStory into a RAGFlow Document."""

        # Build an HTML representation of the story.
        # HTML preserves the original structure and handles RTL Arabic well.
        html_parts = [
            '<!DOCTYPE html>',
            '<html lang="ar" dir="rtl">',
            '<head><meta charset="utf-8"></head>',
            '<body>',
            f'<h1>{_escape_html(story.title)}</h1>',
        ]

        # Metadata block
        meta_items = []
        if story.category:
            meta_items.append(f"التصنيف: {_escape_html(story.category)}")
        if story.country_name:
            meta_items.append(f"البلد: {_escape_html(story.country_name)}")
        if story.source:
            meta_items.append(f"المصدر: {_escape_html(story.source)}")
        if story.media_type:
            meta_items.append(f"نوع الوسائط: {_escape_html(story.media_type)}")
        if story.news_types:
            meta_items.append(f"النوع: {_escape_html(', '.join(story.news_types))}")
        if story.published:
            meta_items.append(f"تاريخ النشر: {story.published.isoformat()}")

        if meta_items:
            html_parts.append('<div class="metadata">')
            for item in meta_items:
                html_parts.append(f"<p>{item}</p>")
            html_parts.append("</div>")

        # Story content (already HTML from the API)
        if story.content:
            html_parts.append(f'<div class="content">{story.content}</div>')

        html_parts.extend(["</body>", "</html>"])
        html_content = "\n".join(html_parts)
        blob = html_content.encode("utf-8")

        return Document(
            id=f"sanad:{story.story_id}",
            source=DocumentSource.SANAD,
            semantic_identifier=story.title or f"Sanad Story {story.story_id}",
            extension=".html",
            blob=blob,
            doc_updated_at=story.published,
            size_bytes=len(blob),
            metadata={
                "category": story.category,
                "country": story.country_name,
                "country_iso2": story.country_iso2,
                "source": story.source,
                "media_type": story.media_type,
                "copyright": story.copyright,
                "news_types": story.news_types,
                "sanad_id": story.story_id,
            },
        )


def _escape_html(text: str) -> str:
    """Basic HTML escaping for metadata strings."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Sanad connector test")

    connector = SanadConnector(lang="ar")
    connector.load_credentials({"sanad_api_key": os.environ["SANAD_API_KEY"]})

    logger.info("Loading all documents from Sanad")
    for batch in connector.load_from_state():
        for doc in batch:
            print(f"  {doc.id} | {doc.semantic_identifier[:80]} | {doc.doc_updated_at}")

    # Test incremental poll (last 24 hours)
    import time as _time
    one_day_ago = _time.time() - 24 * 60 * 60
    logger.info("Polling for documents from the last 24 hours")
    for batch in connector.poll_source(one_day_ago):
        for doc in batch:
            print(f"  [poll] {doc.id} | {doc.semantic_identifier[:80]}")

    logger.info("Sanad connector test completed")
