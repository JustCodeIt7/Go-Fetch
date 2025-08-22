import asyncio
import zipfile
import re
from io import BytesIO
from dataclasses import dataclass
from typing import List, Dict, Callable

import streamlit as st
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import (
    BFSDeepCrawlStrategy,
    DFSDeepCrawlStrategy,
    BestFirstCrawlingStrategy,
)
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy


################################ Streamlit Page Config ################################

# Configure the Streamlit page's title, icon, and layout settings.
st.set_page_config(
    page_title="Go-Fetch - Web Crawler",
    page_icon=":mag_right:",
    layout="wide",
    initial_sidebar_state="expanded",
)


################################ Data & Strategy ################################

@dataclass
class CrawlerSettings:
    """Hold all user-configurable settings for the crawler."""

    url: str
    max_depth: int
    max_pages: int
    strategy: str
    include_external: bool
    keywords: List[str]
    verbose: bool


def extract_title(markdown: str, url: str) -> str:
    """Extract a title from a markdown H1, falling back to the URL slug."""


def build_strategy(settings: CrawlerSettings):
    """Construct the appropriate crawl strategy based on user settings."""


################################ Core Crawling Logic ################################

def update_progress_ui(current: int, total: int, current_url: str):
    """Update the Streamlit UI with the current crawl progress."""


async def crawl_website_async(
    settings: CrawlerSettings,
    progress_cb: Callable[[int, int, str], None],
) -> List[Dict]:
    """Asynchronously crawl a website using the specified settings."""


def run_crawl(settings: CrawlerSettings):
    """Run the async crawl function in a synchronous context."""


################################ UI Components ################################

def render_sidebar() -> CrawlerSettings:
    """Render the settings sidebar and return user inputs as a dataclass."""


def render_download_buttons():
    """Render download buttons for a combined markdown file and a zip archive."""


def render_preview():
    """Render a preview of the first crawled page inside an expander."""


################################ Main Application ################################

def main():
    """Run the main Streamlit application."""


if __name__ == "__main__":
    main()