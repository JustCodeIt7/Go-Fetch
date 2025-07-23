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
st.set_page_config(
    page_title="Go-Fetch - Web Crawler",
    page_icon=":mag_right:",
    layout="wide",  # Use the full-width layout
    initial_sidebar_state="expanded",  # Start with the sidebar open
)


################################ Data & Strategy ################################
@dataclass
class CrawlerSettings:
    """Holds all the user-configurable settings for the crawler."""


def extract_title(markdown: str, url: str) -> str:
    """Extracts a title from markdown H1, falling back to the URL slug."""


def build_strategy(settings: CrawlerSettings):
    """Constructs the appropriate crawl strategy based on user settings."""


################################ Core Crawling Logic ################################
def update_progress_ui(current: int, total: int, current_url: str):
    """Callback function to update the Streamlit UI with crawl progress."""


async def crawl_website_async(
    settings: CrawlerSettings,
    progress_cb: Callable[[int, int, str], None],
) -> List[Dict]:
    """Asynchronously crawls a website using the specified settings."""


def run_crawl(settings: CrawlerSettings):
    """Wrapper to run the async crawl function in a synchronous context."""


################################ UI Components ################################
def render_sidebar() -> CrawlerSettings:
    """Renders the settings sidebar and returns user inputs as a dataclass."""


def render_download_buttons():
    """Renders download buttons for combined markdown and a zip of individual files."""


def render_preview():
    """Renders a preview of the first crawled page in an expander."""


################################ Main Application ################################
def main():
    """Main function to run the Streamlit application."""


if __name__ == "__main__":
    main()
