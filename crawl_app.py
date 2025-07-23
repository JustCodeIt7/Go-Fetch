# crawler_app.py

"""
Website Crawler with Crawl4AI
Refactored for tutorial video:
  • Step 1: Setup and Imports
  • Step 2: Configuration & Utilities
  • Step 3: Crawling Logic
  • Step 4: Streamlit UI
"""

# ─── Step 1: Setup and Imports ────────────────────────────────────────────────
import asyncio
import zipfile
import re
from io import BytesIO
from dataclasses import dataclass
from typing import List, Dict, Callable

import streamlit as st
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy, DFSDeepCrawlStrategy, BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import DFSDeepCrawlStrategy

# Configure Streamlit app
st.set_page_config(
    page_title="Crawl4AI Website Crawler",
    page_icon=":mag_right:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Step 2: Configuration & Utilities ───────────────────────────────────────
@dataclass
class CrawlerSettings:
    url: str
    max_depth: int
    max_pages: int
    strategy: str
    include_external: bool
    keywords: List[str]
    verbose: bool


def extract_title(markdown: str, url: str) -> str:
    """Grab the first '# ' header or fallback to URL slug, then sanitize."""
    match = re.search(r"^# (.+)", markdown, re.MULTILINE)
    title = match[1].strip() if match else url.rstrip("/").split("/")[-1] or "Untitled_Page"
    return re.sub(r"[^a-zA-Z0-9_-]", "_", title)


def build_strategy(settings: CrawlerSettings):
    """Instantiate the selected deep‐crawl strategy."""
    if settings.strategy == "BFS":
        return BFSDeepCrawlStrategy(
            max_depth=settings.max_depth, include_external=settings.include_external, max_pages=settings.max_pages
        )
    elif settings.strategy == "BestFirst":
        scorer = KeywordRelevanceScorer(keywords=settings.keywords, weight=0.7) if settings.keywords else None
        return BestFirstCrawlingStrategy(
            max_depth=settings.max_depth,
            include_external=settings.include_external,
            max_pages=settings.max_pages,
            url_scorer=scorer,
        )
    elif settings.strategy == "DFS":
        return DFSDeepCrawlStrategy(
            max_depth=settings.max_depth, include_external=settings.include_external, max_pages=settings.max_pages
        )
    else:
        raise ValueError(f"Unknown strategy: {settings.strategy}")


def update_progress_ui(current: int, total: int, current_url: str):
    """Streamlit callback to update progress bar & status text."""
    progress = min(current / total, 1.0)
    st.session_state.progress_bar.progress(progress)
    st.session_state.status_text.text(f"Crawled {current}/{total}: {current_url}")


# ─── Step 3: Crawling Logic ─────────────────────────────────────────────────
async def crawl_website_async(settings: CrawlerSettings, progress_cb: Callable[[int, int, str], None]) -> List[Dict]:
    """Run the AsyncWebCrawler and stream progress."""
    config = CrawlerRunConfig(
        deep_crawl_strategy=build_strategy(settings),
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=settings.verbose,
        stream=True,
    )
    pages = []
    async with AsyncWebCrawler() as crawler:
        async for result in await crawler.arun(settings.url, config=config):
            pages.append({"url": result.url, "markdown": result.markdown})
            progress_cb(len(pages), settings.max_pages, result.url)
    return pages


def run_crawl(settings: CrawlerSettings):
    """Sync wrapper around our async crawl function."""
    return asyncio.run(crawl_website_async(settings, update_progress_ui))


# ─── Step 4: Streamlit UI ───────────────────────────────────────────────────
def render_sidebar() -> CrawlerSettings:
    """Render controls and return a filled‐out settings object."""
    st.sidebar.header("Crawl Settings")
    url = st.sidebar.text_input("Website URL", value="https://python.langchain.com/docs/")
    max_depth = st.sidebar.slider("Max Crawl Depth", 1, 10, 1)
    max_pages = st.sidebar.slider("Max Pages to Crawl", 5, 100, 5)
    strategy = st.sidebar.selectbox("Strategy", ["BFS", "DFS", "BestFirst"])
    include_external = st.sidebar.checkbox("Include External Links")
    keywords = st.sidebar.text_input("Keywords (comma‑sep)", "")
    verbose = st.sidebar.checkbox("Verbose Mode", value=True)

    return CrawlerSettings(
        url=url,
        max_depth=max_depth,
        max_pages=max_pages,
        strategy=strategy,
        include_external=include_external,
        keywords=[k.strip() for k in keywords.split(",") if k.strip()],
        verbose=verbose,
    )


def render_download_buttons(st):
    # combined markdown
    combined = "\n\n---\n\n".join(f"# {p['url']}\n\n{p['markdown']}" for p in st.session_state.pages)
    st.download_button("Download All as Markdown", combined, "site.md", "text/markdown")

    # ZIP of individual files
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, p in enumerate(st.session_state.pages, 1):
            name = f"{i:03d}_{extract_title(p['markdown'], p['url'])}.md"
            zf.writestr(name, p["markdown"])
    st.download_button("Download ZIP of Pages", buf.getvalue(), "pages.zip", "application/zip")


def render_preview(st):
    """Render a preview of the first page."""
    # Display progress bar and status
    st.subheader("Preview First Page")
    # add in expanded markdown
    with st.expander("# Expand Preview"):
        st.write(st.session_state.pages[0]["markdown"][:5000])


def main():
    """Entrypoint for the Streamlit app."""
    st.title("Website Crawler with Crawl4AI")
    st.markdown("Enter a URL, tweak settings, then crawl. Download as MD or ZIP.")
    # configure page

    settings = render_sidebar()

    # initialize session state
    if "pages" not in st.session_state:
        st.session_state.pages = []
    if "progress_bar" not in st.session_state:
        st.session_state.progress_bar = st.progress(0)
    if "status_text" not in st.session_state:
        st.session_state.status_text = st.empty()

    if st.button("Start Crawling", type="primary"):
        if not settings.url:
            st.error("Please enter a valid URL.")
        else:
            try:
                pages = run_crawl(settings)
                st.success(f"Crawled {len(pages)} pages!")
                st.session_state.pages = pages
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.pages:
        render_download_buttons(st)
        render_preview(st)

    st.markdown("**Note**: `pip install crawl4ai` & run `crawl4ai-setup` before first use.")


if __name__ == "__main__":
    main()
