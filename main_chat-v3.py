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
    # Find the first H1 header in the markdown content.
    match = re.search(r"^# (.+)", markdown, re.MULTILINE)
    # Use the H1 as the title, otherwise fall back to the URL's last segment.
    title = match[1].strip() if match else url.rstrip("/").split("/")[-1] or "Untitled_Page"
    # Sanitize the title to create a valid filename.
    return re.sub(r"[^a-zA-Z0-9_-]", "_", title)


def build_strategy(settings: CrawlerSettings):
    """Construct the appropriate crawl strategy based on user settings."""
    if settings.strategy == "BFS":
        return BFSDeepCrawlStrategy(
            max_depth=settings.max_depth,
            include_external=settings.include_external,
            max_pages=settings.max_pages,
        )
    elif settings.strategy == "BestFirst":
        # Create a scorer only if keywords are provided to guide the crawl.
        scorer = KeywordRelevanceScorer(keywords=settings.keywords, weight=0.7) if settings.keywords else None
        return BestFirstCrawlingStrategy(
            max_depth=settings.max_depth,
            include_external=settings.include_external,
            max_pages=settings.max_pages,
            url_scorer=scorer,  # Prioritize pages with relevant keywords.
        )
    elif settings.strategy == "DFS":
        return DFSDeepCrawlStrategy(
            max_depth=settings.max_depth,
            include_external=settings.include_external,
            max_pages=settings.max_pages,
        )
    else:
        # Raise an error for any unknown strategy type.
        raise ValueError(f"Unknown strategy: {settings.strategy}")


################################ Core Crawling Logic ################################

def update_progress_ui(current: int, total: int, current_url: str):
    """Update the Streamlit UI with the current crawl progress."""
    progress = min(current / total, 1.0)  # Ensure progress value never exceeds 1.0.
    st.session_state.progress_bar.progress(progress)
    st.session_state.status_text.text(f"Crawled {current}/{total}: {current_url}")


async def crawl_website_async(
    settings: CrawlerSettings,
    progress_cb: Callable[[int, int, str], None],
) -> List[Dict]:
    """Asynchronously crawl a website using the specified settings."""
    # Configure the crawler run with chosen strategies and options.
    config = CrawlerRunConfig(
        deep_crawl_strategy=build_strategy(settings),  # Set strategy from sidebar choice.
        scraping_strategy=LXMLWebScrapingStrategy(),  # Use fast lxml for HTML parsing.
        verbose=settings.verbose,
        stream=True,  # Enable streaming to process results as they arrive.
    )
    pages = []
    # Initialize the crawler instance.
    async with AsyncWebCrawler() as crawler:
        # Start the crawl and process each result as it is yielded.
        async for result in await crawler.arun(settings.url, config=config):
            pages.append({"url": result.url, "markdown": result.markdown})
            progress_cb(len(pages), settings.max_pages, result.url)  # Update UI progress.
    return pages


def run_crawl(settings: CrawlerSettings):
    """Run the async crawl function in a synchronous context."""
    return asyncio.run(crawl_website_async(settings, update_progress_ui))


################################ UI Components ################################

def render_sidebar() -> CrawlerSettings:
    """Render the settings sidebar and return user inputs as a dataclass."""
    st.sidebar.header("Crawl Settings")
    url = st.sidebar.text_input("Website URL", value="https://python.langchain.com/docs/")
    max_depth = st.sidebar.slider("Max Crawl Depth", 1, 10, 1)
    max_pages = st.sidebar.number_input("Max Pages to Crawl", 5, 1000, 10)
    strategy = st.sidebar.selectbox("Strategy", ["BFS", "DFS", "BestFirst"])
    include_external = st.sidebar.checkbox("Include External Links")
    keywords_input = st.sidebar.text_input("Keywords (comma‑sep)", "")
    verbose = st.sidebar.checkbox("Verbose Mode", value=True)

    # Package all sidebar settings into a single dataclass instance.
    return CrawlerSettings(
        url=url,
        max_depth=max_depth,
        max_pages=max_pages,
        strategy=strategy,
        include_external=include_external,
        # Parse comma-separated keywords into a clean list of strings.
        keywords=[k.strip() for k in keywords_input.split(",") if k.strip()],
        verbose=verbose,
    )


def render_download_buttons():
    """Render download buttons for a combined markdown file and a zip archive."""
    # Combine all pages into a single markdown string, separated by horizontal rules.
    combined_md = "\n\n---\n\n".join(f"# {p['url']}\n\n{p['markdown']}" for p in st.session_state.pages)
    st.download_button("Download All as Markdown", combined_md, "site.md", "text/markdown")

    # Create a zip archive of individual pages in memory.
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Loop through each page to add it to the archive.
        for i, page in enumerate(st.session_state.pages, 1):
            # Generate a sanitized filename for each page.
            filename = f"{i:03d}_{extract_title(page['markdown'], page['url'])}.md"
            zf.writestr(filename, page["markdown"])  # Add the page content to the zip file.
    st.download_button("Download ZIP of Pages", buf.getvalue(), "pages.zip", "application/zip")


def render_preview():
    """Render a preview of the first crawled page inside an expander."""
    st.subheader("Preview First Page")
    # Place the preview inside a collapsible container.
    with st.expander("Expand Preview"):
        # Display the first 5000 characters of the first page's content.
        st.write(st.session_state.pages[0]["markdown"][:5000])


################################ Main Application ################################

def main():
    """Run the main Streamlit application."""
    st.title("Website Crawler with Crawl4AI")
    st.markdown("Enter a URL, tweak settings, then crawl. Download as MD or ZIP.")

    # Draw the sidebar and get the current user settings.
    settings = render_sidebar()

    # Initialize session state for pages and UI elements if they don't exist.
    if "pages" not in st.session_state:
        st.session_state.pages = []
    if "progress_bar" not in st.session_state:
        st.session_state.progress_bar = st.progress(0)
    if "status_text" not in st.session_state:
        st.session_state.status_text = st.empty()

    # Handle the main "Start Crawling" button action.
    if st.button("Start Crawling", type="primary"):
        # Validate that a URL has been entered.
        if not settings.url:
            st.error("Please enter a valid URL.")
        else:
            # Run the crawl and handle potential errors.
            try:
                pages = run_crawl(settings)
                st.success(f"Crawled {len(pages)} pages!")
                # Store results in the session state to persist them across reruns.
                st.session_state.pages = pages
            except Exception as e:
                st.error(f"An error occurred during crawling: {e}")

    # Show download and preview buttons only if crawl results exist.
    if st.session_state.pages:
        render_download_buttons()
        render_preview()

    st.markdown("**Note**: `pip install crawl4ai` & run `crawl4ai-setup` before first use.")


if __name__ == "__main__":
    main()
