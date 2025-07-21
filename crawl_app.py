import streamlit as st
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy


# Function to perform the crawl (async)
async def crawl_website(url: str, max_depth: int = 1, max_pages: int = 10) -> str:
    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=max_depth,
            include_external=False,  # Stay within domain
            max_pages=max_pages,  # Limit total pages
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=True,
    )

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(url, config=config)

    # Concatenate all markdown results
    combined_markdown = ""
    for i, result in enumerate(results):
        combined_markdown += f"# Page {i + 1}: {result.url}\n\n{result.markdown}\n\n---\n\n"

    return combined_markdown


# Streamlit UI
st.title("Website Crawler with Crawl4AI")
st.markdown("Enter a URL to crawl the website and download all pages as Markdown.")

url = st.text_input("Website URL", value="https://example.com")
max_depth = st.slider(
    "Max Crawl Depth",
    min_value=1,
    max_value=5,
    value=1,
    help="How many levels deep to crawl (1 = starting page + direct links).",
)
max_pages = st.slider(
    "Max Pages to Crawl", min_value=5, max_value=50, value=10, help="Limit total pages to prevent excessive crawling."
)

if st.button("Start Crawling"):
    if not url:
        st.error("Please enter a valid URL.")
    else:
        with st.spinner("Crawling website... This may take a while."):
            try:
                combined_md = asyncio.run(crawl_website(url, max_depth, max_pages))
                st.success(f"Crawled {len(combined_md.split('---')) - 1} pages successfully!")

                # Download button
                st.download_button(
                    label="Download Markdown", data=combined_md, file_name="crawled_website.md", mime="text/markdown"
                )

                # Preview first 500 chars
                st.text_area("Markdown Preview", combined_md[:500] + "...", height=200)
            except Exception as e:
                st.error(f"Error during crawling: {str(e)}")

# Installation note
st.markdown(
    "**Note**: Ensure Crawl4AI is installed: `pip install crawl4ai`. Run `crawl4ai-setup` for browser setup if needed."
)
