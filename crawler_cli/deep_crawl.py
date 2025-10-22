import asyncio
import os
from pathlib import Path
from urllib.parse import urlparse
import re
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import (
    FilterChain,
    DomainFilter,
    URLPatternFilter,
    ContentTypeFilter,
)
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer


def sanitize_filename(url):
    """Create a safe filename from a URL."""
    parsed = urlparse(url)
    # Combine domain and path
    filename = parsed.netloc + parsed.path
    # Remove trailing slash
    filename = filename.rstrip("/")
    # Replace invalid characters with underscores
    filename = re.sub(r"[^\w\-_.]", "_", filename)
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    # Add .md extension
    return filename + ".md"


async def save_crawled_page(result, output_dir="crawled_pages"):
    """Save the crawled page content to a file."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate filename from URL
    filename = sanitize_filename(result.url)
    filepath = os.path.join(output_dir, filename)

    # Prepare content to save
    content = f"""# {result.url}

**Depth:** {result.metadata.get("depth", 0)}
**Score:** {result.metadata.get("score", 0):.2f}
**Crawled at:** {result.metadata.get("timestamp", "N/A")}

---

## Content

{result.markdown or result.html}
"""

    # Save to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✓ Saved: {filepath}")
    return filepath


async def run_advanced_crawler():
    # Create a sophisticated filter chain
    filter_chain = FilterChain(
        [
            # Domain boundaries
            DomainFilter(
                allowed_domains=["docs.example.com"],
                blocked_domains=["old.docs.example.com"],
            ),
            # URL patterns to include
            URLPatternFilter(patterns=["*guide*", "*tutorial*", "*blog*"]),
            # Content type filtering
            ContentTypeFilter(allowed_types=["text/html"]),
        ]
    )

    # Create a relevance scorer
    keyword_scorer = KeywordRelevanceScorer(
        keywords=["crawl", "example", "async", "configuration"], weight=0.7
    )

    # Set up the configuration
    config = CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(
            max_depth=7,
            max_pages=10,
            include_external=False,
            # filter_chain=filter_chain,
            # url_scorer=keyword_scorer,
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        stream=True,
        verbose=True,
    )

    # Execute the crawl
    results = []
    saved_files = []
    async with AsyncWebCrawler() as crawler:
        async for result in await crawler.arun(
            "https://www.promptingguide.ai/", config=config
        ):
            results.append(result)
            score = result.metadata.get("score", 0)
            depth = result.metadata.get("depth", 0)
            print(f"Depth: {depth} | Score: {score:.2f} | {result.url}")

            # Save the crawled page to a file
            filepath = await save_crawled_page(result)
            saved_files.append(filepath)

    # Analyze the results
    print(f"Crawled {len(results)} high-value pages")
    print(
        f"Average score: {sum(r.metadata.get('score', 0) for r in results) / len(results):.2f}"
    )

    # Group by depth
    depth_counts = {}
    for result in results:
        depth = result.metadata.get("depth", 0)
        depth_counts[depth] = depth_counts.get(depth, 0) + 1

    print("Pages crawled by depth:")
    for depth, count in sorted(depth_counts.items()):
        print(f"  Depth {depth}: {count} pages")

    print(f"\nSaved {len(saved_files)} files to 'crawled_pages/' directory")


if __name__ == "__main__":
    asyncio.run(run_advanced_crawler())
