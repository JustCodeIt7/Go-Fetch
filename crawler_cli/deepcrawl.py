import asyncio
import os
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse
import re
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import (
    FilterChain,
    DomainFilter,
    URLPatternFilter,
    ContentTypeFilter,
)
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.components.crawler_monitor import CrawlerMonitor
from crawl4ai.models import CrawlStatus


################################ Configuration ################################

# Crawling configuration
MAX_DEPTH = 10
MAX_PAGES = 100
START_URL = "https://www.promptingguide.ai/"

# Keywords for relevance scoring
KEYWORDS = ["guide", "prompt", "AI", "model", "tutorial", "example"]


################################ Utility Functions ################################


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
    return f"{filename}.md"


async def save_crawled_page(result, output_dir="crawled_pages"):
    """Save only markdown-formatted content for the crawled page."""
    try:
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Generate filename from URL
        filename = sanitize_filename(result.url)
        filepath = os.path.join(output_dir, filename)

        # Save only the clean markdown content
        content = (
            result.markdown
            or result.html
            or "# No Content Available\n\nThe page was crawled but no content was extracted."
        )

        # Save to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return filepath
    except Exception as e:
        print(f"✗ Error saving {result.url}: {str(e)}")
        return None


def update_queue_stats(monitor, results, completed_count):
    """Calculate and update queue statistics based on crawl progress."""
    queued_count = max(0, len(results) - completed_count)

    monitor.update_queue_statistics(
        total_queued=queued_count,
        highest_wait_time=0.0,
        avg_wait_time=0.0,
    )


################################ Main Crawler Function ################################


async def run_advanced_crawler():
    """
    Enhanced deep crawler with integrated monitoring and progress tracking.
    """
    print("\n" + "=" * 70)
    print("🚀 CRAWL4AI DEEP CRAWLER WITH MONITOR")
    print("=" * 70)
    print(f"Starting URL: {START_URL}")
    print(f"Max Depth: {MAX_DEPTH} | Max Pages: {MAX_PAGES}")
    print(f"Keywords: {', '.join(KEYWORDS)}")
    print("=" * 70 + "\n")

    # Initialize the crawler monitor
    monitor = CrawlerMonitor(
        urls_total=MAX_PAGES,
        refresh_rate=0.5,
        enable_ui=True,
    )

    # Start the monitor's rendering loop
    monitor.start()

    # Track crawl statistics
    results = []
    saved_files = []
    failed_urls = []
    task_map = {}  # Map URLs to task IDs
    start_time = time.perf_counter()

    try:
        # Create a sophisticated filter chain
        parsed_start = urlparse(START_URL)
        base_domain = parsed_start.netloc

        filter_chain = FilterChain(
            [
                # Domain boundaries - stay within the same domain
                DomainFilter(
                    allowed_domains=[base_domain],
                ),
                # URL patterns to include
                URLPatternFilter(
                    patterns=["*guide*", "*tutorial*", "*introduction*", "*techniques*"]
                ),
                # Content type filtering
                ContentTypeFilter(allowed_types=["text/html"]),
            ]
        )

        # Create a relevance scorer
        keyword_scorer = KeywordRelevanceScorer(keywords=KEYWORDS, weight=0.8)

        # Set up the configuration
        config = CrawlerRunConfig(
            deep_crawl_strategy=BestFirstCrawlingStrategy(
                max_depth=MAX_DEPTH,
                max_pages=MAX_PAGES,
                include_external=False,
                # filter_chain=filter_chain,
                # url_scorer=keyword_scorer,
            ),
            scraping_strategy=LXMLWebScrapingStrategy(),
            stream=True,
            verbose=False,  # We'll use our own logging
            cache_mode=CacheMode.BYPASS,
        )

        # Execute the crawl with monitoring
        print("Starting deep crawl with real-time monitoring...\n")

        async with AsyncWebCrawler() as crawler:
            async for result in await crawler.arun(START_URL, config=config):
                # Generate unique task ID for this result
                task_id = str(uuid.uuid4())
                task_map[result.url] = task_id

                # Get metadata
                score = result.metadata.get("score", 0)
                depth = result.metadata.get("depth", 0)

                # Add task to monitor
                monitor.add_task(task_id, result.url)

                # Simulate processing time for monitor visualization
                task_start = time.perf_counter()

                # Update task as in progress
                monitor.update_task(
                    task_id=task_id,
                    status=CrawlStatus.IN_PROGRESS,
                    start_time=task_start,
                    wait_time=0.1,
                )

                # Check if crawl was successful
                if result.success:
                    # Save the crawled page
                    filepath = await save_crawled_page(result)

                    task_end = time.perf_counter()
                    process_time = task_end - task_start

                    if filepath:
                        saved_files.append(filepath)
                        results.append(result)

                        # Update task as completed
                        monitor.update_task(
                            task_id=task_id,
                            status=CrawlStatus.COMPLETED,
                            end_time=task_end,
                            memory_usage=0.0,
                        )

                        print(
                            f"✓ Depth:{depth} | Score:{score:.2f} | Time:{process_time:.2f}s | {os.path.basename(filepath)}"
                        )
                    else:
                        # Save failed but file write failed
                        failed_urls.append(result.url)
                        monitor.update_task(
                            task_id=task_id,
                            status=CrawlStatus.FAILED,
                            end_time=task_end,
                            memory_usage=0.0,
                            error_message="File save error",
                        )
                        print(f"✗ Failed to save: {result.url}")
                else:
                    # Crawl failed
                    task_end = time.perf_counter()
                    error_msg = result.error_message or "Unknown error"
                    failed_urls.append(result.url)

                    monitor.update_task(
                        task_id=task_id,
                        status=CrawlStatus.FAILED,
                        end_time=task_end,
                        memory_usage=0.0,
                        error_message=error_msg,
                    )
                    print(f"✗ Crawl failed: {result.url} - {error_msg}")

                # Update queue statistics
                update_queue_stats(monitor, results, len(saved_files))

                # Update memory status based on progress
                progress = len(results) / MAX_PAGES if MAX_PAGES > 0 else 0
                if progress > 0.8:
                    monitor.update_memory_status("PRESSURE")
                elif progress > 0.5:
                    monitor.update_memory_status("NORMAL")
                else:
                    monitor.update_memory_status("NORMAL")

        # Wait a moment to view final monitor state
        print("\n\nWaiting to view final state...")
        time.sleep(3)

    except KeyboardInterrupt:
        print("\n\n⚠️  Crawl interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Error during crawl: {str(e)}")
        import traceback

        traceback.print_exc()
    finally:
        # Stop the monitor
        monitor.stop()

        duration = time.perf_counter() - start_time

        # Print comprehensive summary
        print("\n" + "=" * 70)
        print("📊 CRAWL SUMMARY")
        print("=" * 70)

        print(f"\n⏱️  Duration: {duration:.2f} seconds")
        print(f"✅ Successfully crawled: {len(results)} pages")
        print(f"💾 Files saved: {len(saved_files)}")
        print(f"❌ Failed: {len(failed_urls)}")

        if results:
            avg_score = sum(r.metadata.get("score", 0) for r in results) / len(results)
            print(f"📈 Average relevance score: {avg_score:.2f}")
            print(f"⚡ Average speed: {len(results) / duration:.2f} pages/second")

        # Group by depth
        depth_counts = {}
        for result in results:
            depth = result.metadata.get("depth", 0)
            depth_counts[depth] = depth_counts.get(depth, 0) + 1

        if depth_counts:
            print("\n📊 Pages crawled by depth:")
            for depth, count in sorted(depth_counts.items()):
                print(f"  Depth {depth}: {count} pages")

        # Show failed URLs if any
        if failed_urls:
            print(f"\n❌ Failed URLs ({len(failed_urls)}):")
            for url in failed_urls[:5]:  # Show first 5
                print(f"  - {url}")
            if len(failed_urls) > 5:
                print(f"  ... and {len(failed_urls) - 5} more")

        # Get monitor summary
        summary = monitor.get_summary()
        status_counts = summary.get("status_counts", {})

        if status_counts:
            print("\n📋 Task Status Summary:")
            for status, count in status_counts.items():
                print(f"  {status}: {count}")

        print(f"\n📁 Output directory: {os.path.abspath('crawled_pages')}/")
        print("=" * 70)
        print("🎉 Deep crawl with monitoring complete!\n")


if __name__ == "__main__":
    asyncio.run(run_advanced_crawler())
