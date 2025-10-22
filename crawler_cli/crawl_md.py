import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter, BM25ContentFilter
from bs4 import BeautifulSoup


async def main():
    # 1. Crawl with minimal or no markdown generator, just get raw HTML
    config = CrawlerRunConfig(
        # If you only want raw HTML, you can skip passing a markdown_generator
        # or provide one but focus on .html in this example
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun("https://example.com/tech-article", config=config)

        if not result.success or not result.html:
            print("Crawl failed or no HTML content.")
            return

        raw_html = result.html

        # 2. First pass: PruningContentFilter on raw HTML
        pruning_filter = PruningContentFilter(threshold=0.5, min_word_threshold=50)

        # filter_content returns a list of "text chunks" or cleaned HTML sections
        pruned_chunks = pruning_filter.filter_content(raw_html)
        # This list is basically pruned content blocks, presumably in HTML or text form

        # For demonstration, let's combine these chunks back into a single HTML-like string
        # or you could do further processing. It's up to your pipeline design.
        pruned_html = "\n".join(pruned_chunks)

        # 3. Second pass: BM25ContentFilter with a user query
        bm25_filter = BM25ContentFilter(
            user_query="machine learning", bm25_threshold=1.2, language="english"
        )

        # returns a list of text chunks
        bm25_chunks = bm25_filter.filter_content(pruned_html)

        if not bm25_chunks:
            print("Nothing matched the BM25 query after pruning.")
            return

        # 4. Combine or display final results
        final_text = "\n---\n".join(bm25_chunks)

        print("==== PRUNED OUTPUT (first pass) ====")
        print(pruned_html[:500], "... (truncated)")  # preview

        print("\n==== BM25 OUTPUT (second pass) ====")
        print(final_text[:500], "... (truncated)")


if __name__ == "__main__":
    asyncio.run(main())
