# Install with pip install firecrawl-py
import asyncio
from firecrawl import AsyncFirecrawlApp
from firecrawl import ScrapeOptions

async def main():
    app = AsyncFirecrawlApp(api_key='fc-2f7944ff36e44505933bc0d7cfb15751')
    response = await app.crawl_url(
        url='https://brendamour.com',
        limit= 100,
        exclude_paths= [ 'blog/.+' ],
    	scrape_options = ScrapeOptions(
            formats= [ 'markdown' ],
            onlyMainContent= True
            parsePDF= True,
            maxAge= 14400000
    	)
    )
    print(response)

asyncio.run(main())