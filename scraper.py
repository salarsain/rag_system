from langchain_community.document_loaders import WikipediaLoader, WebBaseLoader
import logging

class DataScraper:
    """
    Scraper to fetch data from Wikipedia or plain URLs
    """
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def scrape_wikipedia(self, query: str, lang: str = "en", load_max_docs: int = 2):
        """
        Scrape Wikipedia articles related to a query.
        """
        try:
            self.logger.info(f"Scraping Wikipedia for: {query}")
            loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
            return loader.load()
        except Exception as e:
            self.logger.error(f"Error scraping Wikipedia for {query}: {str(e)}")
            return []

    def scrape_url(self, url: str):
        """
        Scrape standard website URL.
        """
        try:
            self.logger.info(f"Scraping URL: {url}")
            loader = WebBaseLoader(web_paths=[url])
            return loader.load()
        except Exception as e:
            self.logger.error(f"Error scraping URL {url}: {str(e)}")
            return []
