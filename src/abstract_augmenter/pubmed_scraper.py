"""
Fetches and processes scientific articles from PubMed.
Supports query building, batch fetching of XML data via Entrez API, 
recursive text extraction from abstracts, and rule-based filtering.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from time import sleep
import logging
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional, Tuple, List, Union

import requests

from ..utils.io import load_json, save_json

# Abstract Base Class for Article Scrapers
class AbstractArticleScraper(ABC):
    def __init__(self, scraper_config: Dict[str, Any], output_dir: Path) -> None:
        """Initializes the scraper with configuration and output path."""
        self.scraper_config = scraper_config
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True) # Ensure output directory exists

    def get_data_from_file(self, scraper_output_file: Path) -> Optional[Dict[str, Any]]:
        """Loads the data from a scraper's output file."""
        logging.info(f"Loading abstracts from {scraper_output_file.name}...")
        data = load_json(scraper_output_file)
        logging.info(f"{data.get('results_count', 0)} results loaded.")
        return data

    @abstractmethod
    def scrape_articles(self) -> Path:
        """
        Abstract method to be implemented by concrete scraper classes.
        Should perform the scraping and return the path to the output file.
        """
        pass

class PubMedScraper(AbstractArticleScraper):
    """
    A scraper for fetching and processing articles from PubMed.
    It builds a query, fetches article data, and applies post-processing filters
    based on the provided configuration.
    """
    def __init__(self, scraper_config: Dict[str, Any], output_dir: Path) -> None:
        super().__init__(scraper_config, output_dir)
        self.api_params = self.scraper_config["api_params"]

    def _build_query(self) -> str:
        """
        Builds the PubMed search query from the configuration, excluding
        filters that will be applied in post-processing.
        """
        if self.scraper_config.get("user_query"):
            return self.scraper_config["user_query"]

        search_params = self.scraper_config["search_params"]
        main_terms = ' OR '.join([f'"{term}"[All Fields]' for term in search_params["main_terms"]])
        secondary_terms = ' OR '.join([f'"{term}"[All Fields]' for term in search_params["secondary_terms"]])
        start_year = search_params["start_year"]
        end_year = search_params["end_year"]
        
        query = (
            f'({main_terms}) '
            f'AND ({secondary_terms}) '
            f'AND ({start_year}:{end_year}[dp])'
        )
        return query

    def _filter_articles(self, articles: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Filters a dictionary of articles based on the configuration."""
        filter_config = self.scraper_config["search_params"].get("filters", {})
        required_keywords = filter_config.get("required_keywords_in_title", [])
        excluded_types = filter_config.get("excluded_publication_types", [])
        
        filtered_articles = {}
        for doi, data in articles.items():
            # Require keywords in the title
            if required_keywords:
                title = data.get("title", "").lower()
                if not any(kw.lower() in title for kw in required_keywords):
                    continue
            
            # Exclude publication types
            if excluded_types:
                types = [pt.lower() for pt in data.get("publication_types", [])]
                if any(et.lower() in types for et in excluded_types):
                    continue
            
            # Require that the abstract exists (already done in _parse_article_xml, but as a double-check)
            if not data.get("abstract"):
                continue
            
            filtered_articles[doi] = data
        return filtered_articles

    @staticmethod
    def _extract_full_text(element: Optional[ET.Element]) -> str:
        """Extracts all text from an XML element, including sub-tags."""
        if element is None:
            return ""
        parts = [element.text] if element.text else []
        for child in element:
            parts.append(PubMedScraper._extract_full_text(child))
            if child.tail:
                parts.append(child.tail)
        return "".join(parts).strip()

    @staticmethod
    def _parse_article_xml(article_element: ET.Element) -> Optional[Dict[str, Any]]:
        """Parses a 'PubmedArticle' XML element and returns a dictionary with the data."""
        medline = article_element.find("MedlineCitation")
        if medline is None: return None
        article_info = medline.find("Article")
        if article_info is None: return None

        doi = next((eid.text.strip().lower() for eid in article_info.findall(".//ELocationID") if eid.attrib.get("EIdType") == "doi"), None)
        
        abstract_text_el = article_info.find("Abstract")
        abstract = None
        if abstract_text_el is not None:
            parts = []
            for t in abstract_text_el.findall("AbstractText"):
                text_content = PubMedScraper._extract_full_text(t)
                if text_content:
                    label = t.attrib.get("Label")
                    parts.append(f"{label}: {text_content}" if label else text_content)
            abstract = " ".join(parts) if parts else None

        if not doi or not abstract:
            return None

        title = PubMedScraper._extract_full_text(article_info.find("ArticleTitle"))
        authors = [f"{author.findtext('ForeName')} {author.findtext('LastName')}" for author in article_info.findall(".//Author") if author.findtext('LastName') and author.findtext('ForeName')]
        year = (year_el.text if (year_el := article_info.find(".//Journal/JournalIssue/PubDate/Year")) is not None else None)
        journal = (journal_el.text.strip() if (journal_el := article_info.find(".//Journal/Title")) is not None else None)
        pub_types = [pt.text for pt in article_info.findall(".//PublicationType")]

        return {"doi": doi, "title": title, "abstract": abstract, "authors": authors, "year": year, "journal": journal, "publication_types": pub_types}

    def scrape_article_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """
        Scrapes a single article from PubMed by its DOI by reusing the parsing logic.

        Args:
            doi (str): The DOI of the article to fetch.

        Returns:
            dict | None: A dictionary containing the article data if found, otherwise None.
        """
        logging.info(f"--- Fetching single article by DOI: {doi} ---")
        base_url = self.api_params["base_url"]
        sleep_seconds = self.api_params.get("sleep_seconds", 0.2)
        
        try:
            # 1. Search for the PubMed ID (PMID) using the DOI as an article identifier.
            esearch_params = {"db": "pubmed", "term": f'"{doi}"[aid]', "retmax": "1", "retmode": "json"}
            response = requests.get(base_url + "esearch.fcgi", params=esearch_params)
            response.raise_for_status()
            ids = response.json()["esearchresult"].get("idlist", [])
            
            if not ids:
                logging.warning(f"No PubMed ID found for DOI: {doi}")
                return None
            
            # 2. Fetch the article data using the found ID and parse it.
            logging.info(f"Found PMID: {ids[0]} for DOI: {doi}. Fetching details...")
            sleep(sleep_seconds) # Be respectful to the API
            efetch_params = {"db": "pubmed", "id": ids[0], "retmode": "xml"}
            resp = requests.get(base_url + "efetch.fcgi", params=efetch_params)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            
            article_element = root.find(".//PubmedArticle")
            return self._parse_article_xml(article_element) if article_element else None
            
        except requests.exceptions.RequestException as e:
            logging.error(f"An API error occurred while fetching DOI {doi}: {e}")
            return None

    def scrape_articles(self) -> Tuple[Path, Dict[str, Any]]:
        """Executes the complete PubMed scraping process with post-processing filters."""
        logging.info("--- Starting PubMed Scraper ---")
        query = self._build_query()
        
        # Use timestamp for filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = self.output_dir / f"pubmed_{timestamp}.json"
            
        logging.info(f"Executing query on PubMed: {query}")

        try:
            base_url = self.api_params["base_url"]
            response = requests.get(base_url + "esearch.fcgi", params={"db": "pubmed", "term": query, "retmax": self.api_params["esearch_retmax"], "retmode": "json"})
            response.raise_for_status()
            ids = response.json()["esearchresult"]["idlist"]
            logging.info(f"Found {len(ids)} paper IDs matching the query.")

            all_papers = {}
            efetch_batch_size = self.api_params["efetch_batch_size"]
            logging.info(f"Fetching details in batches of {efetch_batch_size}...")
            for i in range(0, len(ids), efetch_batch_size):
                batch_ids = ids[i:i+efetch_batch_size]
                resp = requests.get(base_url + "efetch.fcgi", params={"db": "pubmed", "id": ",".join(batch_ids), "retmode": "xml"})
                resp.raise_for_status()
                root = ET.fromstring(resp.text)
                for article_element in root.findall(".//PubmedArticle"):
                    if (article_data := self._parse_article_xml(article_element)):
                        all_papers[article_data["doi"]] = article_data
                logging.info(f"Processed batch {i//efetch_batch_size + 1}")
                sleep(self.api_params["sleep_seconds"])

            logging.info(f"Applying filters to {len(all_papers)} collected articles...")
            filtered_papers = self._filter_articles(all_papers)
            logging.info(f"Filtering complete. Kept {len(filtered_papers)} of {len(all_papers)} articles.")

            search_metadata = {"database": "PubMed", "query": query, "date": datetime.now().strftime("%Y-%m-%d"), "results_found": len(ids), "results_processed": len(all_papers), "results_filtered": len(filtered_papers), "query_settings": self.scraper_config["search_params"], "results": filtered_papers}
            save_json(search_metadata, output_file)
            logging.info(f"Saved {len(filtered_papers)} filtered results to {output_file}")
            return output_file, search_metadata

        except requests.exceptions.RequestException as e:
            logging.error(f"An API error occurred during scraping: {e}")
            raise
