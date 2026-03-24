"""Solr indexer for Reddit dataset"""
import logging
from typing import Dict, List
import pysolr
from indexing.data_loader import combine_text_content, extract_title

logger = logging.getLogger(__name__)


class SolrIndexer:
    """Index Reddit posts and comments to Solr"""

    def __init__(self, solr_url: str, collection_name: str, timeout: int = 30):
        """
        Initialize Solr client

        Args:
            solr_url: Solr base URL (e.g., http://localhost:8983/solr)
            collection_name: Collection name
            timeout: Request timeout in seconds
        """
        self.solr_url = solr_url
        self.collection_name = collection_name
        self.solr = pysolr.Solr(f"{solr_url}/{collection_name}", timeout=timeout)

        logger.info(f"Initialized Solr indexer for {solr_url}/{collection_name}")

    def transform_record(self, record: Dict) -> Dict:
        """
        Transform JSONL record to Solr document

        Args:
            record: JSON record from dataset

        Returns:
            Solr document dict
        """
        doc = {
            'doc_id': record['doc_id'],
            'title': extract_title(record),
            'text': combine_text_content(record),
            'source': record['source_platform'],
            'url': record['source_url'],
            'content_type': record['content_type'],
            'tool_mentioned': record['labels']['agents'],
            'sentiment_label': record['labels']['polarity'],
            'subjectivity': record['labels']['subjectivity'],
            'aspects': record['labels']['aspects'],
            'sarcasm': record['labels']['sarcasm'],
            'subreddit': record['platform_context']['subreddit'],
            'author': record['author']['username'],
            'date': record['timestamps']['created_at'],
            'upvotes': record['engagement']['upvotes'],
            'num_replies': record['engagement'].get('num_replies', 0)
        }

        return doc

    def batch_index(self, records: List[Dict], batch_size: int = 100):
        """
        Index records in batches for performance

        Args:
            records: List of JSON records
            batch_size: Number of documents per batch
        """
        docs = []
        total_indexed = 0
        total_skipped = 0

        logger.info(f"Starting batch indexing ({batch_size} docs per batch)")

        for i, record in enumerate(records, start=1):
            try:
                doc = self.transform_record(record)
                docs.append(doc)

                if len(docs) >= batch_size:
                    try:
                        self.solr.add(docs)
                        total_indexed += len(docs)
                        logger.info(f"Indexed {total_indexed} documents to Solr")
                        docs = []
                    except Exception as batch_error:
                        logger.error(f"Batch indexing to Solr failed: {batch_error}")
                        # Skip this batch and continue
                        total_skipped += len(docs)
                        docs = []

            except Exception as e:
                logger.warning(f"Skipping record {record.get('doc_id', 'unknown')}: {e}")
                total_skipped += 1
                continue

        # Index remaining documents
        if docs:
            try:
                self.solr.add(docs)
                total_indexed += len(docs)
                logger.info(f"Indexed {total_indexed} documents to Solr")
            except Exception as batch_error:
                logger.error(f"Final batch indexing to Solr failed: {batch_error}")
                total_skipped += len(docs)

        # Commit changes
        try:
            self.solr.commit()
            logger.info(f"Committed {total_indexed} documents to Solr (Skipped: {total_skipped})")
        except Exception as commit_error:
            logger.error(f"Solr commit failed: {commit_error}")

    def clear_collection(self):
        """Delete all documents from collection (use with caution)"""
        logger.warning("Clearing all documents from Solr collection")
        self.solr.delete(q='*:*')
        self.solr.commit()

    def ping(self) -> bool:
        """Check if Solr is accessible"""
        try:
            self.solr.ping()
            return True
        except Exception as e:
            logger.error(f"Solr ping failed: {e}")
            return False
