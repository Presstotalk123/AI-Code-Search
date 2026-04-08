"""Solr indexer for Reddit dataset - includes vector embeddings for HNSW search"""
import logging
from typing import Dict, List, Optional
import pysolr
from indexing.data_loader import combine_text_content, extract_title

logger = logging.getLogger(__name__)


class SolrIndexer:
    """Index Reddit posts and comments to Solr with BM25 fields + vector embeddings"""

    def __init__(self, solr_url: str, collection_name: str, timeout: int = 30, embedding_model=None):
        """
        Initialize Solr client

        Args:
            solr_url: Solr base URL (e.g., http://localhost:8983/solr)
            collection_name: Collection name
            timeout: Request timeout in seconds
            embedding_model: EmbeddingModel instance (required for vector indexing)
        """
        self.solr_url = solr_url
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.solr = pysolr.Solr(f"{solr_url}/{collection_name}", timeout=timeout)

        if embedding_model is None:
            logger.warning(
                "SolrIndexer initialized WITHOUT an embedding_model. "
                "Documents will be indexed without vector embeddings — HNSW/KNN semantic search will return no results. "
                "Pass an EmbeddingModel instance to enable vector indexing."
            )
        logger.info(f"Initialized Solr indexer for {solr_url}/{collection_name}")

    def _transform_aspects(self, aspects) -> List[str]:
        """Transform aspects to aspect:polarity format.
        Handles both old (list) and new (dict) formats for backward compatibility."""
        if not aspects:
            return []

        # New format: dict mapping aspect to polarity
        if isinstance(aspects, dict):
            return [f"{aspect}:{polarity}" for aspect, polarity in aspects.items()]

        # Old format: simple list - keep as-is
        if isinstance(aspects, list):
            return aspects

        # Fallback for unexpected types
        logger.warning(f"Unexpected aspects type: {type(aspects)}")
        return []

    def transform_record(self, record: Dict) -> Dict:
        """
        Transform JSONL record to Solr document

        Args:
            record: JSON record from dataset

        Returns:
            Solr document dict (without vector field - added during batch_index)
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
            'aspects': self._transform_aspects(record['labels']['aspects']),
            'sarcasm': record['labels']['sarcasm'],
            'subreddit': record['platform_context']['subreddit'],
            'author': record['author']['username'],
            'date': record['timestamps']['created_at'],
            'upvotes': record['engagement']['upvotes'],
            'num_replies': record['engagement'].get('num_replies', 0)
        }

        return doc

    def batch_index(self, records: List[Dict], batch_size: int = 32):
        """
        Index records in batches. Computes and stores vector embeddings per batch
        if embedding_model is set.

        Args:
            records: List of JSON records
            batch_size: Documents per batch (32 to match embedding batch size)
        """
        docs = []
        texts_for_embedding = []
        total_indexed = 0
        total_skipped = 0

        logger.info(f"Starting batch indexing with vectors ({batch_size} docs per batch)")

        def flush_batch(doc_list, text_list):
            if not doc_list:
                return 0

            # Compute and attach vector embeddings
            if self.embedding_model is not None and text_list:
                try:
                    embeddings = self.embedding_model.encode(text_list, batch_size=batch_size)
                    for doc, emb in zip(doc_list, embeddings):
                        doc['vector'] = emb.tolist()  # numpy float32 -> Python list
                except Exception as emb_error:
                    logger.error(f"Embedding failed for batch: {emb_error}")
                    logger.warning("Indexing batch without vector embeddings")

            try:
                self.solr.add(doc_list)
                logger.info(f"Indexed batch of {len(doc_list)} documents")
                return len(doc_list)
            except Exception as batch_error:
                logger.error(f"Batch indexing to Solr failed: {batch_error}")
                return 0

        for i, record in enumerate(records, start=1):
            try:
                text = combine_text_content(record)
                doc = self.transform_record(record)
                docs.append(doc)
                texts_for_embedding.append(text)

                if len(docs) >= batch_size:
                    count = flush_batch(docs, texts_for_embedding)
                    total_indexed += count
                    total_skipped += len(docs) - count
                    docs, texts_for_embedding = [], []

            except Exception as e:
                logger.warning(f"Skipping record {record.get('doc_id', 'unknown')}: {e}")
                total_skipped += 1
                continue

        # Flush remaining documents
        if docs:
            count = flush_batch(docs, texts_for_embedding)
            total_indexed += count
            total_skipped += len(docs) - count

        # Commit changes
        try:
            self.solr.commit()
            logger.info(f"Committed. Total indexed: {total_indexed}, Skipped: {total_skipped}")
        except Exception as commit_error:
            logger.error(f"Solr commit failed: {commit_error}")

        # Merge all Lucene segments into one.
        # IMPORTANT: Solr/Lucene builds a separate HNSW graph per segment. If multiple
        # segments exist (e.g. from autoCommit firing during long indexing runs), KNN
        # search retrieves topK candidates per segment independently and merges — this
        # degrades recall compared to a single global HNSW graph (as ChromaDB uses).
        # optimize() forces a full segment merge so the entire corpus is one HNSW graph.
        try:
            logger.info("Optimizing Solr index (merging segments for best HNSW recall)...")
            self.solr.optimize()
            logger.info("Index optimization complete — single HNSW graph over all documents")
        except Exception as opt_error:
            logger.error(f"Solr optimize failed: {opt_error}")

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
