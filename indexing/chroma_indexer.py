"""ChromaDB indexer for Reddit dataset"""
import logging
from typing import Dict, List
import chromadb
from indexing.data_loader import combine_text_content

logger = logging.getLogger(__name__)


class ChromaIndexer:
    """Index Reddit posts and comments to ChromaDB with embeddings"""

    def __init__(self, persist_directory: str, collection_name: str, embedding_model):
        """
        Initialize ChromaDB client

        Args:
            persist_directory: Directory for persistent storage
            collection_name: Collection name
            embedding_model: EmbeddingModel instance
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        logger.info(f"Initializing ChromaDB at {persist_directory}")

        # Create persistent client
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_model.chroma_ef,
            metadata={"hnsw:space": "cosine"}  # Cosine similarity
        )

        logger.info(f"Collection '{collection_name}' ready (current count: {self.collection.count()})")

    def index_record(self, record: Dict, embedding_text: str):
        """
        Index single record with embedding

        Args:
            record: JSON record from dataset
            embedding_text: Text to embed (combined content)
        """
        metadata = {
            'doc_id': record['doc_id'],
            'source': record['source_platform'],
            'url': record['source_url'],
            'sentiment': record['labels']['polarity'],
            'tools': ','.join(record['labels']['agents']),
            'date': record['timestamps']['created_at'],
            'subreddit': record['platform_context']['subreddit'],
            'aspects': ','.join(record['labels']['aspects']),
            'content_type': record['content_type'],
            'title': record['content'].get('thread_title', '')
        }

        self.collection.add(
            documents=[embedding_text],
            metadatas=[metadata],
            ids=[record['doc_id']]
        )

    def _clean_metadata(self, metadata: Dict) -> Dict:
        """
        Clean metadata to ensure ChromaDB compatibility
        - Remove None values
        - Convert all values to str, int, float, or bool

        Args:
            metadata: Raw metadata dict

        Returns:
            Cleaned metadata dict
        """
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                cleaned[key] = ''  # Convert None to empty string
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                # Convert other types to string
                cleaned[key] = str(value)
        return cleaned

    def batch_index(self, records: List[Dict], batch_size: int = 32):
        """
        Batch index for performance

        Args:
            records: List of JSON records
            batch_size: Number of documents per batch (smaller for embeddings)
        """
        documents = []
        metadatas = []
        ids = []
        total_indexed = 0
        total_skipped = 0

        logger.info(f"Starting batch indexing to ChromaDB ({batch_size} docs per batch)")

        for i, record in enumerate(records, start=1):
            try:
                text = combine_text_content(record)

                # Skip if text is empty
                if not text or text == "[No text content]":
                    logger.warning(f"Skipping record {record.get('doc_id', 'unknown')}: Empty text content")
                    total_skipped += 1
                    continue

                # Build metadata with default values for None
                metadata = {
                    'doc_id': record.get('doc_id', ''),
                    'source': record.get('source_platform', 'reddit'),
                    'url': record.get('source_url', ''),
                    'sentiment': record.get('labels', {}).get('polarity', 'not_applicable'),
                    'tools': ','.join(record.get('labels', {}).get('agents', [])),
                    'date': record.get('timestamps', {}).get('created_at', ''),
                    'subreddit': record.get('platform_context', {}).get('subreddit', ''),
                    'aspects': ','.join(record.get('labels', {}).get('aspects', [])),
                    'content_type': record.get('content_type', 'post'),
                    'title': record.get('content', {}).get('thread_title', '')
                }

                # Clean metadata to remove None values
                metadata = self._clean_metadata(metadata)

                documents.append(text)
                metadatas.append(metadata)
                ids.append(record['doc_id'])

                if len(documents) >= batch_size:
                    try:
                        self.collection.add(
                            documents=documents,
                            metadatas=metadatas,
                            ids=ids
                        )
                        total_indexed += len(documents)
                        logger.info(f"Indexed {total_indexed} documents to ChromaDB")
                        documents, metadatas, ids = [], [], []
                    except Exception as batch_error:
                        logger.error(f"Batch indexing failed: {batch_error}")
                        # Skip this batch and continue
                        total_skipped += len(documents)
                        documents, metadatas, ids = [], [], []

            except Exception as e:
                logger.warning(f"Skipping record {record.get('doc_id', 'unknown')}: {e}")
                total_skipped += 1
                continue

        # Index remaining documents
        if documents:
            try:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                total_indexed += len(documents)
                logger.info(f"Indexed {total_indexed} documents to ChromaDB")
            except Exception as batch_error:
                logger.error(f"Final batch indexing failed: {batch_error}")
                total_skipped += len(documents)

        logger.info(f"ChromaDB indexing complete. Total indexed: {total_indexed}, Skipped: {total_skipped}, Collection count: {self.collection.count()}")

    def clear_collection(self):
        """Delete collection (use with caution)"""
        logger.warning(f"Deleting collection: {self.collection_name}")
        self.client.delete_collection(name=self.collection_name)

    def get_count(self) -> int:
        """Get total document count"""
        return self.collection.count()
