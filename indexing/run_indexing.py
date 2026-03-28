"""Main indexing script - single Solr pipeline with BM25 fields + vector embeddings"""
import logging
import yaml
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from indexing.data_loader import load_jsonl
from indexing.embeddings import EmbeddingModel
from indexing.solr_indexer import SolrIndexer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Load configuration from YAML file"""
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main indexing pipeline"""
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")

        # Initialize embedding model (used by SolrIndexer for vector field)
        logger.info("Initializing embedding model (this may take a minute on first run)...")
        embedding_model = EmbeddingModel(
            model_name=config['embeddings']['model_name'],
            device=config['embeddings']['device']
        )

        # Initialize Solr indexer with embedding model
        logger.info("Initializing Solr indexer (with vector support)...")
        solr_indexer = SolrIndexer(
            solr_url=config['solr']['url'],
            collection_name=config['solr']['collection'],
            timeout=config['solr']['timeout'],
            embedding_model=embedding_model
        )

        # Check Solr connectivity
        if not solr_indexer.ping():
            logger.error("Cannot connect to Solr. Make sure Solr is running (docker-compose up -d)")
            sys.exit(1)

        # Load data
        data_path = config['data']['source_path']
        logger.info(f"Loading data from {data_path}")

        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            sys.exit(1)

        records = list(load_jsonl(data_path))
        logger.info(f"Loaded {len(records)} records from dataset")

        if len(records) == 0:
            logger.error("No records found in dataset")
            sys.exit(1)

        # Ask user confirmation before indexing
        print(f"\n{'='*60}")
        print(f"Ready to index {len(records)} records")
        print(f"  Solr (BM25 + vectors): {config['solr']['url']}/{config['solr']['collection']}")
        print(f"  Embedding model: {config['embeddings']['model_name']}")
        print(f"{'='*60}\n")

        response = input("Proceed with indexing? (yes/no): ").strip().lower()
        if response != 'yes':
            logger.info("Indexing cancelled by user")
            sys.exit(0)

        # Index to Solr with BM25 fields + vector embeddings
        logger.info("=" * 60)
        logger.info("Indexing to Solr with BM25 fields + vector embeddings")
        logger.info("=" * 60)
        solr_indexer.batch_index(
            records,
            batch_size=config['embeddings']['batch_size']  # 32 - matches embedding batch size
        )

        # Summary
        logger.info("=" * 60)
        logger.info("INDEXING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Indexed {len(records)} records to Solr")
        logger.info(f"Solr collection: {config['solr']['url']}/{config['solr']['collection']}")
        logger.info("=" * 60)

        print("\nIndexing completed successfully!")
        print("\nNext steps:")
        print("  1. Start the Flask API: python api/app.py")
        print("  2. Open http://localhost:5000 in your browser")

    except KeyboardInterrupt:
        logger.info("Indexing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
