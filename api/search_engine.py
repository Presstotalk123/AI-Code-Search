"""Core search engine with hybrid search (Solr BM25 + ChromaDB semantic)"""
import logging
import time
from typing import Dict, List, Tuple
from collections import Counter
from datetime import datetime
import concurrent.futures
import pysolr
import chromadb
from api.rrf_fusion import RRFFusion

logger = logging.getLogger(__name__)


class SearchEngine:
    """
    Hybrid search engine combining Solr (keyword/BM25) and ChromaDB (semantic/embeddings)
    """

    def __init__(self, config: Dict):
        """
        Initialize search engine

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config

        # Initialize Solr client
        solr_url = f"{config['solr']['url']}/{config['solr']['collection']}"
        self.solr = pysolr.Solr(solr_url, timeout=config['solr']['timeout'])
        logger.info(f"Initialized Solr client: {solr_url}")

        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=config['chromadb']['persist_directory']
        )
        self.chroma_collection = self.chroma_client.get_collection(
            name=config['chromadb']['collection_name']
        )
        logger.info(f"Initialized ChromaDB collection: {config['chromadb']['collection_name']}")

        # Initialize RRF fusion
        self.rrf = RRFFusion(k=config['search']['rrf_k'])

    def search_solr(self, query: str, rows: int = 100) -> List[Dict]:
        """
        BM25 keyword search using Solr

        Args:
            query: Search query string
            rows: Number of results to retrieve

        Returns:
            List of Solr documents
        """
        try:
            results = self.solr.search(
                query,
                **{
                    'rows': rows,
                    'defType': 'edismax',  # Extended DisMax query parser
                    'qf': 'title^2 text combined_content',  # Query fields (boost title 2x)
                    'fl': '*,score'  # Return all fields + relevance score
                }
            )
            docs = list(results)
            logger.debug(f"Solr search returned {len(docs)} results for query: {query}")
            return docs

        except Exception as e:
            logger.error(f"Solr search error: {e}")
            return []

    def search_chroma(self, query: str, n_results: int = 100) -> List[Dict]:
        """
        Semantic vector search using ChromaDB

        Args:
            query: Search query string
            n_results: Number of results to retrieve

        Returns:
            List of documents with metadata
        """
        try:
            results = self.chroma_collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['metadatas', 'distances']
            )

            # Transform to unified format
            formatted_results = []
            if results['metadatas'] and results['metadatas'][0]:
                for i, metadata in enumerate(results['metadatas'][0]):
                    doc = metadata.copy()
                    doc['distance'] = results['distances'][0][i]
                    formatted_results.append(doc)

            logger.debug(f"ChromaDB search returned {len(formatted_results)} results for query: {query}")
            return formatted_results

        except Exception as e:
            logger.error(f"ChromaDB search error: {e}")
            return []

    def search_hybrid(
        self,
        query: str,
        filters: Dict = None,
        mode: str = 'hybrid',
        apply_sentiment_boost: bool = True,
        page: int = 1,
        page_size: int = 10
    ) -> Dict:
        """
        Main search method with multiple modes

        Args:
            query: Search query string
            filters: Dict with keys: date_from, date_to, source, tools, sentiment
            mode: 'keyword', 'semantic', or 'hybrid'
            apply_sentiment_boost: Whether to boost by sentiment
            page: Page number (1-indexed)
            page_size: Results per page

        Returns:
            Dict with: results, total_count, page, page_size, facets, query_time_ms
        """
        start_time = time.time()

        # Execute searches based on mode
        if mode == 'keyword':
            solr_results = self.search_solr(query, rows=self.config['search']['solr_rows'])
            final_results = [(doc['doc_id'], doc.get('score', 0), doc) for doc in solr_results]

        elif mode == 'semantic':
            chroma_results = self.search_chroma(query, n_results=self.config['search']['chroma_n_results'])
            # Convert distance to similarity score (lower distance = higher similarity)
            final_results = [(doc['doc_id'], 1 - doc['distance'], doc) for doc in chroma_results]

        else:  # hybrid - PARALLEL execution for speed
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                solr_future = executor.submit(self.search_solr, query, self.config['search']['solr_rows'])
                chroma_future = executor.submit(self.search_chroma, query, self.config['search']['chroma_n_results'])

                solr_results = solr_future.result()
                chroma_results = chroma_future.result()

            # Fuse results using RRF
            final_results = self.rrf.fuse_results(solr_results, chroma_results)

        # Apply sentiment boosting
        if apply_sentiment_boost and self.config['search']['sentiment_boost']['enabled']:
            final_results = self.rrf.apply_sentiment_boosting(
                final_results,
                self.config['search']['sentiment_boost']
            )

        # Apply filters
        filtered_results = self._apply_filters(final_results, filters)

        # Compute facets
        facets = self._compute_facets(filtered_results)

        # Pagination
        total_count = len(filtered_results)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_results = filtered_results[start_idx:end_idx]

        query_time = time.time() - start_time

        return {
            'results': [self._enrich_result(r) for r in paginated_results],
            'total_count': total_count,
            'page': page,
            'page_size': page_size,
            'total_pages': (total_count + page_size - 1) // page_size,
            'facets': facets,
            'query_time_ms': round(query_time * 1000, 2),
            'mode': mode
        }

    def _apply_filters(self, results: List[Tuple], filters: Dict) -> List[Tuple]:
        """
        Apply post-fusion filters

        Args:
            results: List of (doc_id, score, metadata) tuples
            filters: Filter parameters

        Returns:
            Filtered results
        """
        if not filters:
            return results

        filtered = []

        for doc_id, score, metadata in results:
            # Date range filter
            if filters.get('date_from') or filters.get('date_to'):
                doc_date_str = metadata.get('date', '')
                if doc_date_str:
                    try:
                        doc_date = datetime.fromisoformat(doc_date_str.replace('Z', '+00:00'))

                        if filters.get('date_from'):
                            filter_date_from = datetime.fromisoformat(filters['date_from'])
                            if doc_date < filter_date_from:
                                continue

                        if filters.get('date_to'):
                            filter_date_to = datetime.fromisoformat(filters['date_to'])
                            if doc_date > filter_date_to:
                                continue
                    except Exception as e:
                        logger.warning(f"Date parsing error for {doc_id}: {e}")

            # Tool filter (multi-valued field)
            if filters.get('tools'):
                doc_tools = metadata.get('tool_mentioned', [])

                # Handle both list and string formats
                if isinstance(doc_tools, str):
                    doc_tools = doc_tools.split(',')

                # Check if any of the filter tools are in document tools
                if not any(tool.strip() in [t.strip() for t in doc_tools] for tool in filters['tools'] if tool):
                    continue

            # Sentiment filter
            if filters.get('sentiment'):
                doc_sentiment = metadata.get('sentiment_label', metadata.get('sentiment', ''))
                if doc_sentiment != filters['sentiment']:
                    continue

            # Source filter
            if filters.get('source'):
                doc_source = metadata.get('source', '')
                if doc_source != filters['source']:
                    continue

            filtered.append((doc_id, score, metadata))

        logger.debug(f"Filtered {len(results)} -> {len(filtered)} results")
        return filtered

    def _compute_facets(self, results: List[Tuple]) -> Dict:
        """
        Compute facet counts for filtering UI

        Args:
            results: List of (doc_id, score, metadata) tuples

        Returns:
            Dict with facet counts
        """
        tools_counter = Counter()
        sentiment_counter = Counter()
        source_counter = Counter()
        subreddit_counter = Counter()

        for doc_id, score, metadata in results:
            # Tools (multi-valued)
            tools = metadata.get('tool_mentioned', [])
            if isinstance(tools, str):
                tools = tools.split(',')
            for tool in tools:
                if tool and tool.strip():
                    tools_counter[tool.strip()] += 1

            # Sentiment
            sentiment = metadata.get('sentiment_label', metadata.get('sentiment', 'not_applicable'))
            if sentiment:
                sentiment_counter[sentiment] += 1

            # Source
            source = metadata.get('source', 'reddit')
            if source:
                source_counter[source] += 1

            # Subreddit
            subreddit = metadata.get('subreddit', '')
            if subreddit:
                subreddit_counter[subreddit] += 1

        return {
            'tools': dict(tools_counter.most_common()),
            'sentiment': dict(sentiment_counter),
            'source': dict(source_counter),
            'subreddits': dict(subreddit_counter.most_common(10))  # Top 10 subreddits
        }

    def _enrich_result(self, result_tuple: Tuple) -> Dict:
        """
        Enrich result with snippet generation and formatting

        Args:
            result_tuple: (doc_id, score, metadata)

        Returns:
            Enriched result dict
        """
        doc_id, score, metadata = result_tuple

        # Generate snippet (first 200 chars)
        text = metadata.get('text', '')
        snippet = text[:200] + '...' if len(text) > 200 else text

        # Handle multi-valued fields
        tools = metadata.get('tool_mentioned', [])
        if isinstance(tools, str):
            tools = tools.split(',')
        tools = [t.strip() for t in tools if t.strip()]

        # Parse aspects with polarity
        aspects_raw = metadata.get('aspects', [])
        if isinstance(aspects_raw, str):
            aspects_raw = aspects_raw.split(',')
        aspects_raw = [a.strip() for a in aspects_raw if a.strip()]

        # Parse aspect:polarity format
        aspects_parsed = []
        for aspect_str in aspects_raw:
            if ':' in aspect_str:
                # New format: "aspect:polarity"
                parts = aspect_str.split(':', 1)
                aspects_parsed.append({
                    'name': parts[0],
                    'polarity': parts[1]
                })
            else:
                # Old format: just "aspect"
                aspects_parsed.append({
                    'name': aspect_str,
                    'polarity': None
                })

        return {
            'doc_id': doc_id,
            'score': round(score, 4),
            'title': metadata.get('title', ''),
            'snippet': snippet,
            'url': metadata.get('url', ''),
            'source': metadata.get('source', 'reddit'),
            'date': metadata.get('date', ''),
            'sentiment': metadata.get('sentiment_label', metadata.get('sentiment', 'not_applicable')),
            'tools': tools,
            'aspects': aspects_parsed,
            'subreddit': metadata.get('subreddit', ''),
            'upvotes': metadata.get('upvotes', 0),
            'content_type': metadata.get('content_type', 'post')
        }

    def get_stats(self) -> Dict:
        """
        Get dataset statistics

        Returns:
            Dict with total documents and facet counts
        """
        try:
            # Query Solr for all documents (no results, just count)
            stats = self.solr.search('*:*', rows=0, facet='on', **{
                'facet.field': ['tool_mentioned', 'sentiment_label', 'subreddit'],
                'facet.mincount': 1
            })

            return {
                'total_documents': stats.hits,
                'facets': stats.facets
            }
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {'total_documents': 0, 'facets': {}}

    def health_check(self) -> Dict:
        """
        Check health of search engine components

        Returns:
            Health status dict
        """
        status = {'status': 'healthy', 'components': {}}

        # Check Solr
        try:
            self.solr.ping()
            status['components']['solr'] = 'connected'
        except Exception as e:
            status['components']['solr'] = f'error: {str(e)}'
            status['status'] = 'unhealthy'

        # Check ChromaDB
        try:
            count = self.chroma_collection.count()
            status['components']['chromadb'] = f'connected ({count} documents)'
        except Exception as e:
            status['components']['chromadb'] = f'error: {str(e)}'
            status['status'] = 'unhealthy'

        return status
