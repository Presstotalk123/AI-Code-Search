"""Core search engine with hybrid search (Solr BM25 + Solr HNSW KNN semantic)"""
import logging
import time
import json
from typing import Dict, List, Tuple
from collections import Counter
from datetime import datetime, timedelta
import concurrent.futures
import pysolr
from sentence_transformers import SentenceTransformer
from api.rrf_fusion import RRFFusion

logger = logging.getLogger(__name__)


class SearchEngine:
    """
    Hybrid search engine combining Solr BM25 (keyword) and Solr HNSW KNN (semantic).
    Both search paths use the same Solr collection - no second database needed.
    """

    def __init__(self, config: Dict):
        """
        Initialize search engine

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config

        # Initialize Solr client (used for both BM25 and KNN queries)
        solr_url = f"{config['solr']['url']}/{config['solr']['collection']}"
        self.solr = pysolr.Solr(solr_url, timeout=config['solr']['timeout'])
        logger.info(f"Initialized Solr client: {solr_url}")

        # Vector field name from config
        self.vector_field = config['solr'].get('vector_field', 'vector')

        # Initialize sentence-transformers for query embedding at search time
        embed_cfg = config['embeddings']
        logger.info(f"Loading embedding model: {embed_cfg['model_name']}")
        self.embedding_model = SentenceTransformer(
            embed_cfg['model_name'],
            device=embed_cfg.get('device', 'cpu')
        )
        logger.info("Embedding model loaded")

        # Initialize RRF fusion
        self.rrf = RRFFusion(k=config['search']['rrf_k'])

    def _embed_query(self, query: str) -> list:
        """
        Embed a query string into a float list for KNN search.

        Args:
            query: Search query string

        Returns:
            List of 384 floats (L2-normalised for cosine similarity)
        """
        embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]
        return embedding.tolist()

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
                    'qf': 'title^2 text',  # Query fields (boost title 2x)
                    'fl': '*,score'  # Return all fields + relevance score
                }
            )
            docs = list(results)
            logger.debug(f"Solr search returned {len(docs)} results for query: {query}")
            return docs

        except Exception as e:
            logger.error(f"Solr search error: {e}")
            return []

    def search_solr_vector(self, query: str, n_results: int = 100) -> List[Dict]:
        """
        Semantic vector search using Solr's HNSW KNN query parser.

        Query syntax: {!knn f=<field> topK=<n>}<JSON array of floats>
        Score returned is cosine similarity (0.0 - 1.0), no inversion needed.

        Args:
            query: Search query string (embedded at call time)
            n_results: Number of nearest neighbours to retrieve

        Returns:
            List of Solr documents with score as cosine similarity
        """
        query_vector = self._embed_query(query)
        return self._search_vector_by_embedding(query_vector, n_results)

    def _search_vector_by_embedding(self, query_vector: list, n_results: int = 100) -> List[Dict]:
        """
        Semantic vector search using a pre-computed embedding vector.
        Used by hybrid mode to avoid calling SentenceTransformer inside a thread.

        Args:
            query_vector: Pre-computed embedding as Python list of floats
            n_results: Number of nearest neighbours to retrieve

        Returns:
            List of Solr documents with score as cosine similarity
        """
        try:
            # {!knn f=vector topK=N}[float, float, ...]
            # json.dumps produces the required "[0.1, 0.2, ...]" format
            knn_query = f"{{!knn f={self.vector_field} topK={n_results}}}{json.dumps(query_vector)}"

            results = self.solr.search(
                knn_query,
                **{
                    'rows': n_results,
                    'fl': '*,score'  # score = cosine similarity
                }
            )
            docs = list(results)
            logger.debug(f"Solr KNN search returned {len(docs)} results")
            return docs

        except Exception as e:
            logger.error(f"Solr vector search error: {e}")
            return []

    def search_hybrid(
        self,
        query: str,
        filters: Dict = None,
        mode: str = 'hybrid',
        apply_sentiment_boost: bool = True,
        page: int = 1,
        page_size: int = 10,
        min_similarity: float = 0.0
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
            # Solr KNN - score is cosine similarity (0-1), no distance inversion needed
            vector_results = self.search_solr_vector(
                query, n_results=self.config['search']['vector_n_results']
            )
            final_results = [(doc['doc_id'], doc.get('score', 0.0), doc) for doc in vector_results]

        else:  # hybrid - PARALLEL execution for speed
            # Pre-compute embedding before spawning threads (SentenceTransformer is not thread-safe)
            query_vector = self._embed_query(query)

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                solr_future = executor.submit(
                    self.search_solr, query, self.config['search']['solr_rows']
                )
                vector_future = executor.submit(
                    self._search_vector_by_embedding,
                    query_vector, self.config['search']['vector_n_results']
                )
                solr_results = solr_future.result()
                vector_results = vector_future.result()

            # Fuse results using RRF
            final_results = self.rrf.fuse_results(solr_results, vector_results)

        # Apply similarity threshold (semantic and hybrid only; keyword uses BM25 scores, not cosine sim)
        if min_similarity > 0.0:
            if mode == 'semantic':
                final_results = [(d, s, m) for d, s, m in final_results if s >= min_similarity]
            elif mode == 'hybrid':
                final_results = [(d, s, m) for d, s, m in final_results
                                 if m.get('_vector_score', 0.0) >= min_similarity]

        # Apply sentiment boosting
        if apply_sentiment_boost and self.config['search']['sentiment_boost']['enabled']:
            boost_multipliers = {
                k: v for k, v in self.config['search']['sentiment_boost'].items()
                if k != 'enabled'
            }
            final_results = self.rrf.apply_sentiment_boosting(final_results, boost_multipliers)

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
            'results': [self._enrich_result(r, mode) for r in paginated_results],
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

        # Validate date formats upfront (raises ValueError → 400 in route handler)
        if filters.get('date_from'):
            try:
                datetime.strptime(filters['date_from'], '%Y-%m-%d')
            except ValueError:
                raise ValueError(f"Invalid date_from format: '{filters['date_from']}'. Expected YYYY-MM-DD")
        if filters.get('date_to'):
            try:
                datetime.strptime(filters['date_to'], '%Y-%m-%d')
            except ValueError:
                raise ValueError(f"Invalid date_to format: '{filters['date_to']}'. Expected YYYY-MM-DD")

        filtered = []

        for doc_id, score, metadata in results:
            # Date range filter
            if filters.get('date_from') or filters.get('date_to'):
                doc_date_str = metadata.get('date', '')
                # Handle list format (Solr can return multi-valued fields as lists)
                if isinstance(doc_date_str, list):
                    doc_date_str = doc_date_str[0] if doc_date_str else ''
                if doc_date_str:
                    try:
                        doc_date = datetime.fromisoformat(doc_date_str.replace('Z', '+00:00'))

                        if filters.get('date_from'):
                            filter_date_from = datetime.fromisoformat(filters['date_from'] + 'T00:00:00+00:00')
                            if doc_date < filter_date_from:
                                continue

                        if filters.get('date_to'):
                            filter_date_to = datetime.fromisoformat(filters['date_to'] + 'T00:00:00+00:00') + timedelta(days=1)
                            if doc_date >= filter_date_to:
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

    def _enrich_result(self, result_tuple: Tuple, mode: str = 'hybrid') -> Dict:
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

        # Cosine similarity: direct score for semantic, stored _vector_score for hybrid
        if mode == 'semantic':
            similarity_score = round(score, 4)
        elif mode == 'hybrid':
            vs = metadata.get('_vector_score', 0.0)
            similarity_score = round(vs, 4)  # 0.0 means doc appeared only in BM25
        else:
            similarity_score = None

        return {
            'doc_id': doc_id,
            'score': round(score, 4),
            'similarity_score': similarity_score,
            'keyword_rank': metadata.get('_keyword_rank'),
            'vector_rank': metadata.get('_vector_rank'),
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

        # Check Solr BM25
        try:
            self.solr.ping()
            status['components']['solr_bm25'] = 'connected'
        except Exception as e:
            status['components']['solr_bm25'] = f'error: {str(e)}'
            status['status'] = 'unhealthy'

        # Check Solr vector index (query document count)
        try:
            result = self.solr.search('*:*', rows=0)
            status['components']['solr_vector'] = f'connected ({result.hits} documents indexed)'
        except Exception as e:
            status['components']['solr_vector'] = f'error: {str(e)}'
            status['status'] = 'unhealthy'

        return status
