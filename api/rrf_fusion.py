"""Reciprocal Rank Fusion (RRF) algorithm for hybrid search"""
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class RRFFusion:
    """
    Reciprocal Rank Fusion for merging results from multiple search systems

    RRF score formula: score(doc) = sum(1 / (k + rank_in_system_i))

    Research: Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
    "Reciprocal rank fusion outperforms condorcet and individual rank learning methods"
    """

    def __init__(self, k: int = 60):
        """
        Initialize RRF fusion

        Args:
            k: Constant to avoid division by zero and smooth ranking differences
               (typical value: 60, empirically validated in IR research)
        """
        self.k = k
        logger.info(f"Initialized RRF fusion with k={k}")

    def compute_rrf_score(self, rank: int) -> float:
        """
        Compute RRF score for a given rank

        Args:
            rank: Document rank (1-indexed)

        Returns:
            RRF score (higher is better)
        """
        return 1.0 / (self.k + rank)

    def fuse_results(
        self,
        solr_results: List[Dict],
        vector_results: List[Dict],
        doc_id_field: str = 'doc_id'
    ) -> List[Tuple[str, float, Dict]]:
        """
        Fuse results from Solr BM25 and Solr KNN vector search using RRF

        Args:
            solr_results: List of Solr documents (ordered by BM25 score)
            vector_results: List of vector search documents (Solr KNN, ordered by cosine similarity)
            doc_id_field: Field name for document ID

        Returns:
            List of (doc_id, rrf_score, merged_metadata) sorted by RRF score descending
        """
        rrf_scores = {}
        doc_data = {}
        keyword_ranks = {}
        vector_ranks = {}

        # Process Solr results
        for rank, doc in enumerate(solr_results, start=1):
            doc_id = doc.get(doc_id_field)
            if not doc_id:
                logger.warning(f"Skipping Solr doc without {doc_id_field}")
                continue

            rrf_score = self.compute_rrf_score(rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_score
            keyword_ranks[doc_id] = rank
            doc_data[doc_id] = doc

        # Process vector results (Solr KNN)
        for rank, doc in enumerate(vector_results, start=1):
            doc_id = doc.get(doc_id_field)
            if not doc_id:
                logger.warning(f"Skipping vector doc without {doc_id_field}")
                continue

            rrf_score = self.compute_rrf_score(rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_score
            vector_ranks[doc_id] = rank

            # Merge metadata if not from Solr (prefer Solr data as it's complete)
            if doc_id not in doc_data:
                doc_data[doc_id] = doc

        # Sort by RRF score descending
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        logger.debug(f"RRF fusion: {len(solr_results)} Solr + {len(vector_results)} vector -> {len(sorted_results)} unique docs")

        # Inject component ranks into metadata so downstream can surface them
        fused = []
        for doc_id, score in sorted_results:
            doc = dict(doc_data[doc_id])  # shallow copy to avoid mutating original
            doc['_keyword_rank'] = keyword_ranks.get(doc_id)
            doc['_vector_rank'] = vector_ranks.get(doc_id)
            fused.append((doc_id, score, doc))
        return fused

    def apply_sentiment_boosting(
        self,
        results: List[Tuple[str, float, Dict]],
        boost_config: Dict[str, float]
    ) -> List[Tuple[str, float, Dict]]:
        """
        Apply sentiment-based score boosting

        Args:
            results: List of (doc_id, score, metadata) tuples
            boost_config: Dict mapping sentiment labels to boost multipliers
                         e.g., {'positive': 1.2, 'negative': 0.8, 'mixed': 1.0}

        Returns:
            Re-ranked results with boosted scores
        """
        boosted_results = []

        for doc_id, score, metadata in results:
            sentiment = metadata.get('sentiment_label', 'not_applicable')
            boost = boost_config.get(sentiment, 1.0)
            boosted_score = score * boost

            boosted_results.append((doc_id, boosted_score, metadata))

        # Re-sort after boosting
        boosted_results.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"Applied sentiment boosting to {len(boosted_results)} results")

        return boosted_results

    def explain_score(self, doc_id: str, solr_rank: int = None, vector_rank: int = None) -> str:
        """
        Generate explanation for RRF score calculation (debugging utility)

        Args:
            doc_id: Document ID
            solr_rank: Rank in Solr BM25 results (1-indexed, None if not present)
            vector_rank: Rank in vector (Solr KNN) results (1-indexed, None if not present)

        Returns:
            Human-readable score explanation
        """
        components = []
        total_score = 0.0

        if solr_rank:
            solr_score = self.compute_rrf_score(solr_rank)
            total_score += solr_score
            components.append(f"Solr(rank={solr_rank}): 1/({self.k}+{solr_rank}) = {solr_score:.4f}")

        if vector_rank:
            vector_score = self.compute_rrf_score(vector_rank)
            total_score += vector_score
            components.append(f"Vector(rank={vector_rank}): 1/({self.k}+{vector_rank}) = {vector_score:.4f}")

        explanation = f"RRF score for '{doc_id}':\n"
        explanation += "\n".join(f"  {comp}" for comp in components)
        explanation += f"\n  Total: {total_score:.4f}"

        return explanation
