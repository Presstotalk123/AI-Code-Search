"""Reciprocal Rank Fusion (RRF) algorithm for hybrid search"""
import logging
from typing import List, Dict, Tuple, Optional

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
        vector_scores = {}  # cosine similarity per doc from KNN results

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
            vector_scores[doc_id] = doc.get('score', 0.0)  # cosine similarity

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
            doc['_vector_score'] = vector_scores.get(doc_id, 0.0)  # cosine similarity for threshold filtering
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

    def apply_aspect_boosting(
        self,
        results: List[Tuple[str, float, Dict]],
        detected_aspects: List[str],
        boost_multiplier: float = 1.3
    ) -> List[Tuple[str, float, Dict]]:
        """
        Boost documents whose indexed aspects overlap with query-detected aspects.

        Aspects are stored in Solr as multi-valued strings in "aspect_name:polarity"
        or plain "aspect_name" format. If any detected aspect matches an indexed
        aspect on the document, the RRF score is multiplied by boost_multiplier.

        Args:
            results: List of (doc_id, score, metadata) tuples
            detected_aspects: Canonical aspect names detected from the query
            boost_multiplier: Score multiplier for matching documents (e.g. 1.3 = +30%)

        Returns:
            Re-ranked results with aspect-boosted scores
        """
        if not detected_aspects:
            return results

        detected = set(detected_aspects)
        boosted = []

        for doc_id, score, metadata in results:
            aspects_raw = metadata.get('aspects', [])
            if isinstance(aspects_raw, str):
                aspects_raw = [aspects_raw]

            doc_aspects = set()
            for a in aspects_raw:
                name = a.split(':')[0].strip().lower() if ':' in a else a.strip().lower()
                doc_aspects.add(name)

            new_score = score * boost_multiplier if doc_aspects & detected else score
            boosted.append((doc_id, new_score, metadata))

        boosted.sort(key=lambda x: x[1], reverse=True)
        logger.debug(
            f"Applied aspect boosting for {list(detected)} to {len(boosted)} results "
            f"(multiplier={boost_multiplier})"
        )
        return boosted

    def apply_time_decay(
        self,
        results: List[Tuple[str, float, Dict]],
        lambda_: float = 0.001,
        score_similarity_threshold: float = 0.005
    ) -> List[Tuple[str, float, Dict]]:
        """
        Apply exponential time-decay as a tiebreaker for near-identical scores.

        Only decays a document's score if an adjacent (in rank) document has a score
        within `score_similarity_threshold`. Formula: score * exp(-lambda_ * days_old).

        Args:
            results: Sorted (doc_id, score, metadata) tuples, highest score first.
            lambda_: Decay rate per day.
            score_similarity_threshold: Max score gap between neighbours to trigger decay.

        Returns:
            Re-sorted results with time-decay applied where relevant.
        """
        from math import exp
        from datetime import datetime, timezone

        if not results:
            return results

        scores = [score for _, score, _ in results]
        n = len(scores)

        # Mark which documents are in a near-tie with a neighbour
        near_tie = [False] * n
        for i in range(n):
            if i > 0 and abs(scores[i] - scores[i - 1]) < score_similarity_threshold:
                near_tie[i] = True
                near_tie[i - 1] = True
            if i < n - 1 and abs(scores[i] - scores[i + 1]) < score_similarity_threshold:
                near_tie[i] = True
                near_tie[i + 1] = True

        now = datetime.now(timezone.utc)
        decayed = []

        for idx, (doc_id, score, metadata) in enumerate(results):
            if near_tie[idx]:
                date_str = metadata.get('date', '')
                if isinstance(date_str, list):
                    date_str = date_str[0] if date_str else ''
                if date_str:
                    try:
                        doc_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        days_old = max((now - doc_date).days, 0)
                        score = score * exp(-lambda_ * days_old)
                    except Exception:
                        pass  # no valid date → no decay, score unchanged
            decayed.append((doc_id, score, metadata))

        decayed.sort(key=lambda x: x[1], reverse=True)
        logger.debug(
            f"Applied time-decay (λ={lambda_}, threshold={score_similarity_threshold}) "
            f"to {sum(near_tie)} near-tie documents"
        )
        return decayed

    def apply_mmr(
        self,
        results: List[Tuple[str, float, Dict]],
        query_vector: Optional[List[float]] = None,
        lambda_: float = 0.5,
        top_n: int = 50,
    ) -> List[Tuple[str, float, Dict]]:
        """
        Maximal Marginal Relevance reordering for result diversity.

        Iteratively selects results that maximise:
            λ * sim(doc, query) − (1−λ) * max(sim(doc, already_selected))

        Args:
            results:      Sorted (doc_id, score, metadata) list — highest relevance first.
            query_vector: L2-normalised query embedding (384 floats). If None, falls back
                          to normalised RRF scores as the relevance signal.
            lambda_:      Trade-off weight. 1.0 = pure relevance, 0.0 = pure diversity.
            top_n:        Only run MMR on the top-N candidates (rest appended unchanged).

        Returns:
            MMR-reordered results (diverse ordering within top_n, then remainder).
        """
        import numpy as np

        if not results or lambda_ >= 1.0:
            return results

        # Split into MMR candidate pool and the tail (appended unchanged)
        pool = results[:top_n]
        tail = results[top_n:]
        n = len(pool)

        # --- Build relevance scores ---
        rrf_scores = np.array([s for _, s, _ in pool], dtype=np.float32)
        max_score = rrf_scores.max()
        rel_scores = rrf_scores / max_score if max_score > 0 else rrf_scores

        if query_vector is not None:
            q = np.array(query_vector, dtype=np.float32)
            q_sims = np.zeros(n, dtype=np.float32)
            for i, (_, _, meta) in enumerate(pool):
                vec = meta.get('vector')
                if vec:
                    q_sims[i] = float(np.dot(q, np.array(vec, dtype=np.float32)))
            lo, hi = q_sims.min(), q_sims.max()
            rel_scores = (q_sims - lo) / (hi - lo) if hi > lo else q_sims

        # --- Build pairwise inter-document similarity matrix ---
        # Vectors are L2-normalised → dot product == cosine similarity
        doc_vecs = []
        for _, _, meta in pool:
            vec = meta.get('vector')
            if vec:
                doc_vecs.append(np.array(vec, dtype=np.float32))
            else:
                doc_vecs.append(np.zeros(384, dtype=np.float32))

        V = np.stack(doc_vecs)   # shape (n, 384)
        sim_matrix = V @ V.T     # shape (n, n) — cosine similarities

        # --- Greedy MMR selection ---
        selected = []
        remaining = list(range(n))

        while remaining:
            if not selected:
                best = max(remaining, key=lambda i: rel_scores[i])
            else:
                best, best_mmr = None, float('-inf')
                for i in remaining:
                    max_sim = max(float(sim_matrix[i, j]) for j in selected)
                    mmr = float(lambda_ * rel_scores[i] - (1 - lambda_) * max_sim)
                    if mmr > best_mmr:
                        best_mmr, best = mmr, i
            selected.append(best)
            remaining.remove(best)

        reordered = [pool[i] for i in selected]
        logger.debug(f"MMR applied (λ={lambda_}, pool={n}): reordered {n} candidates")
        return reordered + tail

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
