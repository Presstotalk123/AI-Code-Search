"""Unit tests for RRF fusion algorithm"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.rrf_fusion import RRFFusion


class TestRRFFusion:
    """Test RRF fusion algorithm"""

    def test_rrf_basic_scoring(self):
        """Test basic RRF score calculation"""
        rrf = RRFFusion(k=60)

        # Test formula: 1 / (k + rank)
        assert rrf.compute_rrf_score(1) == pytest.approx(1/61, rel=1e-5)
        assert rrf.compute_rrf_score(2) == pytest.approx(1/62, rel=1e-5)
        assert rrf.compute_rrf_score(100) == pytest.approx(1/160, rel=1e-5)

    def test_rrf_fusion_simple(self):
        """Test basic fusion of Solr and ChromaDB results"""
        rrf = RRFFusion(k=60)

        solr_results = [
            {'doc_id': 'doc1', 'score': 10.0, 'text': 'Solr result 1'},
            {'doc_id': 'doc2', 'score': 8.0, 'text': 'Solr result 2'},
            {'doc_id': 'doc3', 'score': 6.0, 'text': 'Solr result 3'}
        ]

        chroma_results = [
            {'doc_id': 'doc2', 'distance': 0.1, 'text': 'Chroma result 1'},
            {'doc_id': 'doc1', 'distance': 0.2, 'text': 'Chroma result 2'},
            {'doc_id': 'doc4', 'distance': 0.3, 'text': 'Chroma result 3'}
        ]

        fused = rrf.fuse_results(solr_results, chroma_results)

        # doc1: rank 1 in solr + rank 2 in chroma = 1/61 + 1/62
        # doc2: rank 2 in solr + rank 1 in chroma = 1/62 + 1/61
        # doc2 should rank equal or higher than doc1

        assert len(fused) == 4  # 4 unique documents
        assert fused[0][0] in ['doc1', 'doc2']  # Top should be doc1 or doc2
        assert fused[1][0] in ['doc1', 'doc2']

    def test_rrf_fusion_overlapping_docs(self):
        """Test fusion with overlapping documents"""
        rrf = RRFFusion(k=60)

        solr_results = [
            {'doc_id': 'doc1', 'score': 10.0}
        ]

        chroma_results = [
            {'doc_id': 'doc1', 'distance': 0.1}
        ]

        fused = rrf.fuse_results(solr_results, chroma_results)

        # doc1 appears in both, should get combined score
        assert len(fused) == 1
        assert fused[0][0] == 'doc1'
        # Combined score: 1/61 + 1/61
        expected_score = 1/61 + 1/61
        assert fused[0][1] == pytest.approx(expected_score, rel=1e-5)

    def test_sentiment_boosting(self):
        """Test sentiment-based score boosting"""
        rrf = RRFFusion()

        results = [
            ('doc1', 1.0, {'sentiment_label': 'positive'}),
            ('doc2', 0.9, {'sentiment_label': 'negative'}),
            ('doc3', 0.8, {'sentiment_label': 'mixed'}),
            ('doc4', 0.7, {'sentiment_label': 'not_applicable'})
        ]

        boost_config = {
            'positive': 1.5,
            'negative': 0.7,
            'mixed': 1.0,
            'not_applicable': 1.0
        }

        boosted = rrf.apply_sentiment_boosting(results, boost_config)

        # doc1: 1.0 * 1.5 = 1.5
        # doc2: 0.9 * 0.7 = 0.63
        # doc3: 0.8 * 1.0 = 0.8
        # doc4: 0.7 * 1.0 = 0.7

        assert boosted[0][0] == 'doc1'  # Highest after boost
        assert boosted[0][1] == pytest.approx(1.5, rel=1e-5)

        assert boosted[1][0] == 'doc3'  # Second highest
        assert boosted[1][1] == pytest.approx(0.8, rel=1e-5)

    def test_empty_results(self):
        """Test fusion with empty results"""
        rrf = RRFFusion()

        solr_results = []
        chroma_results = []

        fused = rrf.fuse_results(solr_results, chroma_results)

        assert len(fused) == 0

    def test_only_solr_results(self):
        """Test fusion with only Solr results"""
        rrf = RRFFusion()

        solr_results = [
            {'doc_id': 'doc1', 'score': 10.0},
            {'doc_id': 'doc2', 'score': 8.0}
        ]
        chroma_results = []

        fused = rrf.fuse_results(solr_results, chroma_results)

        assert len(fused) == 2
        assert fused[0][0] == 'doc1'
        assert fused[1][0] == 'doc2'

    def test_only_chroma_results(self):
        """Test fusion with only ChromaDB results"""
        rrf = RRFFusion()

        solr_results = []
        chroma_results = [
            {'doc_id': 'doc1', 'distance': 0.1},
            {'doc_id': 'doc2', 'distance': 0.2}
        ]

        fused = rrf.fuse_results(solr_results, chroma_results)

        assert len(fused) == 2
        assert fused[0][0] == 'doc1'
        assert fused[1][0] == 'doc2'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
