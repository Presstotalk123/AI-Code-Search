"""API routes for search engine"""
import logging
from flask import request, jsonify, current_app
from api.utils import format_response_error, validate_search_params

logger = logging.getLogger(__name__)


def register_routes(app):
    """Register all API routes"""

    @app.route('/')
    def index():
        """Serve frontend"""
        return app.send_static_file('index.html')

    @app.route('/api/search', methods=['GET'])
    def search():
        """
        Main search endpoint

        Query Parameters:
            q (str, required): Search query
            mode (str): 'keyword', 'semantic', or 'hybrid' (default: hybrid)
            page (int): Page number (default: 1)
            page_size (int): Results per page (default: 10, max: 100)

            Filters:
            date_from (str): ISO8601 date (e.g., "2025-01-01")
            date_to (str): ISO8601 date
            tools (str): Comma-separated list (e.g., "cursor,copilot")
            sentiment (str): 'positive', 'negative', 'mixed', 'not_applicable'
            source (str): 'reddit'

            Options:
            sentiment_boost (bool): Apply sentiment boosting (default: true)

        Returns:
            JSON response with search results
        """
        try:
            # Parse parameters
            query = request.args.get('q', '').strip()
            mode = request.args.get('mode', 'hybrid')
            page = int(request.args.get('page', 1))
            page_size = min(int(request.args.get('page_size', 10)), 100)

            # Validate parameters
            is_valid, error_msg = validate_search_params(query, mode, page, page_size)
            if not is_valid:
                return format_response_error(error_msg, 400)

            # Parse filters
            filters = {}
            if request.args.get('date_from'):
                filters['date_from'] = request.args.get('date_from')
            if request.args.get('date_to'):
                filters['date_to'] = request.args.get('date_to')
            if request.args.get('tools'):
                tools_str = request.args.get('tools', '')
                filters['tools'] = [t.strip() for t in tools_str.split(',') if t.strip()]
            if request.args.get('sentiment'):
                filters['sentiment'] = request.args.get('sentiment')
            if request.args.get('source'):
                filters['source'] = request.args.get('source')

            sentiment_boost = request.args.get('sentiment_boost', 'true').lower() == 'true'

            min_similarity = float(request.args.get('min_similarity', 0.0))
            if not (0.0 <= min_similarity <= 1.0):
                return format_response_error('min_similarity must be between 0.0 and 1.0', 400)

            # Execute search
            results = current_app.search_engine.search_hybrid(
                query=query,
                filters=filters if filters else None,
                mode=mode,
                apply_sentiment_boost=sentiment_boost,
                page=page,
                page_size=page_size,
                min_similarity=min_similarity
            )

            logger.info(f"Search: q='{query}', mode={mode}, results={results['total_count']}, time={results['query_time_ms']}ms")

            return jsonify(results)

        except ValueError as e:
            return format_response_error(str(e), 400)
        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            return format_response_error('Internal server error', 500)

    @app.route('/api/trend', methods=['GET'])
    def trend():
        """
        Timeline trend endpoint

        Query Parameters:
            q (str, required): Search keyword
            tools (str, optional): Comma-separated tool names
            aspect (str, optional): Aspect name substring filter
            date_from (str, optional): ISO8601 date
            date_to (str, optional): ISO8601 date
            granularity (str, optional): 'day', 'week', or 'month' (default: month)
            min_similarity (float, optional): Similarity threshold for hybrid mode (0.0–1.0)

        Returns:
            JSON with timeline buckets containing polarity counts and avg upvotes
        """
        try:
            from datetime import datetime, timedelta

            query = request.args.get('q', '').strip()
            effective_query = query if query else '*:*'
            requested_mode = request.args.get('search_mode', '').strip()
            search_mode = requested_mode if requested_mode in ('keyword', 'hybrid') else ('keyword' if not query else 'hybrid')

            granularity = request.args.get('granularity', 'month')
            if granularity not in ('day', 'week', 'month'):
                granularity = 'month'

            try:
                min_similarity = float(request.args.get('min_similarity', 0.0))
                min_similarity = max(0.0, min(1.0, min_similarity))
            except (ValueError, TypeError):
                min_similarity = 0.0

            filters = {}
            if request.args.get('date_from'):
                filters['date_from'] = request.args.get('date_from')
            if request.args.get('date_to'):
                filters['date_to'] = request.args.get('date_to')
            if request.args.get('tools'):
                tools_str = request.args.get('tools', '')
                filters['tools'] = [t.strip() for t in tools_str.split(',') if t.strip()]

            aspect_filter = request.args.get('aspect', '').strip().lower()

            # BM25 pre-check: if keyword provided but has no literal matches, skip hybrid search
            if query:
                bm25_check = current_app.search_engine.search_solr(query, rows=1)
                if not bm25_check:
                    return jsonify({
                        'timeline': [],
                        'granularity': granularity,
                        'total_results': 0,
                        'message': 'No posts matched this keyword. Try a different search term.'
                    })

            results_data = current_app.search_engine.search_hybrid(
                query=effective_query,
                filters=filters if filters else None,
                mode=search_mode,
                apply_sentiment_boost=False,
                page=1,
                page_size=None,
                min_similarity=min_similarity
            )

            docs = results_data.get('results', [])

            # Apply aspect filter if provided
            if aspect_filter:
                filtered_docs = []
                for doc in docs:
                    aspects = doc.get('aspects', [])
                    if any(aspect_filter in a.get('name', '').lower() for a in aspects):
                        filtered_docs.append(doc)
                docs = filtered_docs

            def get_bucket_key(date_str):
                try:
                    dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    if granularity == 'day':
                        return dt.strftime('%Y-%m-%d')
                    elif granularity == 'week':
                        # ISO week start (Monday)
                        week_start = dt - timedelta(days=dt.weekday())
                        return week_start.strftime('%Y-%m-%d')
                    else:  # month
                        return dt.strftime('%Y-%m')
                except Exception:
                    return None

            # Aggregate into buckets
            buckets = {}
            for doc in docs:
                key = get_bucket_key(doc.get('date', ''))
                if key is None:
                    continue
                if key not in buckets:
                    buckets[key] = {
                        'date': key,
                        'positive': 0,
                        'negative': 0,
                        'mixed': 0,
                        'not_applicable': 0,
                        'upvotes_sum': 0,
                        'count': 0
                    }
                sentiment = doc.get('sentiment', 'not_applicable')
                if sentiment in buckets[key]:
                    buckets[key][sentiment] += 1
                buckets[key]['upvotes_sum'] += doc.get('upvotes', 0)
                buckets[key]['count'] += 1

            # Build sorted timeline
            timeline = []
            for key in sorted(buckets.keys()):
                b = buckets[key]
                avg_upvotes = round(b['upvotes_sum'] / b['count'], 2) if b['count'] > 0 else 0
                timeline.append({
                    'date': b['date'],
                    'positive': b['positive'],
                    'negative': b['negative'],
                    'mixed': b['mixed'],
                    'not_applicable': b['not_applicable'],
                    'avg_upvotes': avg_upvotes,
                    'count': b['count']
                })

            return jsonify({
                'timeline': timeline,
                'granularity': granularity,
                'total_results': len(docs)
            })

        except Exception as e:
            logger.error(f"Trend error: {e}", exc_info=True)
            return format_response_error('Internal server error', 500)

    @app.route('/api/stats', methods=['GET'])
    def stats():
        """
        Get dataset statistics

        Returns:
            JSON with total documents and facet counts
        """
        try:
            stats_data = current_app.search_engine.get_stats()
            return jsonify(stats_data)
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return format_response_error('Failed to retrieve statistics', 500)

    @app.route('/api/health', methods=['GET'])
    def health():
        """
        Health check endpoint

        Returns:
            JSON with health status of Solr and ChromaDB
        """
        try:
            health_data = current_app.search_engine.health_check()

            if health_data['status'] == 'healthy':
                return jsonify(health_data)
            else:
                return jsonify(health_data), 503

        except Exception as e:
            logger.error(f"Health check error: {e}")
            return jsonify({
                'status': 'unhealthy',
                'error': str(e)
            }), 503

    @app.route('/api/document/<doc_id>', methods=['GET'])
    def get_document(doc_id):
        """
        Get full document by ID

        Args:
            doc_id: Document ID

        Returns:
            JSON with document details
        """
        try:
            # Query Solr for specific document
            results = current_app.search_engine.solr.search(f'doc_id:"{doc_id}"', rows=1)
            docs = list(results)

            if not docs:
                return format_response_error('Document not found', 404)

            return jsonify(docs[0])

        except Exception as e:
            logger.error(f"Document retrieval error: {e}")
            return format_response_error('Failed to retrieve document', 500)

    logger.info("Routes registered successfully")
