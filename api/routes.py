"""API routes for search engine"""
import base64
import logging
from io import BytesIO
from flask import request, jsonify, current_app
from wordcloud import WordCloud
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

            # Execute search
            results = current_app.search_engine.search_hybrid(
                query=query,
                filters=filters if filters else None,
                mode=mode,
                apply_sentiment_boost=sentiment_boost,
                page=page,
                page_size=page_size
            )

            logger.info(f"Search: q='{query}', mode={mode}, results={results['total_count']}, time={results['query_time_ms']}ms")

            return jsonify(results)

        except ValueError as e:
            return format_response_error(str(e), 400)
        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
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

    @app.route('/api/wordcloud', methods=['GET'])
    def wordcloud():
        """
        Generate a word cloud image from current search result set

        Query Parameters:
            q (str, required): Search query
            mode (str): 'keyword', 'semantic', or 'hybrid' (default: hybrid)
            page_size (int): Number of docs to consider for text generation (default: 100, max: 100)

            Filters:
            date_from (str): ISO8601 date (e.g., "2025-01-01")
            date_to (str): ISO8601 date
            tools (str): Comma-separated list (e.g., "cursor,copilot")
            sentiment (str): 'positive', 'negative', 'mixed', 'not_applicable'
            source (str): 'reddit'

            Options:
            sentiment_boost (bool): Apply sentiment boosting (default: true)

        Returns:
            JSON with base64-encoded PNG image data URI
        """
        try:
            query = request.args.get('q', '').strip()
            mode = request.args.get('mode', 'hybrid')
            page_size = min(int(request.args.get('page_size', 100)), 100)

            is_valid, error_msg = validate_search_params(query, mode, 1, page_size)
            if not is_valid:
                return format_response_error(error_msg, 400)

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

            results = current_app.search_engine.search_hybrid(
                query=query,
                filters=filters if filters else None,
                mode=mode,
                apply_sentiment_boost=sentiment_boost,
                page=1,
                page_size=page_size
            )

            text_parts = []
            for result in results.get('results', []):
                title = result.get('title', '')
                snippet = result.get('snippet', '')
                aspects = result.get('aspects', [])

                if isinstance(aspects, list):
                    aspects_text = ' '.join(a for a in aspects if isinstance(a, str))
                elif isinstance(aspects, str):
                    aspects_text = aspects
                else:
                    aspects_text = ''

                # Snippets are truncated during enrichment, so frequencies are approximate.
                text_parts.append(f"{title} {snippet} {aspects_text}".strip())

            text_blob = ' '.join(part for part in text_parts if part).strip()
            if not text_blob:
                return format_response_error('No text to generate word cloud', 400)

            wc = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=50
            ).generate(text_blob)

            img_buffer = BytesIO()
            wc.to_image().save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

            logger.info(
                "Word cloud generated: q='%s', mode=%s, docs=%s",
                query,
                mode,
                len(results.get('results', []))
            )

            return jsonify({'image': f'data:image/png;base64,{img_base64}'})

        except ValueError as e:
            return format_response_error(str(e), 400)
        except Exception as e:
            logger.error(f"Word cloud error: {e}", exc_info=True)
            return format_response_error('Failed to generate word cloud', 500)

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
