"""Flask application for AI Coding Tools Opinion Search Engine"""
import logging
import yaml
import sys
import os
from pathlib import Path
from flask import Flask
from flask_cors import CORS

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.search_engine import SearchEngine
from api.routes import register_routes


def setup_logging(config: dict):
    """Configure logging"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = log_config.get('file', './logs/app.log')

    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_app(config_path: str = 'config/config.yaml'):
    """
    Create and configure Flask application

    Args:
        config_path: Path to configuration file

    Returns:
        Configured Flask app
    """
    # Load configuration
    config = load_config(config_path)

    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info("Starting AI Coding Tools Opinion Search Engine")

    # Create Flask app
    app = Flask(__name__, static_folder='../frontend', static_url_path='')
    CORS(app)  # Enable CORS for all routes

    # Store config in app
    app.config['SEARCH_CONFIG'] = config

    # Initialize search engine
    logger.info("Initializing search engine...")
    try:
        app.search_engine = SearchEngine(config)
        logger.info("Search engine initialized successfully")

        # Perform health check
        health = app.search_engine.health_check()
        if health['status'] != 'healthy':
            logger.warning(f"Search engine health check failed: {health}")
        else:
            logger.info("Search engine health check passed")

    except Exception as e:
        logger.error(f"Failed to initialize search engine: {e}", exc_info=True)
        raise

    # Register routes
    register_routes(app)
    logger.info("Routes registered")

    return app


def main():
    """Main entry point"""
    try:
        app = create_app()
        config = app.config['SEARCH_CONFIG']

        host = config['server']['host']
        port = config['server']['port']
        debug = config['server']['debug']

        print("\n" + "=" * 60)
        print("  AI Coding Tools Opinion Search Engine")
        print("=" * 60)
        print(f"  Server: http://{host}:{port}")
        print(f"  API:    http://{host}:{port}/api/search?q=cursor")
        print(f"  Health: http://{host}:{port}/api/health")
        print(f"  Stats:  http://{host}:{port}/api/stats")
        print("=" * 60)
        print("\nPress Ctrl+C to stop\n")

        app.run(host=host, port=port, debug=debug)

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\nFailed to start server: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
