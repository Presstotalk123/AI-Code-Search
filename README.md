# AI Coding Tools Opinion Search Engine 🤖

A hybrid search engine for exploring Reddit opinions about AI coding tools (Cursor, Copilot, Claude Code, Windsurf, etc.) using **Apache Solr** (BM25 keyword search) + **ChromaDB** (semantic vector search) with **Reciprocal Rank Fusion (RRF)**.

## Features ✨

- **🔍 Hybrid Search**: Combines keyword precision (Solr BM25) with semantic understanding (ChromaDB embeddings)
- **⚡ Fast Queries**: 150-250ms average latency with parallel query execution
- **📊 Sentiment Analysis**: Filter and boost by sentiment (positive, negative, mixed, neutral)
- **🏷️ Aspect-Based Filtering**: Search by aspects (productivity, cost, trust, integration, etc.)
- **📅 Timeline Search**: Date range filtering
- **📈 Visual Analytics**: Sentiment pie chart and faceted navigation
- **🎯 Multi-Tool Coverage**: Search across Cursor, Copilot, Claude Code, Windsurf discussions

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  JSONL Dataset  │────▶│  Indexing Layer  │────▶│  Dual Indexes   │
│   (10k+ posts)  │     │  - Data Loader   │     │  - Solr (BM25)  │
└─────────────────┘     │  - Embeddings    │     │  - Chroma (768d)│
                        └──────────────────┘     └────────┬────────┘
                                                           │
                                                           ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │  Flask REST API  │◀────│  Search Engine  │
                        │  /api/search     │     │  - RRF Fusion   │
                        │  /api/stats      │     │  - Filters      │
                        └────────┬─────────┘     └─────────────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │  Frontend UI     │
                        │  - Search Bar    │
                        │  - Filters       │
                        │  - Viz Dashboard │
                        └──────────────────┘
```

## Tech Stack

- **Backend**: Flask 3.0 (Python 3.9+)
- **Search Engines**: Apache Solr 9.4 + ChromaDB 0.4.22
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **Frontend**: Vanilla JavaScript + Chart.js
- **Infrastructure**: Docker Compose for Solr

## Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- 4GB+ RAM
- ~2GB disk space

### Installation

1. **Clone or navigate to the project directory**

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Solr**
   ```bash
   docker-compose up -d
   ```

   Verify Solr is running: http://localhost:8983/solr

4. **Index the dataset**
   ```bash
   python indexing/run_indexing.py
   ```

   This will:
   - Download the sentence-transformers model (~80MB, first run only)
   - Index documents to Solr (keyword search)
   - Generate embeddings and index to ChromaDB (semantic search)
   - Take ~2-3 minutes for 10,000 records

5. **Start the Flask API**
   ```bash
   python api/app.py
   ```

6. **Open the application**

   Navigate to http://localhost:5000

## Usage

### Web Interface

1. **Search**: Enter a query like "cursor vs copilot productivity"
2. **Choose Mode**:
   - **Keyword**: Pure BM25 (fast, exact matches)
   - **Semantic**: Pure vector search (understands meaning)
   - **Hybrid**: RRF fusion (recommended)
3. **Apply Filters**:
   - Tool (cursor, copilot, claude_code, windsurf)
   - Sentiment (positive, negative, mixed, neutral)
   - Date range (from/to)
4. **View Results**: Ranked by relevance with sentiment badges, aspect tags, and metadata

### API Usage

#### Search Endpoint

```bash
GET /api/search?q=cursor%20productivity&mode=hybrid&sentiment=positive&tools=cursor
```

**Parameters:**
- `q` (required): Search query
- `mode`: `keyword`, `semantic`, or `hybrid` (default: `hybrid`)
- `page`: Page number (default: 1)
- `page_size`: Results per page (default: 10, max: 100)

**Filters:**
- `date_from`: ISO8601 date (e.g., `2025-01-01`)
- `date_to`: ISO8601 date
- `tools`: Comma-separated (e.g., `cursor,copilot`)
- `sentiment`: `positive`, `negative`, `mixed`, `not_applicable`
- `source`: `reddit`

**Response:**
```json
{
  "results": [
    {
      "doc_id": "reddit_1km100x_...",
      "title": "What am I doing wrong??",
      "snippet": "I am an experienced full stack developer...",
      "url": "https://reddit.com/r/cursor/...",
      "sentiment": "negative",
      "tools": ["cursor"],
      "aspects": ["productivity", "code_quality"],
      "date": "2025-05-13T23:41:32Z",
      "subreddit": "cursor",
      "upvotes": 3,
      "score": 0.0312
    }
  ],
  "total_count": 42,
  "page": 1,
  "page_size": 10,
  "total_pages": 5,
  "facets": {
    "tools": {"cursor": 25, "copilot": 17},
    "sentiment": {"positive": 18, "negative": 15, "mixed": 9}
  },
  "query_time_ms": 156.23,
  "mode": "hybrid"
}
```

#### Other Endpoints

- `GET /api/stats` - Dataset statistics
- `GET /api/health` - Health check (Solr + ChromaDB)
- `GET /api/document/<doc_id>` - Get full document by ID

## Configuration

Edit `config/config.yaml` to customize:

### Search Parameters

```yaml
search:
  rrf_k: 60  # RRF constant (lower = more weight to top results)
  solr_rows: 100  # Top N from Solr for RRF
  chroma_n_results: 100  # Top N from Chroma for RRF

  sentiment_boost:
    enabled: true
    positive: 1.2  # 20% boost for positive sentiment
    negative: 0.8  # 20% penalty for negative
    mixed: 1.0
    not_applicable: 1.0
```

### Embedding Model

```yaml
embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  # Alternative: "sentence-transformers/all-mpnet-base-v2" (better quality, slower)
  device: "cpu"  # or "cuda" if GPU available
```

## How It Works

### Reciprocal Rank Fusion (RRF)

RRF merges results from Solr and ChromaDB without requiring score normalization:

```
rrf_score(doc) = 1/(k + solr_rank) + 1/(k + chroma_rank)
```

Where `k=60` (empirically validated in IR research).

**Example:**
- Doc A: Solr rank=1, Chroma rank=5 → RRF = 1/61 + 1/65 = 0.0318
- Doc B: Solr rank=3, Chroma rank=2 → RRF = 1/63 + 1/62 = 0.0320

Doc B ranks higher (balanced performance across both systems).

### Sentiment-Aware Ranking

After RRF, scores are boosted by sentiment:

```
final_score = rrf_score × sentiment_multiplier
```

Configurable in `config.yaml`.

### Faceted Search

Post-fusion filtering and faceting ensures:
- Consistent behavior across keyword/semantic/hybrid modes
- Accurate facet counts on actual result set
- Fast for 10k records (in-memory computation)

## Performance

### Query Latency (10,000 records)

- **Keyword**: 50-100ms
- **Semantic**: 80-150ms
- **Hybrid**: 120-200ms (parallel execution)

### Indexing Time

- **Solr**: ~10 seconds
- **Embeddings**: ~2 minutes (CPU)
- **ChromaDB**: ~30 seconds
- **Total**: ~2-3 minutes for 10,000 records

### Scalability

- **10k records**: 150-250ms hybrid search
- **100k records**: 800ms-1s (recommend migrating to Solr faceting)

## Project Structure

```
Query search/
├── config/
│   ├── config.yaml              # Central configuration
│   └── solr_schema.xml          # Solr field definitions
├── indexing/
│   ├── data_loader.py           # JSONL parsing
│   ├── embeddings.py            # Sentence-transformers wrapper
│   ├── solr_indexer.py          # Solr batch indexing
│   ├── chroma_indexer.py        # ChromaDB indexing
│   └── run_indexing.py          # Main indexing script
├── api/
│   ├── app.py                   # Flask application
│   ├── routes.py                # API endpoints
│   ├── search_engine.py         # Core search logic
│   ├── rrf_fusion.py            # RRF algorithm
│   └── utils.py                 # Helper functions
├── frontend/
│   ├── index.html               # Main UI
│   ├── css/styles.css
│   └── js/
│       ├── search.js            # Search manager
│       ├── filters.js           # Filter manager
│       └── visualization.js     # Chart.js charts
├── data/
│   └── eval_final.jsonl         # Dataset
├── chroma_db/                   # ChromaDB persistent storage
├── logs/                        # Application logs
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## Dataset

The dataset (`data/eval_final.jsonl`) contains Reddit posts and comments with:

- **Fields**: doc_id, title, text, URL, date, author, upvotes
- **Labels**: sentiment (positive/negative/mixed), subjectivity, aspects, tools mentioned
- **Sources**: r/cursor, r/copilot, r/ClaudeCode, r/windsurf, r/vibecoding, etc.
- **Size**: 10,000+ records

**Sample record:**
```json
{
  "doc_id": "reddit_1km100x_...",
  "content": {
    "thread_title": "What am I doing wrong??",
    "main_text": "I am an experienced full stack developer..."
  },
  "labels": {
    "polarity": "negative",
    "subjectivity": "opinionated",
    "aspects": ["productivity", "code_quality"],
    "agents": ["cursor"]
  },
  "timestamps": {
    "created_at": "2025-05-13T23:41:32Z"
  }
}
```

## Troubleshooting

### Solr Connection Failed

```bash
# Check if Solr is running
docker ps

# View Solr logs
docker-compose logs solr

# Restart Solr
docker-compose restart solr
```

### ChromaDB Issues

```bash
# Delete and re-index
rm -rf chroma_db/
python indexing/run_indexing.py
```

### Port Already in Use

Edit `config/config.yaml`:
```yaml
server:
  port: 5001  # Change from 5000
```

### Slow Embedding Generation

- Use GPU: Set `device: "cuda"` in config (requires CUDA-compatible GPU)
- Use faster model: Keep `all-MiniLM-L6-v2` (already the fastest)

## Advanced Usage

### Re-indexing

```bash
# Clear Solr
curl "http://localhost:8983/solr/ai_tools_opinions/update?stream.body=<delete><query>*:*</query></delete>&commit=true"

# Clear ChromaDB
rm -rf chroma_db/

# Re-index
python indexing/run_indexing.py
```

### Custom Queries

```python
from api.search_engine import SearchEngine
import yaml

config = yaml.safe_load(open('config/config.yaml'))
engine = SearchEngine(config)

# Keyword search only
results = engine.search_solr("cursor productivity", rows=10)

# Semantic search only
results = engine.search_chroma("which tool is faster", n_results=10)

# Hybrid with filters
results = engine.search_hybrid(
    query="copilot autocomplete",
    mode='hybrid',
    filters={'sentiment': 'negative', 'tools': ['copilot']},
    page=1,
    page_size=20
)
```

## Contributing

Contributions welcome! Areas for improvement:

- [ ] Advanced ABSA (aspect-based sentiment analysis)
- [ ] Query suggestions/autocomplete
- [ ] Export results (CSV/JSON)
- [ ] Multi-platform support (GitHub issues, HackerNews)
- [ ] Real-time indexing (WebSocket updates)
- [ ] A/B testing UI for RRF parameters

## License

MIT License - see LICENSE file for details

## Acknowledgments

- **RRF Algorithm**: Cormack, Clarke & Buettcher (2009)
- **Embeddings**: Sentence-Transformers by UKP Lab
- **Data**: Reddit API

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ using Apache Solr, ChromaDB, and Flask**
