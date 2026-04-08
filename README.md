# AI Coding Tools Opinion Search Engine 🤖

A hybrid search engine for exploring Reddit opinions about AI coding tools (Cursor, Copilot, Claude Code, Windsurf, etc.) using **Apache Solr 10** for both BM25 keyword search and HNSW vector search, with **Reciprocal Rank Fusion (RRF)**.

## Features ✨

- **🔍 Hybrid Search**: Combines keyword precision (Solr BM25) with semantic understanding (Solr HNSW vector search)
- **⚡ Fast Queries**: 150-250ms average latency with parallel query execution
- **📊 Sentiment Analysis**: Filter and boost by sentiment (positive, negative, mixed, neutral)
- **🏷️ Aspect-Based Filtering**: Search by aspects (productivity, cost, trust, integration, etc.)
- **📅 Timeline Search**: Date range filtering
- **📈 Visual Analytics**: Sentiment pie chart and faceted navigation
- **🎯 Multi-Tool Coverage**: Search across Cursor, Copilot, Claude Code, Windsurf discussions
- **📈 Timeline Trend Dashboard**: Visualize sentiment trends over time with customizable granularity (daily/weekly/monthly)
- **🎚️ Similarity Threshold Control**: Filter semantic/hybrid results by cosine similarity (configurable 0.0-1.0)
- **🎯 Advanced Aspect Filtering**: Aspect-based filtering with polarity support (e.g., "productivity:positive")

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  JSONL Dataset  │────▶│  Indexing Layer  │────▶│  Solr 10 Collection │
│   (887 posts)   │     │  - Data Loader   │     │  - BM25 fields      │
└─────────────────┘     │  - Embeddings    │     │  - HNSW vector field│
                        └──────────────────┘     └──────────┬──────────┘
                                                             │
                                                             ▼
                        ┌──────────────────┐     ┌─────────────────────┐
                        │  Flask REST API  │◀────│  Search Engine      │
                        │  /api/search     │     │  - Solr BM25        │
                        │  /api/stats      │     │  - Solr KNN (HNSW)  │
                        └────────┬─────────┘     │  - RRF Fusion       │
                                 │               └─────────────────────┘
                                 ▼
                        ┌──────────────────┐
                        │  Frontend UI     │
                        │  - Search Bar    │
                        │  - Filters       │
                        │  - Viz Dashboard │
                        │  - Trend Charts  │
                        └──────────────────┘
```

## Tech Stack

- **Backend**: Flask 3.0 (Python 3.9+)
- **Search Engine**: Apache Solr 10 (BM25 + HNSW `DenseVectorField`)
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

3. **Start Solr 10**
   ```bash
   docker-compose up -d
   ```

   Verify Solr is running: http://localhost:8983/solr

4. **Set up Solr schema and index the dataset**
   ```bash
   python config/setup_solr.py
   ```

   This single script handles everything:
   - Copies `config/merged-managed-schema.xml` into the Solr container
   - Reloads the Solr core to apply the schema
   - Clears any existing documents
   - Indexes all documents with BM25 fields + 384-dim vector embeddings (~2-4 minutes on CPU, downloads ~80MB model on first run)
   - Optimizes the index to a single segment for best HNSW recall

   > **Note:** You only need to re-run `setup_solr.py` if the data or schema has changed. For normal restarts, skip straight to step 5.

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
   - **Semantic**: Pure HNSW vector search (understands meaning)
   - **Hybrid**: RRF fusion (recommended)
3. **Apply Filters**:
   - Tool (cursor, copilot, claude_code, windsurf)
   - Sentiment (positive, negative, mixed, neutral)
   - Date range (from/to)
   - Similarity threshold (for semantic/hybrid modes)
4. **View Results**: Ranked by relevance with sentiment badges, aspect tags, and metadata
5. **View Dashboard**: Click "Dashboard" to switch to trend visualization mode

### Dashboard & Trends

The application includes a dedicated dashboard view for analyzing sentiment trends over time.

#### Dashboard Overview

- **Two View Modes**: Toggle between Search view (main search interface) and Dashboard view (trend visualization) using header buttons
- **Interactive Visualizations**: Built with Chart.js for smooth, interactive timeline charts

#### Timeline Trend Visualization

The dashboard supports two chart display modes:

- **Polarity Mode**: Shows positive/negative/mixed sentiment counts over time as separate lines
  - Green line: Positive sentiment posts
  - Red line: Negative sentiment posts
  - Orange line: Mixed sentiment posts
  - Gray line: Not applicable sentiment posts

- **Popularity Mode**: Shows total post count over time
  - Blue line: Total number of posts per time bucket
  - Useful for identifying overall discussion volume trends

Charts support customizable time granularity:
- **Day**: Daily aggregation for fine-grained analysis
- **Week**: Weekly buckets for medium-term trends
- **Month**: Monthly aggregation for long-term patterns

#### Dashboard Controls

The dashboard provides comprehensive filtering and configuration:

- **Keyword Search**: Enter topics or keywords to analyze (e.g., "cursor productivity")
- **Search Mode**:
  - `keyword`: BM25-based keyword matching
  - `hybrid`: Combined semantic + keyword search (default when query provided)
- **Tool Filter**: Select specific tools (cursor, copilot, claude_code, windsurf)
- **Aspect Filter**: Filter by aspect name substring (e.g., "productivity", "trust")
- **Date Range**: Specify from/to dates to limit the analysis window
- **Similarity Threshold**: Slider control (0.0-1.0, default 0.60)
  - Only visible in hybrid mode
  - Filters results by cosine similarity score
  - Higher values = more semantically similar results only
- **Granularity**: Choose day/week/month for time bucket size
- **Chart Mode**: Toggle between Polarity and Popularity visualization modes
- **Generate Trend Button**: Executes the trend analysis and renders the chart

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
- `min_similarity` (float): Similarity threshold for semantic/hybrid modes (0.0-1.0, default: 0.0)
  - Filters results by cosine similarity score
  - Only applicable to semantic and hybrid modes
  - In hybrid mode, filters based on the vector score component

**Filters:**
- `date_from`: ISO8601 date (e.g., `2025-01-01`)
- `date_to`: ISO8601 date
- `tools`: Comma-separated (e.g., `cursor,copilot`)
- `sentiment`: `positive`, `negative`, `mixed`, `not_applicable`
- `aspect`: Aspect name substring filter (e.g., `productivity`)
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

#### Trend Endpoint

```bash
GET /api/trend?q=cursor&tools=cursor&granularity=month
```

**Parameters:**
- `q` (required): Search keyword or topic
- `granularity`: Time bucket size - `day`, `week`, or `month` (default: `month`)
- `search_mode`: `keyword` or `hybrid` (default: `keyword` for empty query, `hybrid` otherwise)
- `min_similarity`: Similarity threshold for hybrid mode (0.0-1.0, default: 0.0)

**Filters:**
- `tools`: Comma-separated tool names (e.g., `cursor,copilot`)
- `aspect`: Aspect name substring filter (e.g., `productivity`)
- `date_from`: ISO8601 date (e.g., `2025-01-01`)
- `date_to`: ISO8601 date

**Response:**
```json
{
  "timeline": [
    {
      "date": "2025-01",
      "positive": 12,
      "negative": 8,
      "mixed": 3,
      "not_applicable": 2,
      "avg_upvotes": 15.4,
      "count": 25
    }
  ],
  "granularity": "month",
  "total_results": 152
}
```

#### Other Endpoints

- `GET /api/stats` - Dataset statistics
- `GET /api/health` - Health check (Solr BM25 + Solr vector)
- `GET /api/document/<doc_id>` - Get full document by ID

## Configuration

Edit `config/config.yaml` to customize:

### Search Parameters

```yaml
search:
  rrf_k: 60  # RRF constant (lower = more weight to top results)
  solr_rows: 100       # Top N from Solr BM25 for RRF
  vector_n_results: 100  # Top N from Solr KNN for RRF

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

RRF merges results from Solr BM25 and Solr KNN without requiring score normalization:

```
rrf_score(doc) = 1/(k + solr_rank) + 1/(k + vector_rank)
```

Where `k=60` (empirically validated in IR research).

**Example:**
- Doc A: Solr rank=1, Vector rank=5 → RRF = 1/61 + 1/65 = 0.0318
- Doc B: Solr rank=3, Vector rank=2 → RRF = 1/63 + 1/62 = 0.0320

Doc B ranks higher (balanced performance across both systems).

### Solr 10 HNSW Vector Search

Each document is indexed with a 384-dimensional embedding in Solr's `DenseVectorField`:

```
vectorDimension=384, similarityFunction=cosine
```

At query time, the search query is embedded and sent as a KNN query:

```
{!knn f=vector topK=100}[0.1, 0.2, ...]
```

Solr returns the top-K nearest neighbours by cosine similarity using its built-in HNSW graph index.

### Sentiment-Aware Ranking

After RRF, scores are boosted by sentiment:

```
final_score = rrf_score × sentiment_multiplier
```

Configurable in `config.yaml`.

### Advanced Retrieval Features

#### Similarity Threshold Filtering

The search engine supports cosine similarity threshold filtering for semantic and hybrid searches:

- **Range**: 0.0 (any similarity) to 1.0 (exact semantic match)
- **Hybrid Mode Behavior**: Filters based on the vector score component after RRF fusion
- **Use Case**: Ensures semantic relevance in trend analysis by excluding low-similarity matches
- **Performance**: Applied after RRF fusion, so it doesn't affect initial retrieval speed

Example: `min_similarity=0.60` only returns results with >60% semantic similarity to the query.

#### Nonsense Query Detection

The search engine includes intelligent query validation to prevent wasted computation:

- **BM25 Pre-check**: For keyword queries, performs a quick BM25 search first
- **Early Return**: If no literal keyword matches found, skips expensive vector embedding computation
- **User Feedback**: Returns early with helpful message explaining no results were found
- **Efficiency**: Saves vector embedding generation time on nonsense or misspelled keywords

This mechanism prevents the system from attempting semantic search on queries like "asdfghjkl" or gibberish.

#### Aspect Polarity Parsing

Enhanced aspect filtering supports sentiment polarity qualifiers:

- **Format**: `aspect:polarity` (e.g., `productivity:positive`, `trust_reliability:negative`)
- **Supported Polarities**: `positive`, `negative`, `mixed`, `not_applicable`
- **Backward Compatible**: Still accepts plain aspect names without polarity (e.g., `productivity`)
- **Use Case**: Fine-grained sentiment analysis by specific aspects

Example: Filter for posts discussing cursor's productivity in a positive light: `aspect=productivity:positive&tools=cursor`

#### Threshold-Based Retrieval for Trends

The trend endpoint uses a special retrieval mode for comprehensive analysis:

- **Pagination-Free Mode**: `page_size=None` retrieves all matching results
- **Full KNN Pool**: Uses entire index as the KNN search pool for better recall
- **Time Bucket Aggregation**: Groups results by date buckets (day/week/month)
- **Sentiment Distribution**: Calculates positive/negative/mixed/not_applicable counts per bucket
- **Performance**: Optimized for trend analysis over large time ranges

This mode ensures accurate trend visualization by analyzing the complete result set rather than paginated subsets.

### Faceted Search

Post-fusion filtering and faceting ensures:
- Consistent behavior across keyword/semantic/hybrid modes
- Accurate facet counts on actual result set
- Fast for 887 records (in-memory computation)

## Performance

### Query Latency

- **Keyword**: 50-100ms
- **Semantic**: 80-150ms
- **Hybrid**: 120-200ms (parallel execution)

### Indexing Time

- **Total (BM25 + vectors)**: ~2-4 minutes on CPU for 887 records

### Scalability

- **1k records**: 150-250ms hybrid search
- **100k records**: 800ms-1s (recommend Solr native faceting)

## Project Structure

```
Query search/
├── config/
│   ├── config.yaml                  # Central configuration
│   ├── solr_schema.xml              # Solr field definitions (including DenseVectorField)
│   ├── merged-managed-schema.xml    # Full Solr managed schema (default types + custom fields)
│   └── setup_solr.py                # One-shot setup: schema + indexing + optimize
├── indexing/
│   ├── data_loader.py           # JSONL parsing
│   ├── embeddings.py            # Sentence-transformers wrapper
│   ├── solr_indexer.py          # Solr batch indexing (BM25 + vector)
│   └── run_indexing.py          # Main indexing script
├── api/
│   ├── app.py                   # Flask application
│   ├── routes.py                # API endpoints
│   ├── search_engine.py         # Core search logic (BM25 + KNN)
│   ├── rrf_fusion.py            # RRF algorithm
│   └── utils.py                 # Helper functions
├── frontend/
│   ├── index.html               # Main UI
│   ├── css/styles.css
│   └── js/
│       ├── search.js            # Search manager
│       ├── filters.js           # Filter manager
│       ├── visualization.js     # Chart.js charts
│       └── dashboard.js         # Trend dashboard & timeline charts
├── data/
│   └── eval_final_labelled.jsonl  # Dataset
├── logs/                        # Application logs
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## Dataset

The dataset (`data/eval_final_labelled.jsonl`) contains Reddit posts and comments with:

- **Fields**: doc_id, title, text, URL, date, author, upvotes
- **Labels**: sentiment (positive/negative/mixed), subjectivity, aspects, tools mentioned
- **Sources**: r/cursor, r/copilot, r/ClaudeCode, r/windsurf, r/vibecoding, etc.
- **Size**: 887 records

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

### Re-indexing from Scratch

```bash
# Stop Solr and wipe data volume
docker-compose down -v

# Start fresh Solr 10
docker-compose up -d

# Apply schema + re-index in one step
python config/setup_solr.py
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

### Custom Queries

```python
from api.search_engine import SearchEngine
import yaml

config = yaml.safe_load(open('config/config.yaml'))
engine = SearchEngine(config)

# Keyword search only
results = engine.search_solr("cursor productivity", rows=10)

# Semantic search only (Solr HNSW KNN)
results = engine.search_solr_vector("which tool is faster", n_results=10)

# Hybrid with filters
results = engine.search_hybrid(
    query="copilot autocomplete",
    mode='hybrid',
    filters={'sentiment': 'negative', 'tools': ['copilot']},
    page=1,
    page_size=20
)

# Generate trend data with similarity threshold
results = engine.search_hybrid(
    query="cursor productivity",
    mode='hybrid',
    filters={'tools': ['cursor'], 'date_from': '2025-01-01'},
    page_size=None,  # Return all results for trend aggregation
    min_similarity=0.60  # Only include results with >60% semantic similarity
)

# Trend aggregation (see api/routes.py lines 172-221 for full implementation)
# Results are grouped into time buckets with sentiment distribution counts
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

**Built with ❤️ using Apache Solr 10 and Flask**
