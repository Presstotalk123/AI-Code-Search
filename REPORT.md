# Report: AI Coding Tools Opinion Search Engine

---

## 1. Overview

The **AI Coding Tools Opinion Search Engine** is a full-stack information retrieval system purpose-built to collect, index, and surface community opinions about AI-powered coding assistants. It indexes 887 labeled Reddit posts and comments discussing tools such as Cursor, GitHub Copilot, Claude Code, Windsurf, ChatGPT, and 18 other AI coding assistants.

**Core Purpose:**
Users — researchers, developers, product managers — can type natural-language queries (e.g., *"is Cursor better than Copilot for productivity?"*) and instantly receive relevant, ranked Reddit discussions. They can further filter by tool, sentiment, date, and topic aspect, and explore how opinions have shifted over time via the Dashboard.

**Key Capabilities:**
- Three search modes: keyword BM25, semantic (vector), and hybrid (both combined via Reciprocal Rank Fusion)
- Sentiment-labeled results (positive / negative / mixed / not applicable)
- Aspect-level annotations: each post is tagged with specific dimensions such as `productivity`, `cost_value`, `code_quality`, `trust_reliability`, and more
- Real-time trend visualization: line charts showing how community sentiment evolves daily, weekly, or monthly
- Word cloud of dominant terms in any result set
- Faceted navigation by tool mentioned, subreddit, and sentiment

**Why It Matters:**
AI coding tools are proliferating rapidly, and developer communities generate enormous volumes of unstructured, nuanced opinion data. This system transforms that noisy social content into a structured, searchable, and analytically rich corpus — enabling evidence-based comparisons between tools and surfacing real-world usage patterns that product documentation alone cannot capture.

---

## 2. Architecture

### 2.1 Tech Stack

The system is organized into four distinct layers, each chosen to maximize simplicity, performance, and maintainability.

#### Backend — Python + Flask 3.0

**Flask** was selected as the web framework for its lightweight footprint and flexibility. Since the application only requires a small REST API surface (five endpoints), a heavier framework such as Django would introduce unnecessary complexity and overhead. Flask's application factory pattern (`create_app()`) cleanly supports the sequential initialization of the search engine and logging subsystem before the first request is ever served, ensuring the application is fully ready — or fails visibly — before accepting traffic.

**Flask-CORS** enables cross-origin requests from the frontend JavaScript running in the browser, which is the standard requirement for single-page application architectures where the UI is served from the same origin but the API may be contacted from a different port during development.

#### Search Engine — Apache Solr 10 (Docker)

Apache Solr 10 is the central data store and retrieval engine of the entire system. It was chosen over popular alternatives such as Elasticsearch, OpenSearch, and Meilisearch for two specific and important technical reasons:

1. **Native HNSW DenseVectorField**: Solr 10 introduced production-ready approximate nearest-neighbor (ANN) indexing via Hierarchical Navigable Small World (HNSW) graphs directly within its Lucene core. This means both BM25 keyword search and vector similarity search operate against the *same collection* with no need for a separate, external vector database. A secondary system would add operational complexity, introduce synchronization risks between the two indices, and increase deployment cost. Solr 10 eliminates all of these concerns.
2. **Mature BM25 + edismax**: Solr's `edismax` (extended DisMax) query parser supports per-field boosting (e.g., `title^2 text`), phrase proximity boosting, and query-time synonym expansion. It has decades of production deployment history and is robust to malformed queries.

Running Solr via Docker Compose (`image: solr:10`) with a dedicated 1 GB JVM heap (`SOLR_HEAP: "1g"`) isolates the search engine process from the Python application server, prevents memory contention, and enables fully reproducible environment setup with a single `docker-compose up` command.

#### Embeddings — Sentence-Transformers (`all-MiniLM-L6-v2`)

The `all-MiniLM-L6-v2` model from Hugging Face's sentence-transformers library produces 384-dimensional dense vector representations that encode the semantic meaning of text. It was selected for this project because:

- It is fast on CPU (approximately 5 ms per sentence), making it practical for query-time embedding without requiring a GPU
- At 384 dimensions it strikes a strong balance between retrieval quality and the storage and computational cost of the HNSW index
- Its output vectors are L2-normalized by default, which makes cosine similarity mathematically equivalent to a dot-product operation — a requirement for Solr's `dot_product` HNSW similarity function

Embeddings are pre-computed in batches of 32 during the indexing pipeline and stored as Solr `DenseVectorField` values. At query time, only the user's input query needs to be embedded, not the entire corpus. This makes semantic search fast and scalable regardless of index size.

#### Frontend — Vanilla JavaScript + Chart.js + WordCloud2.js

The frontend is deliberately dependency-light: no React, Vue, or Angular. This avoids the overhead of build tooling (Webpack, Babel, npm scripts) while remaining fully functional for the scope of the application — one HTML page with approximately 800 lines of modular JavaScript organized into five files. External visualization libraries are loaded via CDN:

- **Chart.js 4.4.0**: An industry-standard, canvas-based charting library. It is used for the sentiment distribution pie chart in the search sidebar and the temporal trend line chart in the Dashboard. Its declarative dataset model, built-in animation, and responsive layout make it well-suited for both real-time data updates and multi-series time series visualization.
- **WordCloud2.js 1.2.2**: A lightweight canvas-based word cloud renderer. It generates frequency-weighted visual summaries of the vocabulary present in each result set, providing users with an at-a-glance thematic overview that complements the ranked list of results.

---

### 2.2 How the Data is Used

#### Dataset

The dataset is stored at `data/eval_final_labelled.jsonl` — 887 records in JSON Lines format, where each line is a self-contained JSON object representing a single Reddit post or comment. The fields in each record are:

| Field | Description |
|---|---|
| `doc_id` | Unique document identifier (e.g., `reddit_1km100x_...`) |
| `source_platform` | Always `"reddit"` for this dataset |
| `source_url` | Full permalink to the original Reddit post or comment |
| `content_type` | `"post"` or `"comment"` |
| `content.thread_title` | The Reddit thread's title |
| `content.main_text` | The body of the post or comment |
| `content.reply_text` | Any reply text included in the record |
| `labels.polarity` | Overall sentiment: `positive`, `negative`, `mixed`, or `not_applicable` |
| `labels.aspects` | A dictionary mapping aspect names to their polarity (e.g., `{"productivity": "negative", "cost_value": "positive"}`) |
| `labels.agents` | The AI coding tools mentioned (e.g., `["cursor", "copilot"]`) |
| `labels.sarcasm` | Boolean sarcasm annotation |
| `engagement.upvotes` | Reddit upvote score at collection time |
| `timestamps.created_at` | Post creation time in ISO8601 format |
| `platform_context.subreddit` | The subreddit where the post appeared |

**Why JSONL?** JSON Lines format allows memory-efficient streaming — the data loader (`indexing/data_loader.py`) reads and yields records one line at a time using a Python generator, rather than loading the entire file into memory at once. At 887 records this is modest, but the architectural decision means the pipeline scales gracefully to millions of records with no code changes.

#### The Indexing Pipeline

Before the application can serve search requests, the raw JSONL dataset must be ingested, enriched with vector embeddings, and indexed into Solr. This is orchestrated by `config/setup_solr.py` as a one-shot five-step pipeline:

1. **Copy the Solr schema** into the Docker container via `docker cp`, placing the `merged-managed-schema.xml` into Solr's configuration directory
2. **Reload the Solr core** via the Solr Admin API so the new schema definition — including the `knn_vector` field type — takes effect
3. **Clear any existing documents** to ensure a clean, consistent index
4. **Run the indexing script** (`indexing/run_indexing.py`), which:
   - Streams each JSONL record through `data_loader.py`
   - Merges `thread_title + main_text + reply_text` into a single `combined_content` field, giving the BM25 index a comprehensive view of all textual content per record
   - Converts aspect dictionaries into `"aspect:polarity"` strings (e.g., `"productivity:negative"`) that can be stored as multi-valued Solr fields and queried as facets
   - Generates 384-dimensional L2-normalized embeddings for each record in batches of 32 using the `EmbeddingModel` wrapper
   - POSTs each enriched batch to Solr via `pysolr`, attaching the vector as the `vector` DenseVectorField
5. **Optimize the Solr index**: Calls Solr's `optimize` API to merge all Lucene segments into a single segment. This step is critical for HNSW vector search — when documents are spread across multiple Lucene segments, each segment maintains its own independent HNSW graph, reducing global recall. Merging into one segment creates a single unified HNSW graph that can find the true nearest neighbors across the entire corpus.

**How each data field enriches the search and analysis experience:**

- `tool_mentioned` (multi-valued): Powers the tool filter dropdown and tool-based facet counts in the sidebar
- `sentiment_label`: Drives the color-coded sentiment badges on result cards, populates the sentiment pie chart, and enables the sentiment filter
- `aspects` (multi-valued, `"aspect:polarity"` format): Enables aspect-level filtering in the Dashboard; each tag displayed on a result card with its polarity color comes from this field
- `date`: Supports time-range filtering on the search page and is the primary grouping key for the trend timeline in the Dashboard
- `upvotes`: Shown in result card metadata to help users gauge community reception of each post
- `vector` (indexed but not stored): Enables the HNSW approximate nearest-neighbor search at query time; not stored in the index to save disk space since it is never needed in retrieval output

---

### 2.3 Search Methods

The system implements three distinct search modes, each with different strengths and tradeoffs. The mode is selectable per query via a radio button in the UI.

#### BM25 Keyword Search

BM25 (Best Matching 25) is Solr's default probabilistic term-based ranking function, derived from the probabilistic relevance framework. It scores a document against a query by combining three signals:

- **Term Frequency (TF)**: How many times does the query term appear in the document? More occurrences = higher relevance, but with diminishing returns (the score saturates to prevent keyword-stuffed documents from dominating)
- **Inverse Document Frequency (IDF)**: How rare is the query term across the entire collection? Rare terms (e.g., `"Windsurf"`) are weighted more heavily than common terms (e.g., `"code"`)
- **Document Length Normalization**: Shorter documents are rewarded relative to longer ones containing the same terms, to avoid penalizing concise, information-dense posts

**Implementation details:**
- Query parser: `edismax` (extended DisMax), which supports field-level boosting and gracefully handles malformed queries
- Query fields: `title^2 text` — the `title` field is boosted by a factor of 2 because a Reddit post's title typically contains the most distilled, signal-dense language describing the post's topic
- The top 100 BM25-ranked documents are retrieved and passed to RRF fusion as input

**Strengths:** Extremely fast (50–100 ms), reliable for exact terminology such as tool names (`cursor`, `copilot`), specific feature names, and technical jargon. Requires no ML model at query time, making it independent of the embedding model's availability.

**Limitation:** BM25 is vocabulary-bound. A query for *"autocomplete feature"* will not match a post that only uses the phrase *"code suggestion capability"*, even though the two phrases are semantically equivalent. It also cannot handle paraphrasing or synonyms unless explicit synonym rules are configured in the Solr schema.

#### Semantic (Vector) Search

At query time, the user's input is encoded into a 384-dimensional vector using the same `all-MiniLM-L6-v2` model used during indexing. Solr's HNSW index then performs approximate nearest-neighbor (ANN) search to find the `topK=100` documents whose stored vectors are most similar to the query vector.

**HNSW Configuration:**
- `hnswMaxConnections=16`: The maximum number of bidirectional links per node in the HNSW proximity graph. Higher values increase recall quality at the cost of memory and graph construction time.
- `hnswBeamWidth=200`: The `efConstruction` parameter controlling search beam width during graph construction. A value of 200 produces a high-quality graph with strong recall, at the cost of a longer initial indexing time.
- Similarity function: `dot_product` on L2-normalized vectors, which is mathematically equivalent to cosine similarity for unit vectors.

**Score range:** Solr reports HNSW scores in the range [0.5, 1.0], derived from the transformation `(1 + dot_product) / 2`. A score of 1.0 indicates identical vectors; 0.5 indicates orthogonal (completely unrelated) vectors.

**Strengths:** Captures semantic meaning beyond exact vocabulary. Conceptually related posts rank highly even when they share no exact terms with the query. Handles paraphrasing, synonyms, and nuanced natural-language queries well. Particularly effective for exploratory, open-ended queries like *"frustrations with AI coding assistants."*

**Limitation:** Slower than BM25 (~80–150 ms due to the embedding step). Can occasionally surface topically adjacent but off-topic results when the semantic space is densely packed. Also dependent on the quality and domain coverage of the embedding model.

#### Hybrid Search with Reciprocal Rank Fusion (RRF)

Hybrid mode combines the strengths of both approaches. BM25 and vector queries are executed **in parallel** via Python's `ThreadPoolExecutor` with two worker threads, halving the latency overhead of running them sequentially. The two resulting ranked lists are then merged using **Reciprocal Rank Fusion (RRF)**.

**RRF Formula:**

```
RRF_score(document) = Σ [ 1 / (k + rank_i) ]
```

Where:
- `k = 60` is an empirically validated smoothing constant. Its role is to prevent the top-ranked documents from exerting disproportionate influence on the final score, ensuring documents ranked slightly lower can still compete effectively.
- `rank_i` is the document's 1-based position in each ranked list (BM25 rank and vector rank are summed together)

**Why RRF instead of score normalization?**

BM25 scores and cosine similarity scores have completely different scales, distributions, and semantics. Any attempt to normalize them — for example, scaling both to [0, 1] — requires knowing the theoretical maximum score for each, which is query-dependent and difficult to compute reliably. Normalization errors can introduce systematic bias toward one retrieval method. RRF sidesteps this entirely by using only *rank order*, which is scale-invariant by definition.

Additional RRF advantages:
- Documents appearing in *both* ranked lists naturally receive higher combined scores, rewarding results that are both lexically and semantically relevant
- Documents absent from one list simply contribute zero RRF score from that list — no special case handling is required
- The merge operation runs in `O(n log n)` time and adds negligible latency to the overall search

**Practical outcome:** Hybrid mode consistently outperforms either standalone method. It inherits keyword precision (exact tool names, specific error messages) from BM25 and semantic recall (paraphrased opinions, thematically related discussions) from vector search. Typical query latency: 120–200 ms.

**Similarity Threshold:** In semantic and hybrid modes, users can set a minimum vector similarity score between 0.6 and 1.0. After fusion, any result whose raw HNSW similarity score falls below this threshold is discarded. This allows users to tighten result precision — for example, setting a threshold of 0.80 returns only highly semantically similar results, filtering out loosely related matches.

---

### 2.4 The Backend Implementation

The backend is a Python package organized under `api/`, with clean architectural separation between the HTTP routing layer, the search orchestration layer, and the fusion algorithm layer.

#### Flask Application (`api/app.py`)

The application is initialized using Flask's factory pattern. The `create_app()` function:
1. Loads all configuration from `config/config.yaml` using PyYAML
2. Sets up a dual-output logging system: colored console output via `colorlog` for development visibility, and persistent file output to `logs/app.log` for operational auditing
3. Initializes the `SearchEngine` as a module-level singleton — this is important because `SentenceTransformer` model loading (~80 MB) is an expensive one-time operation that must not repeat per request
4. Runs a Solr health check on startup, explicitly verifying both BM25 connectivity and HNSW index readiness before accepting any requests
5. Registers all API route blueprints

The fail-fast startup health check is a meaningful design decision: if Solr is not running or the vector index is not populated, the application reports the error immediately at launch rather than serving silent failures to the first users who search.

#### API Routes (`api/routes.py`)

The application exposes six endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Serve the `index.html` single-page frontend |
| `/api/search` | GET | Main hybrid search with full filtering and pagination |
| `/api/trend` | GET | Timeline trend analysis grouped by time bucket |
| `/api/stats` | GET | Dataset-level statistics and facet counts |
| `/api/health` | GET | Live BM25 + vector index health check |
| `/api/document/<doc_id>` | GET | Retrieve a single document by its ID |

**`/api/search`** accepts the parameters: `q` (query string), `mode` (keyword / semantic / hybrid), `page`, `page_size` (capped at 100), `min_similarity`, `date_from`, `date_to`, `tools` (comma-separated), `sentiment`, and `source`. Input validation enforces type constraints and sensible bounds before the search engine is invoked.

**`/api/trend`** includes a notable optimization: a **BM25 pre-check**. Before computing any vector embedding or running KNN search — both of which are computationally expensive — the endpoint first executes a cheap BM25 query. If BM25 returns zero results for the given query and filters, the full trend computation is immediately aborted and an informative status message is returned to the frontend. This avoids wasting CPU cycles (and significant latency) embedding and searching for queries that have no matching documents. This is particularly valuable for protecting against nonsense queries or over-constrained filter combinations.

#### Search Engine (`api/search_engine.py`)

The `SearchEngine` class is the central orchestrator of all retrieval logic. It holds references to the `pysolr` client, the loaded `SentenceTransformer` model, and the `RRFFusion` instance.

**`search_hybrid()`** is the primary entry point, called by the `/api/search` route for every query. Its processing pipeline is:

1. **Embed the query once**: The user's query string is encoded to a 384-dimensional vector. This vector is computed exactly once and shared between both search branches, avoiding redundant model inference.
2. **Parallel retrieval**: `ThreadPoolExecutor` launches the BM25 query and the HNSW KNN query simultaneously on two worker threads. This parallelism reduces hybrid search latency to approximately the maximum of the two individual query times (rather than their sum).
3. **RRF fusion**: Both ranked result lists are passed to `RRFFusion.fuse_results()`, which returns a single list sorted by descending RRF score.
4. **Similarity threshold filtering**: Results whose HNSW vector score falls below the configured `min_similarity` are removed.
5. **Optional sentiment boosting**: When enabled in `config.yaml`, each result's RRF score is multiplied by a sentiment-specific factor (e.g., positive × 1.2, negative × 0.8) and the list is re-ranked. This is disabled by default to preserve neutral retrieval behavior.
6. **Post-fusion filtering**: The fused result list is filtered in memory by date range, tool, sentiment label, and source. Performing this step *after* fusion — rather than pushing filters into the Solr query — ensures that the facet counts computed in the next step accurately reflect the filtered result set. Pre-filtering in Solr would produce facet counts that only count documents matching both the query and the filter, obscuring the true distribution.
7. **Facet computation**: Tool mentions (aggregated across multi-valued `tool_mentioned` fields), sentiment label counts, and top-10 subreddits by frequency are computed from the filtered result set.
8. **Pagination**: The result list is sliced to the requested `page` and `page_size` window.
9. **Result enrichment**: Each document is transformed from its raw Solr representation into the API response format — a 200-character snippet is generated, `"aspect:polarity"` strings are parsed into structured objects, and keyword rank, vector rank, RRF score, and similarity score are all attached as first-class fields.

#### RRF Fusion (`api/rrf_fusion.py`)

The `RRFFusion` class is a standalone, unit-tested algorithm module with no dependencies on Solr or the embedding model. This separation is intentional: it allows the fusion algorithm to be tested in complete isolation using only Python lists, verified by `tests/test_rrf.py`.

The `fuse_results()` method accepts the BM25 result list and the vector result list as inputs. It iterates through both lists, accumulates `1/(60 + rank)` scores keyed by `doc_id`, and returns the merged list sorted by descending total RRF score. Each result in the output carries its original `keyword_rank`, `vector_rank`, and raw `vector_score` as metadata — values that are surfaced in the result card UI for full transparency into how each result was ranked.

**Sentiment Boosting** (`apply_sentiment_boosting()`): When called, this method multiplies each result's RRF score by a configurable sentiment multiplier (e.g., `positive: 1.2`, `negative: 0.8`, `mixed: 1.0`). The list is then re-sorted by the updated scores. This feature allows operators to tune the retrieval system to promote or demote certain sentiment classes — for example, emphasizing positive community reception in a product showcase context. It is disabled by default in `config.yaml` to preserve objective, sentiment-neutral retrieval.

---

### 2.5 Search

The search interface (`frontend/js/search.js`) is managed by the `SearchManager` class, which encapsulates all query execution, result rendering, and pagination state.

#### User Interaction Flow

1. The user types a query into the search input and presses Enter or clicks the Search button
2. `SearchManager.performSearch()` assembles a `URLSearchParams` object containing the query string, selected mode, current page number, page size, and all active filter values
3. A `fetch()` call is dispatched to `/api/search` with these parameters
4. On a successful response, `renderResults()` iterates through `data.results` and injects HTML result cards into the DOM
5. Facet counts in the sidebar are updated via `FilterManager.updateFacets()`
6. The sentiment pie chart is refreshed via `VisualizationManager.updateSentimentChart()`
7. The word cloud is regenerated from the new result set via `WordCloudManager.update()`

#### Result Cards

Each result card renders the following information:

- **Title** as a hyperlink pointing to the original Reddit post or comment URL, allowing users to immediately read the full source discussion
- **Sentiment badge** in color: green for positive, red for negative, orange for mixed, gray for not applicable
- **Content type** indicator (post or comment)
- **Snippet**: The first 200 characters of the document's combined content, HTML-escaped to prevent XSS
- **Metadata row**: Post date, subreddit name, upvote count, tools mentioned, RRF score, and semantic similarity score — giving users all the signals needed to evaluate result relevance
- **Aspect tags**: Each aspect annotation is rendered as a colored chip — green background for positive polarity, red for negative, yellow for neutral — providing a rapid visual summary of what dimensions of the tool the post discusses

#### Filters (`frontend/js/filters.js`)

The `FilterManager` class controls five filter inputs, all of which automatically re-execute the current search when their value changes:

- **Tool filter**: A dropdown of 23 AI coding tools with human-readable display names (e.g., `claude_code` is displayed as `"Claude Code"`)
- **Sentiment filter**: Restricts results to a single sentiment class
- **Date range**: ISO8601 date inputs for `date_from` and `date_to`
- **Similarity threshold slider**: Ranges from 0.60 to 1.00; hidden entirely when keyword-only mode is active since BM25 produces no vector scores
- **Clear Filters**: Resets all inputs to their default values and re-runs the search

The automatic re-search on filter change means users never need to manually click "Search" after adjusting a filter — the results update immediately, providing a responsive, interactive exploration experience.

#### Word Cloud (`frontend/js/wordcloud.js`)

The `WordCloudManager` generates a canvas-based word cloud from the current result set on every search. Its construction process:

1. Tokenizes the `title` and `snippet` fields of every result into lowercase words
2. Removes 150+ English stop words (pronouns, articles, prepositions, common verbs) and Reddit-specific noise words (`lol`, `yeah`, `comment`, `post`, `reddit`)
3. Discards tokens shorter than 3 characters and pure numeric strings
4. Counts the frequency of each remaining term and applies the result's `similarity_score` as a relevance boost (weight = frequency × (1 + similarity_score))
5. Takes the top 60 terms by weighted frequency
6. Maps terms to a three-tier purple color gradient based on their relative frequency (high-frequency terms in `#667eea`, mid-frequency in `#764ba2`, low-frequency in `#a78bda`)
7. Renders the word cloud on a 240×200 canvas via WordCloud2.js with 30% of words rotated for visual variety

The word cloud serves as a rapid visual vocabulary summary of the result set. It is particularly useful for confirming that a query has retrieved on-topic results, or for discovering unexpected terminology clusters that suggest related queries worth exploring.

---

### 2.6 Dashboard Showing the Trends

The Dashboard view provides temporal trend analysis of community opinion, accessible by clicking "Dashboard" in the header toggle. It is implemented in `frontend/js/dashboard.js` by the `DashboardManager` class, backed by the `/api/trend` endpoint in `api/routes.py`.

#### Controls

The Dashboard exposes a full set of parameters for constructing a targeted trend query:

- **Agent (Tool)**: Restricts the trend to posts that mention a specific AI coding tool
- **Aspect**: Filters to posts annotated with a particular topic dimension, such as `productivity`, `cost_value`, `code_quality`, `trust_reliability`, `integration_ux`, `learning_impact`, `job_security`, `security_privacy`, or `token_usage`
- **Keyword**: An optional free-text query to further focus the trend — for example, searching for *"hallucination"* across all tools or within a specific tool's mentions
- **Search Mode**: Keyword (BM25 only) or Hybrid (BM25 + vector, enabling semantic expansion of the keyword)
- **Min Similarity**: A threshold slider that appears only in hybrid mode, controlling how strictly results must match the query semantically
- **Date Range**: Restricts the timeline to a specific date window
- **Granularity**: Groups results into **daily**, **weekly**, or **monthly** time buckets

#### Backend Processing (`/api/trend`)

When the user clicks **"Generate Trend"**, a GET request is sent to `/api/trend`. The backend:

1. **BM25 pre-check**: Runs a fast keyword-only query first. If no documents match, the trend computation is immediately aborted and a meaningful status message is returned — avoiding unnecessary vector embedding computation
2. **Full retrieval**: Runs the selected search mode (keyword or hybrid) with all specified filters, retrieving all matching documents (no pagination applied — the full result set is needed for time-series aggregation)
3. **Time bucketing**: Each document's `date` field is parsed and assigned to a time bucket key. For daily granularity the key is `"YYYY-MM-DD"`, for weekly it is `"YYYY-WNN"` (ISO week number), and for monthly it is `"YYYY-MM"`
4. **Sentiment aggregation**: Within each bucket, counts of `positive`, `negative`, `mixed`, and `not_applicable` documents are accumulated
5. **Post count**: The total number of posts per bucket is counted separately
6. **Sorted output**: Buckets are sorted chronologically and returned as an ordered array

#### Chart Modes

The trend chart supports two visualization modes, toggled by the Polarity / Popularity buttons:

**Polarity Mode** renders three simultaneous line series on the same time axis:
- **Positive** (green, `#4caf50`)
- **Negative** (red, `#f44336`)
- **Mixed** (orange, `#ff9800`)

Each series plots the count of posts with that sentiment label per time bucket. This allows direct visual comparison of how positive, negative, and mixed community sentiment evolve over time. For example: a tool that shows a rising negative trend and a declining positive trend over several months is exhibiting signs of growing community dissatisfaction — a pattern that aggregate statistics would completely obscure.

**Popularity Mode** renders a single filled area chart plotting the **total number of posts per time bucket** (`b.count`). This reveals when a tool or topic spiked in community discussion volume. Volume spikes often correlate with significant external events — a major product release, a viral benchmark comparison, a high-profile failure, or a controversial pricing change. Popularity mode lets users locate these events on the timeline and then switch to polarity mode to understand whether the community reaction was positive or negative.

#### Why Temporal Analysis Is Essential

A single aggregate sentiment percentage is insufficient for understanding community opinion dynamics. Consider a scenario where a tool has an overall sentiment distribution of 55% positive, 35% negative, and 10% mixed. That looks reasonably favorable. But the trend dashboard might reveal that those positive posts are concentrated in a period six months ago, while the most recent three months show a sharp inversion — 70% negative and only 20% positive. The static aggregate hides a developing narrative of decline.

The trend dashboard makes the time dimension a first-class analytical axis. Combined with aspect filtering, it enables fine-grained questions such as: *"How has community sentiment about Cursor's productivity impact changed since January?"* or *"Did discussion volume around Claude Code increase after its March release?"* These are questions that no static search result or aggregate count can answer, but the trend dashboard answers directly.

The Chart.js line chart is configured with `tension: 0.3` for smooth, readable curves (avoiding the jagged appearance of linear interpolation on sparse data), an `index` tooltip interaction mode that shows all series values on hover for easy comparison, and a maximum of 12 x-axis ticks to prevent label overcrowding on high-granularity timelines.
