// Search Manager - Main search logic
class SearchManager {
    constructor(apiBaseUrl = 'http://localhost:5000/api') {
        this.apiBaseUrl = apiBaseUrl;
        this.currentPage = 1;
        this.pageSize = 10;
        this.currentQuery = '';
        this.currentMode = 'hybrid';
        this.currentFilters = {};
    }

    async performSearch(query, mode, filters, page = 1) {
        try {
            // Update current state
            this.currentQuery = query;
            this.currentMode = mode;
            this.currentFilters = filters;
            this.currentPage = page;

            // Build query parameters
            const params = new URLSearchParams({
                q: query,
                mode: mode,
                page: page,
                page_size: this.pageSize
            });

            // Add filters
            if (filters.dateFrom) params.append('date_from', filters.dateFrom);
            if (filters.dateTo) params.append('date_to', filters.dateTo);
            if (filters.tool) params.append('tools', filters.tool);
            if (filters.sentiment) params.append('sentiment', filters.sentiment);
            if (filters.minSimilarity) params.append('min_similarity', filters.minSimilarity);

            // Fetch results
            const response = await fetch(`${this.apiBaseUrl}/search?${params}`);

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Search failed');
            }

            return await response.json();

        } catch (error) {
            console.error('Search error:', error);
            throw error;
        }
    }

    renderResults(data) {
        // Show spelling suggestion if present
        const suggestionEl = document.getElementById('spellingSuggestion');
        if (data.spelling_suggestion) {
            suggestionEl.innerHTML =
                `Did you mean: <a href="#" id="spellingLink">${data.spelling_suggestion}</a>?`;
            suggestionEl.style.display = 'block';
            document.getElementById('spellingLink').addEventListener('click', (e) => {
                e.preventDefault();
                document.getElementById('searchInput').value = data.spelling_suggestion;
                document.getElementById('searchButton').click();
            });
        } else {
            suggestionEl.style.display = 'none';
        }

        const container = document.getElementById('resultsContainer');
        container.innerHTML = '';

        // Show pagination
        document.getElementById('paginationContainer').style.display = 'flex';

        if (data.total_count === 0) {
            container.innerHTML = '<p class="no-results">No results found. Try different keywords or filters.</p>';
            if (window.wordCloudManager) window.wordCloudManager.update([]);
            return;
        }

        // Render result cards
        data.results.forEach(result => {
            const card = this.createResultCard(result);
            container.appendChild(card);
        });

        // Update metadata
        document.getElementById('resultsCount').textContent =
            `${data.total_count} result${data.total_count !== 1 ? 's' : ''} found`;
        document.getElementById('queryTime').textContent =
            `⚡ ${data.query_time_ms}ms (${data.mode})`;

        // Update pagination
        this.updatePagination(data);

        // Update facets
        if (window.filterManager) {
            window.filterManager.updateFacets(data.facets);
        }

        // Update sentiment chart
        if (window.visualizationManager) {
            window.visualizationManager.updateSentimentChart(data.facets.sentiment);
        }

        // Update word cloud
        if (window.wordCloudManager) {
            window.wordCloudManager.update(data.results);
        }
    }

    createResultCard(result) {
        const card = document.createElement('div');
        card.className = 'result-card';

        // Sentiment badge class
        const sentimentClass = `sentiment-${result.sentiment}`;

        // Format date
        let dateStr = '';
        if (result.date) {
            const date = new Date(result.date);
            dateStr = date.toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'short',
                day: 'numeric'
            });
        }

        // Title or fallback
        const title = result.title ||
                     (result.content_type === 'comment' ? 'Reddit Comment' : 'Reddit Post');

        // Tools display
        const toolsStr = Array.isArray(result.tools) ?
                        result.tools.join(', ') :
                        result.tools;

        // Aspects display with polarity
        const aspectsHtml = result.aspects && result.aspects.length > 0 ?
            `<div class="aspects">
                ${result.aspects.map(aspect => {
                    if (aspect.polarity) {
                        // New format with polarity
                        const polarityClass = `aspect-${aspect.polarity}`;
                        return `<span class="aspect-tag ${polarityClass}" title="${aspect.name}: ${aspect.polarity}">
                            ${aspect.name} <span class="polarity-badge">${aspect.polarity}</span>
                        </span>`;
                    } else {
                        // Old format without polarity (backward compatibility)
                        return `<span class="aspect-tag">${aspect.name}</span>`;
                    }
                }).join('')}
            </div>` : '';

        card.innerHTML = `
            <div class="result-header">
                <h3><a href="${result.url}" target="_blank">${this.escapeHtml(title)}</a></h3>
                <span class="badge ${sentimentClass}">${result.sentiment}</span>
            </div>
            <p class="snippet">${this.escapeHtml(result.snippet)}</p>
            <div class="result-meta">
                <span class="meta-item">📅 ${dateStr}</span>
                <span class="meta-item">📍 r/${result.subreddit}</span>
                <span class="meta-item">⬆️ ${result.upvotes} upvotes</span>
                <span class="meta-item">🤖 ${toolsStr}</span>
                <span class="meta-item">📊 Score: ${result.score}</span>
                ${result.similarity_score != null ? `<span class="meta-item">🎯 Similarity: ${result.similarity_score > 0 ? result.similarity_score : 'N/A'}</span>` : ''}
            </div>
            ${aspectsHtml}
        `;

        return card;
    }

    updatePagination(data) {
        const prevButton = document.getElementById('prevPage');
        const nextButton = document.getElementById('nextPage');
        const pageInfo = document.getElementById('pageInfo');

        // Update page info
        pageInfo.textContent = `Page ${data.page} of ${data.total_pages}`;

        // Update button states
        prevButton.disabled = data.page <= 1;
        nextButton.disabled = data.page >= data.total_pages;

        // Add click handlers
        prevButton.onclick = () => this.goToPage(data.page - 1);
        nextButton.onclick = () => this.goToPage(data.page + 1);
    }

    async goToPage(page) {
        if (page < 1) return;

        try {
            const data = await this.performSearch(
                this.currentQuery,
                this.currentMode,
                this.currentFilters,
                page
            );
            this.renderResults(data);

            // Scroll to top of results
            document.getElementById('resultsContainer').scrollIntoView({
                behavior: 'smooth'
            });

        } catch (error) {
            alert('Failed to load page: ' + error.message);
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    showError(message) {
        const container = document.getElementById('resultsContainer');
        container.innerHTML = `
            <div class="no-results">
                <p style="color: #f44336;">❌ ${this.escapeHtml(message)}</p>
            </div>
        `;
    }
}

// Initialize search manager
const searchManager = new SearchManager();

// Search button click handler
document.getElementById('searchButton').addEventListener('click', async () => {
    const query = document.getElementById('searchInput').value.trim();
    if (!query) {
        alert('Please enter a search query');
        return;
    }

    const mode = document.querySelector('input[name="mode"]:checked').value;
    const filters = window.filterManager ? window.filterManager.getFilters() : {};

    try {
        const results = await searchManager.performSearch(query, mode, filters);
        searchManager.renderResults(results);
    } catch (error) {
        searchManager.showError(error.message);
    }
});

// Enter key support
document.getElementById('searchInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        document.getElementById('searchButton').click();
    }
});

// Mode change handler (re-search with new mode, update similarity slider visibility)
document.querySelectorAll('input[name="mode"]').forEach(radio => {
    radio.addEventListener('change', () => {
        if (window.filterManager) {
            window.filterManager.updateSimilarityVisibility(radio.value);
        }
        // If there's an active search, re-execute with new mode
        if (searchManager.currentQuery) {
            document.getElementById('searchButton').click();
        }
    });
});

