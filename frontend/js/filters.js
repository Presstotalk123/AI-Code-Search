// Filter Manager - Handle filters and facets
class FilterManager {
    constructor() {
        this.filters = {};
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // Clear filters button
        document.getElementById('clearFilters').addEventListener('click', () => {
            this.clearFilters();
        });

        // Filter change handlers (re-search when filters change)
        document.getElementById('toolFilter').addEventListener('change', () => {
            if (window.searchManager && window.searchManager.currentQuery) {
                document.getElementById('searchButton').click();
            }
        });

        document.getElementById('sentimentFilter').addEventListener('change', () => {
            if (window.searchManager && window.searchManager.currentQuery) {
                document.getElementById('searchButton').click();
            }
        });

        document.getElementById('dateFrom').addEventListener('change', () => {
            if (window.searchManager && window.searchManager.currentQuery) {
                document.getElementById('searchButton').click();
            }
        });

        document.getElementById('dateTo').addEventListener('change', () => {
            if (window.searchManager && window.searchManager.currentQuery) {
                document.getElementById('searchButton').click();
            }
        });
    }

    getFilters() {
        const filters = {};

        const tool = document.getElementById('toolFilter').value;
        if (tool) filters.tool = tool;

        const sentiment = document.getElementById('sentimentFilter').value;
        if (sentiment) filters.sentiment = sentiment;

        const dateFrom = document.getElementById('dateFrom').value;
        if (dateFrom) filters.dateFrom = dateFrom;

        const dateTo = document.getElementById('dateTo').value;
        if (dateTo) filters.dateTo = dateTo;

        return filters;
    }

    clearFilters() {
        document.getElementById('toolFilter').value = '';
        document.getElementById('sentimentFilter').value = '';
        document.getElementById('dateFrom').value = '';
        document.getElementById('dateTo').value = '';

        // Re-search if there's an active query
        if (window.searchManager && window.searchManager.currentQuery) {
            document.getElementById('searchButton').click();
        }
    }

    updateFacets(facets) {
        // Update tool facets
        this.renderToolFacets(facets.tools);

        // Update subreddit facets
        this.renderSubredditFacets(facets.subreddits);
    }

    renderToolFacets(tools) {
        const container = document.getElementById('toolFacets');
        container.innerHTML = '';

        if (!tools || Object.keys(tools).length === 0) {
            container.innerHTML = '<p style="color: #999; font-size: 0.9em;">No data</p>';
            return;
        }

        // Sort by count descending
        const sortedTools = Object.entries(tools).sort((a, b) => b[1] - a[1]);

        sortedTools.forEach(([tool, count]) => {
            const facetItem = document.createElement('div');
            facetItem.className = 'facet-item';
            facetItem.innerHTML = `
                <span>${this.formatToolName(tool)}</span>
                <span class="facet-count">${count}</span>
            `;
            container.appendChild(facetItem);
        });
    }

    renderSubredditFacets(subreddits) {
        const container = document.getElementById('subredditFacets');
        container.innerHTML = '';

        if (!subreddits || Object.keys(subreddits).length === 0) {
            container.innerHTML = '<p style="color: #999; font-size: 0.9em;">No data</p>';
            return;
        }

        // Already sorted by the API (top 10)
        Object.entries(subreddits).forEach(([subreddit, count]) => {
            const facetItem = document.createElement('div');
            facetItem.className = 'facet-item';
            facetItem.innerHTML = `
                <span>r/${subreddit}</span>
                <span class="facet-count">${count}</span>
            `;
            container.appendChild(facetItem);
        });
    }

    formatToolName(tool) {
        // Format tool names for display
        const nameMap = {
            'cursor': 'Cursor',
            'copilot': 'Copilot',
            'claude_code': 'Claude Code',
            'windsurf': 'Windsurf',
            'general': 'General'
        };
        return nameMap[tool] || tool;
    }
}

// Initialize filter manager
window.filterManager = new FilterManager();
