// Filter Manager - Handle filters and facets
class FilterManager {
    constructor() {
        this.filters = {};
        this.initializeEventListeners();
        // Set initial visibility based on the default checked mode
        const defaultMode = document.querySelector('input[name="mode"]:checked');
        if (defaultMode) {
            this.updateSimilarityVisibility(defaultMode.value);
        }
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

        // Similarity threshold slider
        document.getElementById('similarityFilter').addEventListener('input', (e) => {
            document.getElementById('similarityValue').textContent = parseFloat(e.target.value).toFixed(2);
        });

        document.getElementById('similarityFilter').addEventListener('change', () => {
            if (window.searchManager && window.searchManager.currentQuery) {
                document.getElementById('searchButton').click();
            }
        });
    }

    updateSimilarityVisibility(mode) {
        const group = document.getElementById('similarityFilterGroup');
        group.style.display = (mode === 'semantic' || mode === 'hybrid') ? '' : 'none';
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

        const minSimilarity = parseFloat(document.getElementById('similarityFilter').value);
        if (minSimilarity > 0) filters.minSimilarity = minSimilarity;

        return filters;
    }

    clearFilters() {
        document.getElementById('toolFilter').value = '';
        document.getElementById('sentimentFilter').value = '';
        document.getElementById('dateFrom').value = '';
        document.getElementById('dateTo').value = '';
        document.getElementById('similarityFilter').value = '0';
        document.getElementById('similarityValue').textContent = '0.00';

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
            'copilot': 'Copilot',
            'cursor': 'Cursor',
            'claude_code': 'Claude Code',
            'devin': 'Devin',
            'codewhisperer': 'CodeWhisperer',
            'tabnine': 'Tabnine',
            'windsurf': 'Windsurf',
            'chatgpt': 'ChatGPT',
            'gemini': 'Gemini',
            'codeium': 'Codeium',
            'supermaven': 'Supermaven',
            'replit_ai': 'Replit AI',
            'amazon_q': 'Amazon Q',
            'jetbrains_ai': 'JetBrains AI',
            'v0': 'v0',
            'bolt': 'Bolt',
            'antigravity': 'Antigravity',
            'lovable': 'Lovable',
            'kimi_code': 'Kimi Code',
            'grok_code': 'Grok Code',
            'cline': 'Cline',
            'roo_code': 'Roo Code',
            'kilo_code': 'Kilo Code'
        };
        return nameMap[tool] || tool;
    }
}

// Initialize filter manager
window.filterManager = new FilterManager();
