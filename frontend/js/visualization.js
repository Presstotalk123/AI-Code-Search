// Visualization Manager - Handle Chart.js sentiment pie chart
class VisualizationManager {
    constructor() {
        this.sentimentChart = null;
        this.initializeSentimentChart();
    }

    initializeSentimentChart() {
        const ctx = document.getElementById('sentimentChart').getContext('2d');

        // Create initial empty chart
        this.sentimentChart = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: []
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 10,
                            font: {
                                size: 11
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.parsed || 0;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((value / total) * 100).toFixed(1);
                                return `${label}: ${value} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    }

    updateSentimentChart(sentimentData) {
        if (!sentimentData || Object.keys(sentimentData).length === 0) {
            // Show empty state
            this.sentimentChart.data.labels = ['No data'];
            this.sentimentChart.data.datasets[0].data = [1];
            this.sentimentChart.data.datasets[0].backgroundColor = ['#e0e0e0'];
            this.sentimentChart.update();
            return;
        }

        // Prepare data
        const labels = [];
        const data = [];
        const colors = [];

        // Color mapping for sentiments
        const colorMap = {
            'positive': '#4caf50',
            'negative': '#f44336',
            'mixed': '#ff9800',
            'not_applicable': '#9e9e9e'
        };

        // Label mapping for display
        const labelMap = {
            'positive': 'Positive',
            'negative': 'Negative',
            'mixed': 'Mixed',
            'not_applicable': 'Neutral'
        };

        // Sort by count descending
        const sortedSentiments = Object.entries(sentimentData).sort((a, b) => b[1] - a[1]);

        sortedSentiments.forEach(([sentiment, count]) => {
            labels.push(labelMap[sentiment] || sentiment);
            data.push(count);
            colors.push(colorMap[sentiment] || '#9e9e9e');
        });

        // Update chart
        this.sentimentChart.data.labels = labels;
        this.sentimentChart.data.datasets[0].data = data;
        this.sentimentChart.data.datasets[0].backgroundColor = colors;
        this.sentimentChart.update();
    }

    destroy() {
        if (this.sentimentChart) {
            this.sentimentChart.destroy();
        }
    }
}

// Initialize visualization manager
window.visualizationManager = new VisualizationManager();

// Export for use in search.js
window.updateSentimentChart = (sentimentData) => {
    window.visualizationManager.updateSentimentChart(sentimentData);
};

let latestWordCloudRequestId = 0;

function getWordCloudContainer() {
    return document.getElementById('wordcloud-container');
}

window.clearWordCloud = (message = '') => {
    const container = getWordCloudContainer();
    if (!container) return;

    if (message) {
        container.innerHTML = `<p class="wordcloud-message">${message}</p>`;
        return;
    }

    container.innerHTML = '';
};

window.loadWordCloud = async (query, options = {}) => {
    const container = getWordCloudContainer();
    if (!container) return;

    if (!query || !query.trim()) {
        window.clearWordCloud('Run a search to generate a word cloud.');
        return;
    }

    const requestId = ++latestWordCloudRequestId;

    container.innerHTML = `
        <div class="wordcloud-loading">
            <span class="wordcloud-spinner" aria-hidden="true"></span>
            <span>Generating word cloud...</span>
        </div>
    `;

    const params = new URLSearchParams({
        q: query,
        mode: options.mode || 'hybrid',
        page_size: '100'
    });

    const filters = options.filters || {};
    if (filters.dateFrom) params.append('date_from', filters.dateFrom);
    if (filters.dateTo) params.append('date_to', filters.dateTo);
    if (filters.tool) params.append('tools', filters.tool);
    if (filters.sentiment) params.append('sentiment', filters.sentiment);
    if (filters.source) params.append('source', filters.source);

    try {
        const response = await fetch(`/api/wordcloud?${params.toString()}`);
        const data = await response.json();

        if (requestId !== latestWordCloudRequestId) {
            return;
        }

        if (!response.ok || !data.image) {
            const message = data && data.error ? data.error : 'No word cloud data available.';
            window.clearWordCloud(message);
            return;
        }

        container.innerHTML = '<img src="" class="wordcloud-image" alt="Search result word cloud">';
        const image = container.querySelector('img');
        image.src = data.image;

    } catch (error) {
        if (requestId !== latestWordCloudRequestId) {
            return;
        }

        console.error('Word cloud error:', error);
        window.clearWordCloud('Failed to load word cloud.');
    }
};
