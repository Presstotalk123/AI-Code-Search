// Dashboard Manager - Timeline trend chart
class DashboardManager {
    constructor(apiBaseUrl = 'http://localhost:5000/api') {
        this.apiBaseUrl = apiBaseUrl;
        this.trendChart = null;
        this.currentMode = 'polarity'; // 'polarity' | 'popularity'
        this.lastData = null;

        document.getElementById('generateTrendBtn').addEventListener('click', () => this.generateTrend());
        document.getElementById('polarityBtn').addEventListener('click', () => this.setMode('polarity'));
        document.getElementById('popularityBtn').addEventListener('click', () => this.setMode('popularity'));
        document.getElementById('dashKeyword').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.generateTrend();
        });
    }

    async generateTrend() {
        const keyword = document.getElementById('dashKeyword').value.trim();
        const searchMode = document.querySelector('input[name="dashSearchMode"]:checked').value;

        const agent = document.getElementById('dashAgent').value;
        const aspect = document.getElementById('dashAspect').value.trim();
        const dateFrom = document.getElementById('dashDateFrom').value;
        const dateTo = document.getElementById('dashDateTo').value;
        const granularity = document.getElementById('dashGranularity').value;

        const params = new URLSearchParams({ q: keyword, granularity });
        if (keyword) params.append('search_mode', searchMode);
        if (agent) params.append('tools', agent);
        if (aspect) params.append('aspect', aspect);
        if (dateFrom) params.append('date_from', dateFrom);
        if (dateTo) params.append('date_to', dateTo);

        this.setStatus('Loading...');
        document.getElementById('generateTrendBtn').disabled = true;

        try {
            const response = await fetch(`${this.apiBaseUrl}/trend?${params}`);
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Request failed');
            }
            this.lastData = await response.json();

            if (!this.lastData.timeline || this.lastData.timeline.length === 0) {
                this.setStatus(this.lastData.message || 'No data found for the selected filters.');
                this.destroyChart();
                return;
            }

            const gran = this.lastData.granularity;
            const total = this.lastData.total_results;
            document.getElementById('trendTitle').textContent =
                `Trend Over Time (${gran.charAt(0).toUpperCase() + gran.slice(1)}) — ${total} result${total !== 1 ? 's' : ''}`;

            this.setStatus('');
            this.renderChart();
        } catch (error) {
            this.setStatus('Error: ' + error.message);
        } finally {
            document.getElementById('generateTrendBtn').disabled = false;
        }
    }

    renderChart() {
        if (!this.lastData) return;

        const timeline = this.lastData.timeline;
        const labels = timeline.map(b => b.date);

        let datasets;
        let yLabel;

        if (this.currentMode === 'polarity') {
            yLabel = 'Post Count';
            datasets = [
                {
                    label: 'Positive',
                    data: timeline.map(b => b.positive),
                    borderColor: '#4caf50',
                    backgroundColor: 'rgba(76,175,80,0.1)',
                    tension: 0.3,
                    fill: false
                },
                {
                    label: 'Negative',
                    data: timeline.map(b => b.negative),
                    borderColor: '#f44336',
                    backgroundColor: 'rgba(244,67,54,0.1)',
                    tension: 0.3,
                    fill: false
                },
                {
                    label: 'Mixed',
                    data: timeline.map(b => b.mixed),
                    borderColor: '#ff9800',
                    backgroundColor: 'rgba(255,152,0,0.1)',
                    tension: 0.3,
                    fill: false
                }
            ];
        } else {
            yLabel = 'Avg Upvotes';
            datasets = [
                {
                    label: 'Avg Upvotes',
                    data: timeline.map(b => b.avg_upvotes),
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102,126,234,0.1)',
                    tension: 0.3,
                    fill: true
                }
            ];
        }

        this.initChart(labels, datasets, yLabel);
    }

    initChart(labels, datasets, yLabel) {
        this.destroyChart();

        const ctx = document.getElementById('trendChart').getContext('2d');
        this.trendChart = new Chart(ctx, {
            type: 'line',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { position: 'top' },
                    tooltip: { mode: 'index', intersect: false }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Date' },
                        ticks: { maxTicksLimit: 12 }
                    },
                    y: {
                        title: { display: true, text: yLabel },
                        beginAtZero: true
                    }
                }
            }
        });
    }

    destroyChart() {
        if (this.trendChart) {
            this.trendChart.destroy();
            this.trendChart = null;
        }
    }

    setMode(mode) {
        this.currentMode = mode;
        document.getElementById('polarityBtn').classList.toggle('active', mode === 'polarity');
        document.getElementById('popularityBtn').classList.toggle('active', mode === 'popularity');
        if (this.lastData) this.renderChart();
    }

    setStatus(msg) {
        document.getElementById('trendStatus').textContent = msg;
    }
}

// View toggle logic
document.getElementById('dashboardViewBtn').addEventListener('click', () => {
    document.querySelector('.search-section').style.display = 'none';
    document.querySelector('.results-section').style.display = 'none';
    document.getElementById('dashboardSection').style.display = 'block';
    document.getElementById('dashboardViewBtn').classList.add('active');
    document.getElementById('searchViewBtn').classList.remove('active');
});

document.getElementById('searchViewBtn').addEventListener('click', () => {
    document.querySelector('.search-section').style.display = '';
    document.querySelector('.results-section').style.display = '';
    document.getElementById('dashboardSection').style.display = 'none';
    document.getElementById('searchViewBtn').classList.add('active');
    document.getElementById('dashboardViewBtn').classList.remove('active');
});

// Initialize dashboard manager
window.dashboardManager = new DashboardManager();
