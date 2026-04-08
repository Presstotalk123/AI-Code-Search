// Word Cloud Manager - Generates a word cloud from search result text
class WordCloudManager {
    constructor() {
        this.canvas = document.getElementById('wordCloudCanvas');
        this.facetContainer = document.getElementById('wordCloudFacet');
        this.stopWords = new Set([
            'a','an','the','and','or','but','if','in','on','at','to','for','of','with',
            'by','from','up','about','into','through','during','after','before','between',
            'out','off','over','under','again','then','once','per','via','is','are','was',
            'were','be','been','being','have','has','had','do','does','did','will','would',
            'could','should','may','might','shall','can','need','i','you','he','she','it',
            'we','they','me','him','her','us','them','my','your','his','its','our','their',
            'this','that','these','those','what','which','who','when','where','why','how',
            'all','each','not','no','so','too','very','just','also','both','few','more',
            'most','other','some','such','only','same','than','there','here','ever','never',
            'always','still','even','though','yet','again','really','well','get','got',
            'go','going','make','made','take','took','give','gave','see','saw','know',
            'think','use','used','using','feel','want','need','try','let','put','say',
            'said','tell','told','find','found','work','look','like','mean','good','bad',
            'great','better','best','new','old','right','true','false','sure','different',
            'reddit','post','comment','thread','lol','yeah','yes','thanks','thank',
            'please','help','anyone','people','someone','something','nothing','things',
            'thing','stuff','lot','bit','way','time','day','year','one','two','three',
            'dont','doesnt','cant','wont','isnt','wasnt','havent','didnt','wouldnt',
            'been','has','its','much','many','any','every','own','since','while','where',
            'when','then','now','back','down','come','came','came','over','does','has'
        ]);
    }

    update(results) {
        if (!results || results.length === 0) {
            this.hide();
            return;
        }
        const wordList = this.buildWordList(results);
        if (wordList.length === 0) {
            this.hide();
            return;
        }
        this.render(wordList);
        this.show();
    }

    buildWordList(results) {
        const freq = {};
        results.forEach(result => {
            const text = [result.title || '', result.snippet || ''].join(' ');
            const boost = result.similarity_score > 0 ? (1 + result.similarity_score) : 1;
            this.tokenize(text).forEach(word => {
                if (this.stopWords.has(word) || word.length < 3 || /^\d+$/.test(word)) return;
                freq[word] = (freq[word] || 0) + boost;
            });
        });

        const sorted = Object.entries(freq)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 60);

        if (sorted.length === 0) return [];

        const max = sorted[0][1];
        const wordColors = new Map(sorted.map(([word, weight]) => {
            const ratio = weight / max;
            return [word, ratio > 0.6 ? '#667eea' : ratio > 0.3 ? '#764ba2' : '#a78bda'];
        }));
        this._wordColors = wordColors;

        return sorted.map(([word, weight]) => [word, Math.round(weight)]);
    }

    tokenize(text) {
        return text.toLowerCase()
            .replace(/[^a-z0-9\s]/g, ' ')
            .split(/\s+/)
            .filter(Boolean);
    }

    render(wordList) {
        this.canvas.width = 240;
        this.canvas.height = 200;
        const colors = this._wordColors;
        const max = wordList[0][1];
        const min = wordList[wordList.length - 1][1];
        const range = max - min || 1;

        WordCloud(this.canvas, {
            list: wordList,
            gridSize: 6,
            weightFactor: size => 10 + ((size - min) / range) * 26,
            fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            color: word => (colors && colors.get(word)) || '#667eea',
            backgroundColor: 'transparent',
            rotateRatio: 0.3,
            rotationSteps: 2,
            minSize: 8,
            drawOutOfBound: false,
            shrinkToFit: true,
        });
    }

    show() { this.facetContainer.style.display = 'block'; }
    hide() { this.facetContainer.style.display = 'none'; }
}

window.wordCloudManager = new WordCloudManager();
