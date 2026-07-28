document.addEventListener('DOMContentLoaded', () => {
    const inputText = document.getElementById('input-text');
    const analyzeBtn = document.getElementById('analyze-btn');
    const outputBox = document.getElementById('output-box');
    const btnText = analyzeBtn.querySelector('span');
    const spinner = document.getElementById('loading-spinner');

    // Make setExample globally available for the onclick handlers in HTML
    window.setExample = function(element) {
        inputText.value = element.innerText;
    };

    analyzeBtn.addEventListener('click', async () => {
        const text = inputText.value.trim();
        if (!text) {
            outputBox.innerHTML = '<span class="placeholder-text" style="color: var(--negative-color);">Please enter some text to analyze.</span>';
            outputBox.classList.add('empty');
            return;
        }

        // Set Loading State
        analyzeBtn.disabled = true;
        btnText.innerText = 'Analyzing...';
        spinner.classList.remove('hidden');
        outputBox.classList.remove('empty');
        outputBox.innerHTML = '<span style="color: var(--text-secondary);">Sending request to model...</span>';

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ texts: [text] }) // Send as an array to support the backend batching format
            });

            if (!response.ok) {
                throw new Error(`Server responded with status ${response.status}`);
            }

            const data = await response.json();
            const result = data.results[0]; // We only sent 1 item

            if (typeof result === 'string') {
                // Handle error messages from backend (e.g. all emojis)
                outputBox.innerHTML = `<div style="color: var(--neutral-color);">${result}</div>`;
            } else {
                // Success
                outputBox.innerHTML = `
                    <div class="result-header">
                        Original: ${text}<br>
                        Cleaned: ${result.cleaned_text}
                    </div>
                    <div class="result-sentiment sentiment-${result.label}">
                        Sentiment: ${result.label}
                    </div>
                    <div class="result-confidence">
                        Confidence: ${(result.score * 100).toFixed(2)}%
                    </div>
                `;
            }

        } catch (error) {
            outputBox.innerHTML = `<span style="color: var(--negative-color);">Error: ${error.message}. Is the server running?</span>`;
        } finally {
            // Reset state
            analyzeBtn.disabled = false;
            btnText.innerText = 'Analyze Sentiment';
            spinner.classList.add('hidden');
        }
    });
});
