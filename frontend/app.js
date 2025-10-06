class PolicyAssistant {
    constructor() {
        this.apiUrl = 'http://localhost:8000';
        this.initializeElements();
        this.bindEvents();
    }

    initializeElements() {
        this.questionInput = document.getElementById('questionInput');
        this.askBtn = document.getElementById('askBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.copyBtn = document.getElementById('copyBtn');
        this.loading = document.getElementById('loading');
        this.result = document.getElementById('result');
        this.error = document.getElementById('error');
        this.answer = document.getElementById('answer');
        this.citations = document.getElementById('citations');
        this.followUp = document.getElementById('followUp');
        this.confidence = document.getElementById('confidence');
        this.errorMessage = document.getElementById('errorMessage');
        this.contactHR = document.getElementById('contactHR');
    }
    bindEvents() {
        this.askBtn.addEventListener('click', () => this.askQuestion());
        this.clearBtn.addEventListener('click', () => this.clearAll());
        this.copyBtn.addEventListener('click', () => this.copyAnswer());
        this.questionInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.askQuestion();
            }
        });
    }

    async askQuestion() {
        const question = this.questionInput.value.trim();
        if (!question) {
            this.showError('Please enter a question');
            return;
        }

        this.showLoading();
        this.hideError();
        this.hideResult();

        try {
            const response = await fetch(`${this.apiUrl}/ask`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: question,
                    top_k: 5
                })
            });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.displayResult(data);
        } catch (error) {
            console.error('Error:', error);
            this.showError('Failed to get answer. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    displayResult(data) {
        // Display answer
        this.answer.innerHTML = this.formatAnswer(data.answer);
        
        // Display confidence
        this.confidence.textContent = `Confidence: ${data.confidence}`;
        this.confidence.className = `confidence ${data.confidence}`;

        // Display citations
        if (data.citations && data.citations.length > 0) {
            this.citations.innerHTML = '<h4>📚 Sources</h4>';
            data.citations.forEach(citation => {
                const citationDiv = document.createElement('div');
                citationDiv.className = 'citation';
                citationDiv.innerHTML = `
                    <div class="citation-header">${citation.doc_id} - ${citation.section}</div>
                    <div class="citation-snippet">"${citation.snippet}"</div>
                `;
                this.citations.appendChild(citationDiv);
            });
        } else {
                   this.citations.innerHTML = '';
        }

        // Display follow-up suggestions
        if (data.follow_up_suggestions && data.follow_up_suggestions.length > 0) {
            this.followUp.innerHTML = '<h4>💡 You might also ask:</h4>';
            data.follow_up_suggestions.forEach(suggestion => {
                const suggestionDiv = document.createElement('div');
                suggestionDiv.className = 'follow-up-suggestion';
                suggestionDiv.textContent = suggestion;
                suggestionDiv.addEventListener('click', () => {
                    this.questionInput.value = suggestion;
                    this.askQuestion();
                });
                this.followUp.appendChild(suggestionDiv);
            });
        } else {
            this.followUp.innerHTML = '';
        }

        // Display disclaimer
        if (data.disclaimer) {
            const disclaimerDiv = document.createElement('div');
            disclaimerDiv.className = 'disclaimer';
            disclaimerDiv.innerHTML = `<p><strong>⚠️ Note:</strong> ${data.disclaimer}</p>`;
            this.result.appendChild(disclaimerDiv);
        }

        this.showResult();
    }
    formatAnswer(answer) {
        // Convert line breaks to HTML
        return answer.replace(/\n/g, '<br>');
    }

    showLoading() {
        this.loading.classList.remove('hidden');
        this.askBtn.disabled = true;
    }

    hideLoading() {
        this.loading.classList.add('hidden');
        this.askBtn.disabled = false;
    }

    showResult() {
        this.result.classList.remove('hidden');
    }

    hideResult() {
        this.result.classList.add('hidden');
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.error.classList.remove('hidden');
    }

    hideError() {
        this.error.classList.add('hidden');
    }

    clearAll() {
        this.questionInput.value = '';
        this.hideResult();
        this.hideError();
        this.questionInput.focus();
         }

    async copyAnswer() {
        const answerText = this.answer.textContent;
        if (!answerText) {
            this.showError('No answer to copy');
            return;
        }

        try {
            await navigator.clipboard.writeText(answerText);
            // Show success feedback
            const originalText = this.copyBtn.textContent;
            this.copyBtn.textContent = 'Copied!';
            setTimeout(() => {
                this.copyBtn.textContent = originalText;
            }, 2000);
        } catch (error) {
            console.error('Failed to copy:', error);
            this.showError('Failed to copy answer');
        }
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new PolicyAssistant();
});