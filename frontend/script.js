// ============================================================================
// MSME FAILURE PREDICTOR - FRONTEND JAVASCRIPT
// ============================================================================

const API_BASE_URL = 'http://localhost:5000/api';
let companiesData = [];

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', function () {
    setupEventListeners();
    setupChatEventListeners();
    console.log('Dashboard initialized');
});

function setupEventListeners() {
    // Tab navigation
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', handleTabSwitch);
    });

    // Form submission
    document.getElementById('predictionForm').addEventListener('submit', handleFormSubmit);

    // Enter key on form inputs
    document.querySelectorAll('.form-input').forEach(input => {
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                document.getElementById('predictionForm').dispatchEvent(new Event('submit'));
            }
        });
    });
}

// ============================================================================
// TAB NAVIGATION
// ============================================================================

function handleTabSwitch(e) {
    const tabName = e.currentTarget.dataset.tab;

    // Update active button
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    e.currentTarget.classList.add('active');

    // Update active tab
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.getElementById(tabName).classList.add('active');

    // Update dashboard if switching to it
    if (tabName === 'dashboard') {
        updateDashboard();
    }
}

// ============================================================================
// FORM HANDLING
// ============================================================================

function handleFormSubmit(e) {
    e.preventDefault();

    const companyName = document.getElementById('companyName').value || 'Company_' + Date.now();
    const features = collectFormData();

    // Validate features
    const { isValid, errors } = validateFeatures(features);
    if (!isValid) {
        showNotification(`Missing required fields: ${errors.join(', ')}`, 'error');
        return;
    }

    // Show loading
    showLoading(true);
    document.getElementById('submitBtn').disabled = true;

    // Make API call
    makePrediction(features, companyName)
        .then(result => {
            handlePredictionResult(result, companyName);
            showLoading(false);
            document.getElementById('submitBtn').disabled = false;
        })
        .catch(error => {
            console.error('Prediction error:', error);
            showNotification(`Error: ${error.message}`, 'error');
            showLoading(false);
            document.getElementById('submitBtn').disabled = false;
        });
}

function collectFormData() {
    const features = {};
    document.querySelectorAll('.form-input[data-feature]').forEach(input => {
        const feature = input.dataset.feature;
        const value = parseFloat(input.value) || null;
        if (value !== null) {
            features[feature] = value;
        }
    });
    return features;
}

function validateFeatures(features) {
    const requiredFeatures = [
        'Cash flow rate', 'Cash Flow to Sales', 'Cash Flow to Liability',
        'Current Ratio', 'Quick Ratio', 'Cash/Current Liability',
        'Debt ratio %', 'Liability to Equity', 'Interest Coverage Ratio', 'DFL',
        'ROA', 'Operating Gross Margin', 'Gross Profit to Sales',
        'Net Income to Total Assets', 'Revenue Growth Rate',
        'Accounts Receivable Turnover', 'Inventory Turnover Rate',
        'Average Collection Days'
    ];

    const missing = requiredFeatures.filter(f => !(f in features));
    return {
        isValid: missing.length === 0,
        errors: missing
    };
}

// ============================================================================
// API CALLS
// ============================================================================

function makePrediction(features, companyName) {
    return fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ features })
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`API Error: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            data.companyName = companyName;
            data.features = features;
            return data;
        });
}

// ============================================================================
// RESULT HANDLING
// ============================================================================

function handlePredictionResult(result, companyName) {
    // Store in companies data
    companiesData.unshift({
        id: Date.now(),
        name: companyName,
        ...result
    });

    // Update chat context so the advisor knows about this prediction
    updateChatContext(result, companyName);

    // Display results
    displayPredictionResults(result);
    displaySuggestions(result.suggestions);

    // Show success message
    showNotification('✓ Prediction completed! 💬 Chat advisor updated with your data.', 'success');

    // Scroll to results
    setTimeout(() => {
        document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
    }, 300);
}

function displayPredictionResults(result) {
    const riskScore = result.bankruptcy_risk_score;
    const healthScore = result.safe_score;
    const isPredicted = result.prediction === 1;

    // Update risk card
    document.getElementById('riskPercentage').textContent = (riskScore * 100).toFixed(1) + '%';
    document.getElementById('riskLabel').textContent = getRiskLabel(riskScore);

    const riskCard = document.getElementById('riskCard');
    riskCard.style.borderColor = getRiskColor(riskScore);

    // Update health card
    document.getElementById('healthPercentage').textContent = (healthScore * 100).toFixed(1) + '%';

    // Update status card
    const statusIcon = document.getElementById('statusIcon');
    const statusValue = document.getElementById('statusValue');
    const statusDescription = document.getElementById('statusDescription');

    if (isPredicted) {
        statusIcon.textContent = '📉';
        statusValue.textContent = 'AT RISK';
        statusValue.style.color = getRiskColor(riskScore);
    } else {
        statusIcon.textContent = '📈';
        statusValue.textContent = 'STABLE';
        statusValue.style.color = '#10b981';
    }
    statusDescription.textContent = getRiskLabel(riskScore) + ' Risk Level';

    // Update gauge
    const gaugeFill = document.querySelector('.gauge-fill');
    const fillPercentage = Math.min(riskScore * 100, 100);
    gaugeFill.style.width = fillPercentage + '%';

    // Get color based on position
    let gaugeFillColor;
    if (fillPercentage <= 33) {
        gaugeFillColor = '#44FF44';
    } else if (fillPercentage <= 66) {
        gaugeFillColor = '#FFA500';
    } else {
        gaugeFillColor = '#FF4444';
    }
    gaugeFill.style.background = gaugeFillColor;

    // Show results section
    document.getElementById('resultsSection').classList.remove('hidden');
}

function displaySuggestions(suggestions) {
    const container = document.getElementById('suggestionsContainer');

    if (!suggestions || !suggestions.advice) {
        container.innerHTML = '<p>No suggestions available.</p>';
        document.getElementById('suggestionsSection').classList.remove('hidden');
        return;
    }

    if (Array.isArray(suggestions.advice)) {
        // Array format (rule-based)
        const html = suggestions.advice.map(suggestion => `
            <div class="suggestion-card">
                <h4>
                    ${suggestion.category}
                    <span class="priority-badge priority-${suggestion.priority.toLowerCase()}">
                        ${suggestion.priority}
                    </span>
                </h4>
                <p class="suggestion-issue">${suggestion.issue}</p>
                <p class="suggestion-action">${suggestion.action}</p>
            </div>
        `).join('');
        container.innerHTML = html;
    } else {
        // Text format (AI-generated) — render full text with basic formatting
        const formatted = simpleMarkdown(suggestions.advice);
        const sourceTag = suggestions.source === 'huggingface'
            ? '<span style="font-size:0.78rem;background:#dcfce7;color:#15803d;padding:2px 8px;border-radius:999px;font-weight:600;margin-left:0.5rem;">AI Generated</span>'
            : '<span style="font-size:0.78rem;background:#fef3c7;color:#92400e;padding:2px 8px;border-radius:999px;font-weight:600;margin-left:0.5rem;">Rule-based</span>';
        container.innerHTML = `
            <div class="suggestion-card">
                <h4>AI Advisor Recommendations ${sourceTag}</h4>
                <div style="white-space:pre-wrap;line-height:1.7;">${formatted}</div>
            </div>
        `;
    }

    document.getElementById('suggestionsSection').classList.remove('hidden');
}

// ============================================================================
// DASHBOARD
// ============================================================================

function updateDashboard() {
    if (companiesData.length === 0) {
        document.getElementById('emptyState').style.display = 'block';
        document.querySelector('.table-container').style.display = 'none';
        document.querySelector('.stats-grid').style.display = 'none';
        return;
    }

    document.getElementById('emptyState').style.display = 'none';
    document.querySelector('.table-container').style.display = 'block';
    document.querySelector('.stats-grid').style.display = 'grid';

    // Update stats
    updateStats();

    // Update table
    updateCompaniesTable();
}

function updateStats() {
    const total = companiesData.length;
    const highRisk = companiesData.filter(c => c.bankruptcy_risk_score > 0.5).length;
    const mediumRisk = companiesData.filter(c => c.bankruptcy_risk_score > 0.3 && c.bankruptcy_risk_score <= 0.5).length;
    const lowRisk = companiesData.filter(c => c.bankruptcy_risk_score <= 0.3).length;

    document.getElementById('totalCompanies').textContent = total;
    document.getElementById('highRiskCount').textContent = highRisk;
    document.getElementById('mediumRiskCount').textContent = mediumRisk;
    document.getElementById('lowRiskCount').textContent = lowRisk;
}

function updateCompaniesTable() {
    const tbody = document.getElementById('tableBody');

    if (companiesData.length === 0) {
        tbody.innerHTML = '<tr class="empty-row"><td colspan="5">No companies analyzed yet.</td></tr>';
        return;
    }

    const html = companiesData.map(company => {
        const riskScore = (company.bankruptcy_risk_score * 100).toFixed(1);
        const healthScore = (company.safe_score * 100).toFixed(1);
        const riskLevel = getRiskLabel(company.bankruptcy_risk_score);

        let riskClass = 'risk-score-low';
        let statusClass = 'status-badge-low';

        if (company.bankruptcy_risk_score > 0.5) {
            riskClass = 'risk-score-high';
            statusClass = 'status-badge-high';
        } else if (company.bankruptcy_risk_score > 0.3) {
            riskClass = 'risk-score-medium';
            statusClass = 'status-badge-medium';
        }

        return `
            <tr>
                <td><strong>${company.name}</strong></td>
                <td><span class="${riskClass}">${riskScore}%</span></td>
                <td><strong style="color: #10b981;">${healthScore}%</strong></td>
                <td><span class="status-badge-table ${statusClass}">${riskLevel}</span></td>
                <td><button class="btn btn-primary" onclick="viewCompanyDetails(${company.id})" style="padding: 0.5rem 1rem; font-size: 0.9rem;">View</button></td>
            </tr>
        `;
    }).join('');

    tbody.innerHTML = html;
}

function viewCompanyDetails(id) {
    const company = companiesData.find(c => c.id === id);
    if (company) {
        alert(`Company: ${company.name}\nRisk Score: ${(company.bankruptcy_risk_score * 100).toFixed(1)}%\nHealth Score: ${(company.safe_score * 100).toFixed(1)}%`);
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function getRiskColor(score) {
    if (score > 0.5) return '#FF4444';
    if (score > 0.3) return '#FFA500';
    return '#44FF44';
}

function getRiskLabel(score) {
    if (score > 0.5) return 'HIGH RISK';
    if (score > 0.3) return 'MEDIUM RISK';
    return 'LOW RISK';
}

function showLoading(show) {
    const spinner = document.getElementById('loadingSpinner');
    if (show) {
        spinner.classList.remove('hidden');
    } else {
        spinner.classList.add('hidden');
    }
}

function showNotification(message, type = 'success') {
    const notification = document.getElementById('notification');
    const messageElement = document.getElementById('notificationMessage');

    messageElement.textContent = message;
    notification.className = 'notification';

    if (type === 'error') {
        notification.classList.add('error');
    } else if (type === 'warning') {
        notification.classList.add('warning');
    }

    notification.classList.remove('hidden');

    // Auto-hide after 5 seconds
    setTimeout(() => {
        notification.classList.add('hidden');
    }, 5000);
}

// ============================================================================
// LOAD SAMPLE DATA FOR TESTING
// ============================================================================

function loadSampleData() {
    const sampleCompany = {
        name: 'Distressed Manufacturing Co.',
        features: {
            'Cash flow rate': 0.05,
            'Cash Flow to Sales': -0.10,
            'Cash Flow to Liability': 0.08,
            'Current Ratio': 0.6,
            'Quick Ratio': 0.3,
            'Cash/Current Liability': 0.1,
            'Debt ratio %': 82,
            'Liability to Equity': 4.5,
            'Interest Coverage Ratio': 0.8,
            'DFL': 6.0,
            'ROA': -0.05,
            'Operating Gross Margin': 0.05,
            'Gross Profit to Sales': 0.06,
            'Net Income to Total Assets': -0.08,
            'Revenue Growth Rate': -0.20,
            'Accounts Receivable Turnover': 2.0,
            'Inventory Turnover Rate': 1.5,
            'Average Collection Days': 120,
        }
    };

    // Populate form
    Object.entries(sampleCompany.features).forEach(([feature, value]) => {
        const input = document.querySelector(`[data-feature="${feature}"]`);
        if (input) {
            input.value = value;
        }
    });

    document.getElementById('companyName').value = sampleCompany.name;

    showNotification('⚠️ Sample data loaded', 'warning');
}

// Add load sample button functionality (can be triggered from console)
console.log('To load sample data, run: loadSampleData()');

// ============================================================================
// CONVERSATIONAL CHAT ADVISOR
// ============================================================================

let chatContext = null;       // Stores the latest prediction result + features
let chatHistory = [];         // Multi-turn conversation history for Gemini
let isChatSending = false;    // Prevent double-send

/**
 * Wires up keyboard listener for the chat input (Enter to send).
 * Called once at DOMContentLoaded.
 */
function setupChatEventListeners() {
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendChatMessage();
            }
        });
    }
}

/**
 * Populate chatContext with the latest prediction result so the chatbot
 * can give contextual answers. Called inside handlePredictionResult().
 */
function updateChatContext(result, companyName) {
    chatContext = {
        companyName: companyName,
        features: result.features,
        bankruptcy_risk_score: result.bankruptcy_risk_score,
        safe_score: result.safe_score,
        risk_level: result.risk_level,
        status: result.status,
        prediction: result.prediction
    };

    // Update the context banner in the Chat tab
    const banner = document.getElementById('chatContextBanner');
    const noCtx = document.getElementById('chatNoContext');
    const nameEl = document.getElementById('contextCompanyName');
    const badge = document.getElementById('contextRiskBadge');

    if (banner && nameEl && badge) {
        nameEl.textContent = companyName;
        badge.textContent = result.risk_level + ' — ' + (result.bankruptcy_risk_score * 100).toFixed(1) + '%';

        badge.className = 'context-risk-badge';
        if (result.bankruptcy_risk_score > 0.5) badge.classList.add('high');
        else if (result.bankruptcy_risk_score > 0.3) badge.classList.add('medium');
        else badge.classList.add('low');

        banner.classList.remove('hidden');
        if (noCtx) noCtx.style.display = 'none';
    }

    // Flash the Chat nav badge to nudge user to the chat tab
    const chatBadge = document.getElementById('chatNewBadge');
    if (chatBadge) chatBadge.classList.remove('hidden');
}

/**
 * Send the current chat input to /api/chat, render bubbles and typing indicator.
 */
async function sendChatMessage() {
    if (isChatSending) return;

    const input = document.getElementById('chatInput');
    const message = (input ? input.value : '').trim();
    if (!message) return;

    isChatSending = true;
    input.value = '';
    input.disabled = true;
    document.getElementById('chatSendBtn').disabled = true;

    // Render user bubble
    appendMessage('user', message);

    // Add to history
    chatHistory.push({ role: 'user', text: message });

    // Show typing indicator
    showTypingIndicator(true);

    try {
        const resp = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                history: chatHistory.slice(-20),   // last 20 turns
                context: chatContext || null
            })
        });

        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        const reply = data.reply || 'Sorry, I could not generate a response.';

        showTypingIndicator(false);
        appendMessage('bot', reply);

        // Add bot reply to history
        chatHistory.push({ role: 'bot', text: reply });

    } catch (err) {
        console.error('Chat error:', err);
        showTypingIndicator(false);
        appendMessage('bot', '⚠️ Could not reach the advisor. Please ensure the backend is running and try again.');
    } finally {
        isChatSending = false;
        input.disabled = false;
        document.getElementById('chatSendBtn').disabled = false;
        input.focus();
    }
}

/**
 * Append a message bubble to the chat feed.
 * @param {'user'|'bot'} role
 * @param {string}       text
 */
function appendMessage(role, text) {
    const feed = document.getElementById('chatMessages');
    if (!feed) return;

    const wrapper = document.createElement('div');
    wrapper.className = `message message-${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? '👤' : '🤖';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    if (role === 'bot') {
        // Render basic markdown for bot responses (bold, lists, line breaks)
        bubble.innerHTML = simpleMarkdown(text);
    } else {
        bubble.textContent = text;   // textContent protects against XSS for user input
    }

    wrapper.appendChild(avatar);
    wrapper.appendChild(bubble);
    feed.appendChild(wrapper);

    // Auto-scroll to bottom
    feed.scrollTop = feed.scrollHeight;
}

/**
 * Show or hide the animated typing indicator.
 */
function showTypingIndicator(show) {
    const indicator = document.getElementById('typingIndicator');
    const feed = document.getElementById('chatMessages');
    if (!indicator || !feed) return;

    if (show) {
        // Move indicator just below the feed so it's always visible
        feed.parentNode.insertBefore(indicator, feed.nextSibling);
        indicator.classList.remove('hidden');
    } else {
        indicator.classList.add('hidden');
    }

    if (feed) feed.scrollTop = feed.scrollHeight;
}

/**
 * Handle a quick-action chip click: inject text and send.
 */
function handleQuickChip(text) {
    const input = document.getElementById('chatInput');
    if (input) input.value = text;
    sendChatMessage();
}

/**
 * Clear the chat history and reset the message feed to the welcome message.
 */
function clearChat() {
    chatHistory = [];
    const feed = document.getElementById('chatMessages');
    if (feed) {
        feed.innerHTML = `
            <div class="message message-bot">
                <div class="message-avatar">🤖</div>
                <div class="message-bubble">
                    <p>Chat cleared! Ask me anything about MSME financial health, your risk score, or improvement strategies.</p>
                </div>
            </div>`;
    }
}

/**
 * Simple markdown renderer for bot/AI text.
 * Converts **bold**, bullet lists, and line breaks to HTML.
 */
function simpleMarkdown(text) {
    // Escape HTML entities to prevent XSS
    let html = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // Bold: **text** or __text__
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/__(.+?)__/g, '<strong>$1</strong>');

    // Numbered lists: "1. item" at start of line
    html = html.replace(/^(\d+)\.\s+(.+)$/gm, '<li>$2</li>');

    // Bullet lists: "- item" or "• item" at start of line
    html = html.replace(/^[-•]\s+(.+)$/gm, '<li>$1</li>');

    // Wrap consecutive <li> in <ul>
    html = html.replace(/(<li>.*?<\/li>\n?)+/gs, (match) => `<ul>${match}</ul>`);

    // Line breaks
    html = html.replace(/\n/g, '<br>');

    return html;
}
