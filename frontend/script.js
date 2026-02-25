// ============================================================================
// MSME FAILURE PREDICTOR - FRONTEND JAVASCRIPT
// ============================================================================

const API_BASE_URL = 'http://localhost:5000/api';
let companiesData = [];

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
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

    // Display results
    displayPredictionResults(result);
    displaySuggestions(result.suggestions);

    // Show success message
    showNotification('✓ Prediction completed successfully!', 'success');

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
        // Text format (Gemini API)
        const text = suggestions.advice;
        const html = `
            <div class="suggestion-card">
                <p>${text.substring(0, 500)}${text.length > 500 ? '...' : ''}</p>
            </div>
        `;
        container.innerHTML = html;
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
        name: 'Sample Company',
        features: {
            'Cash flow rate': 0.5,
            'Cash Flow to Sales': 0.3,
            'Cash Flow to Liability': 0.25,
            'Current Ratio': 1.8,
            'Quick Ratio': 1.2,
            'Cash/Current Liability': 0.4,
            'Debt ratio %': 45,
            'Liability to Equity': 1.2,
            'Interest Coverage Ratio': 3.5,
            'DFL': 1.5,
            'ROA': 0.08,
            'Operating Gross Margin': 0.35,
            'Gross Profit to Sales': 0.4,
            'Net Income to Total Assets': 0.06,
            'Revenue Growth Rate': 0.10,
            'Accounts Receivable Turnover': 4,
            'Inventory Turnover Rate': 3,
            'Average Collection Days': 45,
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

    showNotification('Sample data loaded. Click "Predict & Get Advice" to test.', 'success');
}

// Add load sample button functionality (can be triggered from console)
console.log('To load sample data, run: loadSampleData()');
