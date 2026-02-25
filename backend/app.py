"""
MSME Failure Predictor - Backend API
Flask REST API for ML predictions and AI advisory
"""

import sys
import os

# Ensure repo root is on the path so 'backend.model_config' resolves correctly
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from datetime import datetime
import logging
from dotenv import load_dotenv
import google.generativeai as genai

from backend.model_config import (
    ModelLoader, validate_features, get_feature_health_score,
    get_feature_interpretation, FEATURE_MAP, FEATURE_CATEGORIES,
    HEALTHY_RANGES
)

# ============================================================================
# CONFIGURATION & LOGGING
# ============================================================================

load_dotenv(os.path.join(_REPO_ROOT, '.env'))

app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(_REPO_ROOT, 'backend.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# MODEL & AI SETUP
# ============================================================================

try:
    model_loader = ModelLoader()
    if not model_loader.is_loaded:
        logger.warning("Model not loaded. Run python model/train.py to train the model.")
except Exception as e:
    logger.error(f"Error initializing model: {str(e)}")
    model_loader = None

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model_gemini = genai.GenerativeModel('gemini-1.5-pro')
        logger.info("✓ Gemini API configured")
    except Exception as e:
        logger.warning(f"Gemini API configuration failed: {str(e)}")
        model_gemini = None
else:
    logger.info("Gemini API key not provided. Using fallback suggestions.")
    model_gemini = None

# ============================================================================
# AI ADVISOR
# ============================================================================

# ============================================================================
# CHAT ADVISOR
# ============================================================================

class ChatAdvisor:
    """Handle multi-turn conversational chat with financial context"""

    FALLBACK_RESPONSES = {
        'risk': "Your bankruptcy risk score reflects the combined weight of your financial ratios. High debt levels, low liquidity, and declining profitability are the primary drivers. Focus on improving your current ratio above 1.5 and reducing your debt ratio below 50%.",
        'liquidity': "To improve liquidity: (1) Speed up accounts receivable collection, (2) Negotiate longer payment terms with suppliers, (3) Reduce unnecessary inventory, (4) Establish a revolving credit line as a safety net.",
        'debt': "To reduce debt: (1) Prioritize paying down high-interest loans, (2) Refinance existing debt at lower rates, (3) Consider equity financing to replace debt, (4) Avoid taking on new debt until ratios improve.",
        'profit': "To improve profitability: (1) Review and optimise your cost structure, (2) Improve pricing strategy, (3) Focus on high-margin products/services, (4) Reduce operational inefficiencies.",
        'revenue': "To grow revenue: (1) Expand into new market segments, (2) Strengthen customer retention programs, (3) Explore strategic partnerships, (4) Invest in marketing for high-ROI channels.",
        'default': "I'm your MSME financial advisor. I can help you understand your risk score, explain financial ratios, and suggest improvements. Please run a prediction first for personalized context-aware advice."
    }

    @staticmethod
    def get_reply(message, history, context):
        """Get a chat reply, using Gemini if available"""
        if model_gemini:
            return ChatAdvisor._get_gemini_reply(message, history, context)
        return ChatAdvisor._get_fallback_reply(message, context)

    @staticmethod
    def _build_system_prompt(context):
        """Build a system prompt embedding the prediction context"""
        if not context:
            return (
                "You are a friendly and concise financial advisor specializing in MSME "
                "(Micro, Small & Medium Enterprises). Help users understand financial ratios, "
                "bankruptcy risk, and improvement strategies. "
                "Keep answers short, practical, and actionable."
            )

        features = context.get('features', {})
        risk_score = context.get('bankruptcy_risk_score', 0)
        status = context.get('status', 'UNKNOWN')
        risk_level = context.get('risk_level', 'UNKNOWN')
        company = context.get('companyName', 'the company')

        features_text = '\n'.join(
            f"  - {k}: {v:.4f}" for k, v in features.items()
        ) if features else '  (No data provided)'

        return f"""You are a friendly, concise financial advisor specializing in MSME financial health.

The user has just run a bankruptcy prediction for **{company}**. Here are their results:
- Bankruptcy Risk Score: {risk_score:.2%}
- Risk Level: {risk_level}
- Status: {status}

Their financial metrics are:
{features_text}

Use this data to give personalized, specific, actionable advice. Reference exact metric values when relevant.
Keep responses concise (2-4 sentences or a short bullet list). Be encouraging but honest."""

    @staticmethod
    def _get_gemini_reply(message, history, context):
        """Get reply from Gemini with conversation history"""
        try:
            system_prompt = ChatAdvisor._build_system_prompt(context)

            # Build conversation turns
            conversation_parts = [system_prompt, "\n\n"]
            for turn in history[-10:]:  # Keep last 10 turns for context window
                role = turn.get('role', 'user')
                text = turn.get('text', '')
                prefix = 'User' if role == 'user' else 'Advisor'
                conversation_parts.append(f"{prefix}: {text}\n")
            conversation_parts.append(f"User: {message}\nAdvisor:")

            full_prompt = ''.join(conversation_parts)
            response = model_gemini.generate_content(full_prompt)

            return {
                'reply': response.text.strip(),
                'source': 'gemini',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.warning(f"Gemini chat error: {str(e)}")
            return ChatAdvisor._get_fallback_reply(message, context)

    @staticmethod
    def _get_fallback_reply(message, context):
        """Keyword-based fallback when Gemini is unavailable"""
        msg_lower = message.lower()
        if any(w in msg_lower for w in ['risk', 'bankrupt', 'score', 'percentage']):
            reply = ChatAdvisor.FALLBACK_RESPONSES['risk']
        elif any(w in msg_lower for w in ['liquid', 'current ratio', 'cash', 'quick']):
            reply = ChatAdvisor.FALLBACK_RESPONSES['liquidity']
        elif any(w in msg_lower for w in ['debt', 'liabilit', 'leverage', 'borrow']):
            reply = ChatAdvisor.FALLBACK_RESPONSES['debt']
        elif any(w in msg_lower for w in ['profit', 'margin', 'roa', 'income']):
            reply = ChatAdvisor.FALLBACK_RESPONSES['profit']
        elif any(w in msg_lower for w in ['revenue', 'sales', 'grow', 'market']):
            reply = ChatAdvisor.FALLBACK_RESPONSES['revenue']
        else:
            reply = ChatAdvisor.FALLBACK_RESPONSES['default']

        return {
            'reply': reply,
            'source': 'fallback',
            'timestamp': datetime.now().isoformat()
        }


class AIAdvisor:
    """Generate improvement suggestions using AI"""

    @staticmethod
    def get_suggestions(features, risk_score, is_bankrupt):
        """Get improvement suggestions"""
        if model_gemini:
            return AIAdvisor._get_gemini_suggestions(features, risk_score, is_bankrupt)
        else:
            return AIAdvisor._get_rule_based_suggestions(features, is_bankrupt)

    @staticmethod
    def _get_gemini_suggestions(features, risk_score, is_bankrupt):
        """Get suggestions from Gemini API"""
        try:
            prompt = f"""
You are a financial advisor specializing in MSME (Micro, Small & Medium Enterprises) financial health.

Based on the following financial metrics, provide 3-5 specific, actionable recommendations to improve the company's financial health and reduce bankruptcy risk.

**Financial Metrics:**
{json.dumps({k: f"{v:.4f}" for k, v in features.items()}, indent=2)}

**Bankruptcy Risk Score:** {risk_score:.2%}
**Current Status:** {'High Risk - BANKRUPTCY WARNING' if is_bankrupt else 'Financially Stable'}

**Please provide:**
1. Most critical issues that need immediate attention
2. Specific metrics that are concerning and their implications
3. Concrete, actionable steps to improve each metric
4. Expected timeline for improvements
5. Risk mitigation strategies

Keep the response concise and actionable."""

            response = model_gemini.generate_content(prompt)
            return {
                'advice': response.text,
                'source': 'gemini',
                'generated_at': datetime.now().isoformat()
            }
        except Exception as e:
            logger.warning(f"Gemini API error: {str(e)}")
            return AIAdvisor._get_rule_based_suggestions(features, is_bankrupt)

    @staticmethod
    def _get_rule_based_suggestions(features, is_bankrupt):
        """Get rule-based suggestions"""
        suggestions = []

        if features.get('Current Ratio', 0) < 1.5:
            suggestions.append({
                'category': 'Liquidity',
                'priority': 'HIGH',
                'issue': f"Low current ratio ({features.get('Current Ratio', 0):.2f}). Short-term liquidity at risk.",
                'action': 'Increase current assets or reduce current liabilities. Optimize working capital management.'
            })

        if features.get('Cash flow rate', 0) < 0.1:
            suggestions.append({
                'category': 'Liquidity',
                'priority': 'HIGH',
                'issue': 'Weak cash flow generation. May struggle to meet obligations.',
                'action': 'Improve cash collection processes. Reduce operational cash burn. Consider asset sales if necessary.'
            })

        if features.get('ROA', 0) < 0.05:
            suggestions.append({
                'category': 'Profitability',
                'priority': 'HIGH',
                'issue': 'Low ROA indicates poor asset utilization efficiency.',
                'action': 'Review operational efficiency. Reduce costs. Improve pricing strategy. Consider business model changes.'
            })

        if features.get('Operating Gross Margin', 0) < 0.15:
            suggestions.append({
                'category': 'Profitability',
                'priority': 'MEDIUM',
                'issue': 'Low operating margin indicates pricing or cost issues.',
                'action': 'Analyze cost structure. Improve operational efficiency. Review pricing competitiveness.'
            })

        if features.get('Debt ratio %', 0) > 60:
            suggestions.append({
                'category': 'Solvency',
                'priority': 'HIGH',
                'issue': f"High debt ratio ({features.get('Debt ratio %', 0):.1f}%). Excessive financial leverage.",
                'action': 'Reduce debt through repayment or refinancing. Increase equity capital. Negotiate better terms with creditors.'
            })

        if features.get('Interest Coverage Ratio', 0) < 2.0:
            suggestions.append({
                'category': 'Solvency',
                'priority': 'HIGH',
                'issue': 'Low interest coverage. Difficulty servicing debt obligations.',
                'action': 'Improve EBIT through revenue growth or cost reduction. Refinance debt at lower rates. Reduce debt principal.'
            })

        if features.get('Average Collection Days', 0) > 90:
            suggestions.append({
                'category': 'Efficiency',
                'priority': 'MEDIUM',
                'issue': 'High average collection days indicates slow receivables collection.',
                'action': 'Tighten credit policies. Improve collection processes. Offer early payment discounts. Review customer creditworthiness.'
            })

        if features.get('Revenue Growth Rate', 0) < 0:
            suggestions.append({
                'category': 'Growth',
                'priority': 'MEDIUM',
                'issue': 'Declining revenue. Business is contracting.',
                'action': 'Explore new markets or product lines. Improve marketing. Enhance customer retention. Consider strategic partnerships.'
            })

        if not suggestions:
            suggestions.append({
                'category': 'General',
                'priority': 'LOW',
                'issue': 'Overall financial metrics appear stable.',
                'action': 'Continue monitoring key indicators. Maintain current financial discipline. Plan for growth opportunities.'
            })

        return {
            'advice': suggestions,
            'source': 'rule_based',
            'generated_at': datetime.now().isoformat()
        }

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loader.is_loaded if model_loader else False,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Expected JSON: {"features": {...}}
    """
    try:
        if not model_loader or not model_loader.is_loaded:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Please ensure the model is trained and saved'
            }), 503

        data = request.get_json()

        if not data or 'features' not in data:
            return jsonify({'error': 'Missing features in request'}), 400

        features = data['features']

        is_valid, errors = validate_features(features)
        if not is_valid:
            return jsonify({
                'error': 'Invalid features',
                'details': errors
            }), 400

        prediction, risk_score, safe_score = model_loader.predict(features)

        is_bankrupt = prediction == 1
        suggestions = AIAdvisor.get_suggestions(features, risk_score, is_bankrupt)

        return jsonify({
            'prediction': int(prediction),
            'bankruptcy_risk_score': float(risk_score),
            'safe_score': float(safe_score),
            'risk_level': 'HIGH' if risk_score > 0.5 else 'MEDIUM' if risk_score > 0.3 else 'LOW',
            'status': 'AT RISK' if prediction == 1 else 'STABLE',
            'suggestions': suggestions,
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint
    Expected JSON: {"companies": [{"name": "...", "features": {...}}, ...]}
    """
    try:
        if not model_loader or not model_loader.is_loaded:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Please ensure the model is trained and saved'
            }), 503

        data = request.get_json()
        companies = data.get('companies', [])

        results = []
        for company in companies:
            try:
                features = company['features']
                is_valid, errors = validate_features(features)

                if not is_valid:
                    results.append({
                        'name': company.get('name', 'Unknown'),
                        'error': 'Invalid features',
                        'details': errors
                    })
                    continue

                prediction, risk_score, safe_score = model_loader.predict(features)

                results.append({
                    'name': company.get('name', 'Unknown'),
                    'prediction': int(prediction),
                    'bankruptcy_risk_score': float(risk_score),
                    'safe_score': float(safe_score),
                    'risk_level': 'HIGH' if risk_score > 0.5 else 'MEDIUM' if risk_score > 0.3 else 'LOW',
                    'status': 'AT RISK' if prediction == 1 else 'STABLE'
                })
            except Exception as e:
                results.append({
                    'name': company.get('name', 'Unknown'),
                    'error': str(e)
                })

        return jsonify({
            'total': len(results),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/feature-insights', methods=['POST'])
def feature_insights():
    """
    Get detailed insights on features
    Expected JSON: {"features": {...}}
    """
    try:
        data = request.get_json()
        features = data.get('features', {})

        insights = {}
        for category, feature_list in FEATURE_CATEGORIES.items():
            category_insights = []

            for feature in feature_list:
                if feature in features:
                    value = features[feature]
                    health_score = get_feature_health_score(feature, value)

                    if health_score >= 80:
                        status = 'EXCELLENT'
                    elif health_score >= 60:
                        status = 'GOOD'
                    elif health_score >= 40:
                        status = 'FAIR'
                    else:
                        status = 'POOR'

                    category_insights.append({
                        'feature': feature,
                        'value': float(value),
                        'health_score': float(health_score),
                        'status': status,
                        'interpretation': get_feature_interpretation(feature, value)
                    })

            if category_insights:
                insights[category] = category_insights

        return jsonify({
            'insights': insights,
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Feature insights error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information and metadata"""
    try:
        metadata = model_loader.get_metadata() if model_loader else {}

        return jsonify({
            'model_loaded': model_loader.is_loaded if model_loader else False,
            'model_type': metadata.get('model_type', 'Unknown'),
            'base_learners': metadata.get('base_learners', []),
            'meta_learner': metadata.get('meta_learner', 'Unknown'),
            'features': list(FEATURE_MAP.keys()),
            'metrics': metadata.get('metrics', {}),
            'training_date': metadata.get('training_date', 'Unknown'),
            'training_samples': metadata.get('training_samples', 'Unknown'),
            'test_samples': metadata.get('test_samples', 'Unknown'),
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Conversational chat endpoint
    Expected JSON: {"message": "...", "history": [...], "context": {...}}
    """
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Missing message in request'}), 400

        message = data['message'].strip()
        if not message:
            return jsonify({'error': 'Empty message'}), 400

        history = data.get('history', [])
        context = data.get('context', None)

        result = ChatAdvisor.get_reply(message, history, context)
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': 'Bad request', 'message': str(e)}), 400


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found', 'message': str(e)}), 404


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500
