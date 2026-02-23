import React, { useState } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, ShieldAlert, ShieldCheck, MessageSquare, ArrowRight, RefreshCw, BarChart3 } from 'lucide-react';
import confetti from 'canvas-confetti';

const API_BASE = 'http://localhost:8000';

const INITIAL_DATA = {
  cash_flow_rate: '',
  cash_flow_to_sales: '',
  cash_flow_to_liability: '',
  current_ratio: '',
  quick_ratio: '',
  cash_current_liability: '',
  debt_ratio: '',
  liability_to_equity: '',
  interest_coverage_ratio: '',
  dfl: '',
  roa: '',
  operating_gross_margin: '',
  gross_profit_to_sales: '',
  net_income_to_total_assets: '',
  revenue_growth_rate: '',
  accounts_receivable_turnover: '',
  inventory_turnover_rate: '',
  average_collection_days: ''
};

const FEATURE_LABELS = {
  cash_flow_rate: "Cash Flow Rate",
  cash_flow_to_sales: "Cash Flow to Sales",
  cash_flow_to_liability: "Cash Flow to Liability",
  current_ratio: "Current Ratio",
  quick_ratio: "Quick Ratio",
  cash_current_liability: "Cash/Current Liability",
  debt_ratio: "Debt Ratio %",
  liability_to_equity: "Liability to Equity",
  interest_coverage_ratio: "Interest Coverage Ratio",
  dfl: "DFL (Financial Leverage)",
  roa: "ROA",
  operating_gross_margin: "Operating Gross Margin",
  gross_profit_to_sales: "Gross Profit to Sales",
  net_income_to_total_assets: "Net Income to Assets",
  revenue_growth_rate: "Revenue Growth Rate",
  accounts_receivable_turnover: "A/R Turnover",
  inventory_turnover_rate: "Inventory Turnover",
  average_collection_days: "Avg Collection Days"
};

function App() {
  const [formData, setFormData] = useState(INITIAL_DATA);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [chatLoading, setChatLoading] = useState(false);
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');

  const handleInputChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      // Convert all to floats
      const payload = {};
      Object.keys(formData).forEach(key => {
        payload[key] = parseFloat(formData[key]) || 0;
      });

      const res = await axios.post(`${API_BASE}/predict`, payload);
      setPrediction(res.data);
      
      if (res.data.status === 'Low Risk') {
        confetti({
          particleCount: 150,
          spread: 70,
          origin: { y: 0.6 },
          colors: ['#6366f1', '#a855f7', '#22c55e']
        });
      }
    } catch (err) {
      console.error(err);
      alert("Error reaching backend. Is the FastAPI server running?");
    } finally {
      setLoading(false);
    }
  };

  const handleChat = async () => {
    if (!userInput.trim()) return;
    
    const newMsg = { role: 'user', text: userInput };
    setMessages([...messages, newMsg]);
    setUserInput('');
    setChatLoading(true);

    try {
      const res = await axios.post(`${API_BASE}/chat`, {
        prediction_result: prediction,
        user_data: formData,
        message: userInput
      });
      setMessages(prev => [...prev, { role: 'bot', text: res.data.response }]);
    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, { role: 'bot', text: "Sorry, I couldn't reach the AI advisor right now." }]);
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen p-4 md:p-8">
      {/* Background Blobs */}
      <div className="bg-blobs">
        <div className="blob blob-1"></div>
        <div className="blob blob-2"></div>
      </div>

      <header className="max-w-6xl mx-auto mb-12 text-center pt-8">
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card mb-4"
        >
          <Activity size={18} className="text-indigo-400" />
          <span className="text-sm font-medium tracking-wide">MSME FINANCIAL GUARDIAN</span>
        </motion.div>
        <motion.h1 
          className="text-5xl md:text-7xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-500 mb-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          Predict. Analyze. <span className="text-indigo-400">Prosper.</span>
        </motion.h1>
        <p className="text-gray-400 text-lg max-w-2xl mx-auto">
          AI-powered bankruptcy prediction and financial advisory for Modern Enterprises.
        </p>
      </header>

      <main className="max-w-6xl mx-auto">
        <AnimatePresence mode="wait">
          {!prediction ? (
            <motion.div 
              key="form"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 1.05 }}
              className="glass-card p-8 md:p-12"
            >
              <div className="flex items-center justify-between mb-8">
                <h2 className="text-2xl font-semibold">Financial Assessment</h2>
                <div className="text-sm text-gray-500 font-mono">18 PARAMETERS</div>
              </div>

              <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {Object.keys(INITIAL_DATA).map((key) => (
                  <div key={key}>
                    <label className="block text-sm font-medium text-gray-400 mb-2">
                      {FEATURE_LABELS[key]}
                    </label>
                    <input
                      type="number"
                      step="any"
                      name={key}
                      value={formData[key]}
                      onChange={handleInputChange}
                      className="glass-input"
                      placeholder="0.00"
                      required
                    />
                  </div>
                ))}
                <div className="md:col-span-3 pt-6 flex justify-end">
                  <button type="submit" className="primary-btn flex items-center gap-2" disabled={loading}>
                    {loading ? <RefreshCw className="animate-spin" /> : <ArrowRight />}
                    {loading ? 'Analyzing Data...' : 'Generate Prediction'}
                  </button>
                </div>
              </form>
            </motion.div>
          ) : (
            <motion.div 
              key="dashboard"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="grid grid-cols-1 md:grid-cols-3 gap-6"
            >
              {/* Prediction Result */}
              <div className="md:col-span-2 glass-card p-8 flex flex-col justify-center items-center text-center relative overflow-hidden">
                {prediction.status === 'High Risk' ? (
                  <ShieldAlert size={80} className="text-red-500 mb-4 animate-pulse" />
                ) : (
                  <ShieldCheck size={80} className="text-green-500 mb-4" />
                )}
                <h3 className="text-3xl font-bold mb-2">
                  Health Status: <span className={prediction.status === 'High Risk' ? 'text-red-500' : 'text-green-500'}>
                    {prediction.status}
                  </span>
                </h3>
                <p className="text-gray-400 text-lg mb-6">
                  Confidence Score: {(prediction.probability * 100).toFixed(1)}% Risk Level
                </p>
                <div className="w-full max-w-md h-4 bg-white/5 rounded-full overflow-hidden mb-8 border border-white/10">
                  <motion.div 
                    initial={{ width: 0 }}
                    animate={{ width: `${prediction.probability * 100}%` }}
                    className={`h-full ${prediction.status === 'High Risk' ? 'bg-red-500' : 'bg-green-500'}`}
                  />
                </div>
                <button 
                  onClick={() => setPrediction(null)} 
                  className="px-6 py-2 rounded-xl border border-white/10 hover:bg-white/5 transition-colors text-sm"
                >
                  Restart Assessment
                </button>
              </div>

              {/* Chatbot Entry */}
              <div className="glass-card p-8 flex flex-col items-center justify-center text-center border-indigo-500/30">
                <div className="w-16 h-16 rounded-2xl bg-indigo-500/20 flex items-center justify-center mb-4">
                  <MessageSquare size={32} className="text-indigo-400" />
                </div>
                <h3 className="text-xl font-bold mb-2">AI Financial Advisor</h3>
                <p className="text-sm text-gray-400 mb-6">
                  Get personalized recommendations based on your risk profile.
                </p>
                <button 
                  onClick={() => setChatOpen(true)}
                  className="primary-btn w-full flex items-center justify-center gap-2"
                >
                  Start Conversation
                </button>
              </div>

              {/* Data Summary Grid */}
              <div className="md:col-span-3 glass-card p-8 grid grid-cols-2 md:grid-cols-6 gap-4">
                 <div className="md:col-span-6 mb-4 flex items-center gap-2">
                    <BarChart3 size={20} className="text-gray-400" />
                    <span className="font-semibold text-gray-400">Parameter Breakdown</span>
                 </div>
                 {Object.entries(formData).slice(0, 6).map(([key, val]) => (
                   <div key={key} className="p-4 rounded-xl bg-white/5 border border-white/5">
                      <div className="text-xs text-gray-500 mb-1">{FEATURE_LABELS[key]}</div>
                      <div className="font-mono text-indigo-400">{val}</div>
                   </div>
                 ))}
                 <div className="md:col-span-6 text-center text-xs text-gray-600 italic">
                    Showing top liquidity and solvency metrics
                 </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Chat Sidebar/Modal */}
      <AnimatePresence>
        {chatOpen && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-end md:items-center justify-center p-4 bg-black/60 backdrop-blur-md"
          >
            <motion.div 
              initial={{ y: 100, scale: 0.9 }}
              animate={{ y: 0, scale: 1 }}
              exit={{ y: 20, opacity: 0 }}
              className="glass-card w-full max-w-2xl h-[80vh] flex flex-col overflow-hidden"
            >
              <div className="p-6 border-b border-white/10 flex justify-between items-center bg-white/5">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-indigo-500 flex items-center justify-center">
                    <MessageSquare size={20} color="white" />
                  </div>
                  <div>
                    <h4 className="font-bold">Gemini Advisor</h4>
                    <span className="text-xs text-green-400">Online & Analyzing</span>
                  </div>
                </div>
                <button onClick={() => setChatOpen(false)} className="text-gray-400 hover:text-white">Close</button>
              </div>

              <div className="flex-1 overflow-y-auto p-6 space-y-4">
                {messages.length === 0 && (
                  <div className="text-center text-gray-500 py-12">
                     <p>I've reviewed your results. Ask me anything about how to improve these metrics.</p>
                  </div>
                )}
                {messages.map((m, i) => (
                  <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[80%] p-4 rounded-2xl ${m.role === 'user' ? 'bg-indigo-600' : 'bg-white/10'} border border-white/10`}>
                      {m.text}
                    </div>
                  </div>
                ))}
                {chatLoading && (
                  <div className="flex justify-start">
                    <div className="animate-pulse bg-white/5 p-4 rounded-2xl border border-white/10 italic text-gray-500">
                      Gemini is thinking...
                    </div>
                  </div>
                )}
              </div>

              <div className="p-6 bg-white/5 border-t border-white/10">
                <div className="flex gap-2">
                  <input 
                    type="text" 
                    value={userInput}
                    onChange={(e) => setUserInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleChat()}
                    placeholder="Ask about debt restructuring, ROA improvement..."
                    className="glass-input"
                  />
                  <button onClick={handleChat} className="primary-btn p-3 aspect-square flex items-center justify-center">
                    <ArrowRight />
                  </button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <footer className="max-w-6xl mx-auto mt-24 pb-8 text-center text-gray-500 text-sm">
        &copy; 2026 MSME Failure Prediction Project &bull; Powered by Google Gemini
      </footer>
    </div>
  );
}

export default App;
