import { useState, useEffect } from 'react'
import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js'
import { predict, getAllProducts, getStores } from '../api'

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement,
  Title, Tooltip, Legend, Filler
)

const SHAPBar = ({ name, value }) => (
  <div className="flex items-center gap-2">
    <span className="text-xs w-24 text-right text-gray-600">{name}</span>
    <div className="flex-1 bg-gray-100 rounded-full h-4 overflow-hidden">
      <div
        className="h-full bg-gradient-to-r from-blue-500 to-blue-600 rounded-full transition-all duration-500"
        style={{ width: `${value * 100}%` }}
      />
    </div>
    <span className="text-xs font-medium w-10">{(value * 100).toFixed(0)}%</span>
  </div>
)

export default function Forecast() {
  const [products, setProducts] = useState([])
  const [stores, setStores] = useState([])
  const [formData, setFormData] = useState({
    item_id: '',
    store_id: '',
    forecast_days: 7,
    current_inventory: 100,
    promotion: false,
    holiday: false,
    price: '',
  })

  const [result, setResult] = useState(null)
  const [insights, setInsights] = useState(null)
  const [loading, setLoading] = useState(false)
  const [dataLoading, setDataLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    loadProductsAndStores()
  }, [])

  const loadProductsAndStores = async () => {
    // Generate default lists locally (always works)
    const defaultProducts = []
    const defaultStores = []
    for (let i = 1; i <= 50; i++) {
      defaultProducts.push({ item_id: `item_${i}` })
      defaultStores.push({ store_id: `store_${i}` })
    }

    try {
      const [productsData, storesData] = await Promise.all([
        getAllProducts(),
        getStores(),
      ])
      // Use API data if it has items, otherwise use defaults
      const seen = new Set()
      const finalProducts = productsData.length > 0
        ? productsData.filter(p => { if (seen.has(p.item_id)) return false; seen.add(p.item_id); return true })
        : defaultProducts
      const finalStores = storesData.length > 0 ? storesData : defaultStores
      setProducts(finalProducts)
      setStores(finalStores)
      if (finalProducts.length > 0) {
        setFormData(prev => ({
          ...prev,
          item_id: finalProducts[0].item_id,
          store_id: finalStores[0]?.store_id || '',
        }))
      }
    } catch (err) {
      console.error('API failed, using default lists:', err)
      setProducts(defaultProducts)
      setStores(defaultStores)
      setFormData(prev => ({
        ...prev,
        item_id: 'item_1',
        store_id: 'store_1',
      }))
    } finally {
      setDataLoading(false)
    }
  }

  const handlePredict = async () => {
    setLoading(true)
    setError(null)
    try {
      const payload = {
        ...formData,
        price: formData.price ? parseFloat(formData.price) : null,
      }
      const data = await predict(payload)
      setResult(data)
      if (data.ai_summary) {
        try {
          setInsights(JSON.parse(data.ai_summary))
        } catch {
          setInsights({ forecast_summary: data.ai_summary })
        }
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to generate forecast. Make sure the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  const forecastChart = result?.forecast_data ? {
    labels: result.forecast_data.dates.map((d) => {
      const parts = d.split('-')
      return `${parts[1]}/${parts[2]}`
    }),
    datasets: [
      {
        label: 'Predicted Demand',
        data: result.forecast_data.values,
        borderColor: '#3B82F6',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: 4,
        pointBackgroundColor: '#3B82F6',
      },
    ],
  } : null

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx) => `${ctx.parsed.y.toFixed(1)} units`,
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        title: { display: true, text: 'Units' },
      },
      x: {
        title: { display: true, text: 'Day' },
      },
    },
  }

  return (
    <div className="space-y-6">
      {/* Input Form */}
      <div className="bg-white rounded-xl shadow-sm border p-6">
        <h2 className="text-lg font-bold mb-4">Forecast Configuration</h2>
        {dataLoading ? (
          <div className="flex items-center gap-2 text-gray-500">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
            Loading products and stores...
          </div>
        ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Product ({products.length} available)</label>
            <select
              value={formData.item_id}
              onChange={(e) => setFormData({ ...formData, item_id: e.target.value })}
              className="w-full border rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 outline-none"
            >
              {products.map((p) => (
                <option key={p.item_id} value={p.item_id}>
                  {p.item_id}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Store ({stores.length} available)</label>
            <select
              value={formData.store_id}
              onChange={(e) => setFormData({ ...formData, store_id: e.target.value })}
              className="w-full border rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 outline-none"
            >
              {stores.map((s) => (
                <option key={s.store_id} value={s.store_id}>
                  {s.store_id}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Forecast Period</label>
            <select
              value={formData.forecast_days}
              onChange={(e) => setFormData({ ...formData, forecast_days: parseInt(e.target.value) })}
              className="w-full border rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 outline-none"
            >
              <option value={7}>7 Days</option>
              <option value={14}>14 Days</option>
              <option value={30}>30 Days</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Current Inventory</label>
            <input
              type="number"
              value={formData.current_inventory}
              onChange={(e) => setFormData({ ...formData, current_inventory: parseInt(e.target.value) || 0 })}
              className="w-full border rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 outline-none"
              min="0"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Price ($)</label>
            <input
              type="number"
              value={formData.price}
              onChange={(e) => setFormData({ ...formData, price: e.target.value })}
              className="w-full border rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 outline-none"
              placeholder="Auto-detect"
              step="0.01"
              min="0"
            />
          </div>

          <div className="flex gap-4 items-end">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={formData.promotion}
                onChange={(e) => setFormData({ ...formData, promotion: e.target.checked })}
                className="w-4 h-4 text-blue-600 rounded"
              />
              <span className="text-sm">Promotion</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={formData.holiday}
                onChange={(e) => setFormData({ ...formData, holiday: e.target.checked })}
                className="w-4 h-4 text-blue-600 rounded"
              />
              <span className="text-sm">Holiday</span>
            </label>
          </div>
        </div>
        )}

        <button
          onClick={handlePredict}
          disabled={loading}
          className="mt-4 bg-blue-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <span className="animate-spin">⏳</span> Predicting...
            </span>
          ) : (
            '🔮 Predict Demand'
          )}
        </button>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-6">
          {/* Key Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white rounded-xl shadow-sm border p-4 text-center">
              <p className="text-sm text-gray-500">Predicted Demand</p>
              <p className="text-2xl font-bold text-blue-600">{result.predicted_demand?.toFixed(0) ?? '--'}</p>
              <p className="text-xs text-gray-400">units</p>
            </div>
            <div className="bg-white rounded-xl shadow-sm border p-4 text-center">
              <p className="text-sm text-gray-500">Confidence</p>
              <p className="text-2xl font-bold text-green-600">{result.confidence != null ? `${(result.confidence * 100).toFixed(0)}%` : '--'}</p>
              <p className="text-xs text-gray-400">score</p>
            </div>
            <div className="bg-white rounded-xl shadow-sm border p-4 text-center">
              <p className="text-sm text-gray-500">Expected Revenue</p>
              <p className="text-2xl font-bold text-purple-600">${result.revenue?.toFixed(2) ?? '--'}</p>
              <p className="text-xs text-gray-400">next {result.forecast_days} days</p>
            </div>
            <div className="bg-white rounded-xl shadow-sm border p-4 text-center">
              <p className="text-sm text-gray-500">Suggested Price</p>
              <p className="text-2xl font-bold text-orange-600">${result.suggested_price?.toFixed(2) ?? '--'}</p>
              <p className="text-xs text-gray-400">per unit</p>
            </div>
            <div className="bg-white rounded-xl shadow-sm border p-4 text-center">
              <p className="text-sm text-gray-500">Model</p>
              <p className="text-lg font-bold text-indigo-600">{result.model_used === 'tft' ? 'TFT' : result.model_used === 'mock' ? 'Fallback' : 'Statistical'}</p>
              <p className="text-xs text-gray-400">prediction engine</p>
            </div>
          </div>

          {/* Chart & SHAP */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 bg-white rounded-xl shadow-sm border p-6">
              <h3 className="font-bold text-lg mb-4">Demand Forecast</h3>
              <div className="h-64">
                {forecastChart && <Line data={forecastChart} options={chartOptions} />}
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-sm border p-6">
              <h3 className="font-bold text-lg mb-4">Feature Importance (SHAP)</h3>
              <div className="space-y-3">
                {result.shap_values && Object.entries(result.shap_values)
                  .sort((a, b) => b[1] - a[1])
                  .map(([name, value]) => (
                    <SHAPBar key={name} name={name} value={value} />
                  ))
                }
              </div>
            </div>
          </div>

          {/* Inventory & Pricing */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white rounded-xl shadow-sm border p-6">
              <h3 className="font-bold text-lg mb-4">📦 Inventory Recommendation</h3>
              <div className="space-y-3">
                <div className="flex justify-between p-3 bg-gray-50 rounded-lg">
                  <span className="text-gray-600">Current Stock</span>
                  <span className="font-semibold">{result.current_inventory ?? '--'} units</span>
                </div>
                <div className="flex justify-between p-3 bg-gray-50 rounded-lg">
                  <span className="text-gray-600">Safety Stock</span>
                  <span className="font-semibold">{result.safety_stock ?? '--'} units</span>
                </div>
                <div className="flex justify-between p-3 bg-gray-50 rounded-lg">
                  <span className="text-gray-600">Days Until Stockout</span>
                  <span className={`font-semibold ${(result.days_until_stockout ?? 999) < 7 ? 'text-red-600' : 'text-green-600'}`}>
                    {result.days_until_stockout ?? '--'} days
                  </span>
                </div>
                <div className="flex justify-between p-3 bg-blue-50 rounded-lg">
                  <span className="text-blue-700 font-medium">Suggested Reorder</span>
                  <span className="font-bold text-blue-700">{result.reorder_quantity ?? '--'} units</span>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-sm border p-6">
              <h3 className="font-bold text-lg mb-4">💰 Pricing Recommendation</h3>
              <div className="space-y-3">
                <div className="flex justify-between p-3 bg-gray-50 rounded-lg">
                  <span className="text-gray-600">Recommended Price</span>
                  <span className="font-semibold">${result.suggested_price?.toFixed(2) ?? '--'}</span>
                </div>
                <div className="flex justify-between p-3 bg-gray-50 rounded-lg">
                  <span className="text-gray-600">Unit Cost</span>
                  <span className="font-semibold">${result.cost?.toFixed(2) ?? '--'}</span>
                </div>
                <div className="flex justify-between p-3 bg-gray-50 rounded-lg">
                  <span className="text-gray-600">Expected Revenue</span>
                  <span className="font-semibold">${result.revenue?.toFixed(2) ?? '--'}</span>
                </div>
                <div className="flex justify-between p-3 bg-gray-50 rounded-lg">
                  <span className="text-gray-600">Price Elasticity</span>
                  <span className="font-semibold">{result.price_elasticity ?? -1.2}</span>
                </div>
                <div className="flex justify-between p-3 bg-green-50 rounded-lg">
                  <span className="text-green-700 font-medium">Expected Profit</span>
                  <span className="font-bold text-green-700">
                    ${result.expected_profit?.toFixed(2) ?? '--'}
                  </span>
                </div>
                <div className="flex justify-between p-3 bg-blue-50 rounded-lg">
                  <span className="text-blue-700 font-medium">Pricing Model</span>
                  <span className="font-bold text-blue-700">
                    {result.pricing_model === 'ppo' ? 'PPO AI' : result.pricing_model === 'mock' ? 'Fallback' : 'Rule-Based'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* AI Insights */}
          {insights && (
            <div className="bg-white rounded-xl shadow-sm border p-6">
              <h3 className="font-bold text-lg mb-4">🧠 AI Business Insights</h3>

              {insights.forecast_summary && (
                <div className="mb-4 p-4 bg-blue-50 rounded-lg">
                  <h4 className="font-semibold text-blue-800 mb-1">Forecast Summary</h4>
                  <p className="text-sm text-blue-700">{insights.forecast_summary}</p>
                </div>
              )}

              {insights.business_recommendations && (
                <div className="mb-4">
                  <h4 className="font-semibold mb-2">Business Recommendations</h4>
                  <ul className="space-y-1">
                    {insights.business_recommendations.map((rec, i) => (
                      <li key={i} className="text-sm text-gray-700 flex items-start gap-2">
                        <span className="text-green-500 mt-0.5">✓</span>
                        {rec}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {insights.risk_analysis && (
                <div className="p-4 bg-amber-50 rounded-lg">
                  <h4 className="font-semibold text-amber-800 mb-1">Risk Analysis</h4>
                  <p className="text-sm text-amber-700">
                    Overall Risk: {insights.risk_analysis.overall_risk}
                  </p>
                  {insights.risk_analysis.risks && (
                    <ul className="mt-2 space-y-1">
                      {insights.risk_analysis.risks.map((risk, i) => (
                        <li key={i} className="text-xs text-amber-600">
                          • {risk.risk} ({risk.level}): {risk.mitigation}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
