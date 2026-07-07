import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { getAnalytics, getHistory } from '../api'

const StatCard = ({ title, value, icon, color, subtitle }) => (
  <div className="bg-white rounded-xl shadow-sm border p-6 hover:shadow-md transition-shadow">
    <div className="flex items-start justify-between">
      <div>
        <p className="text-sm font-medium text-gray-500">{title}</p>
        <p className={`text-2xl font-bold mt-1 ${color}`}>{value}</p>
        {subtitle && <p className="text-xs text-gray-400 mt-1">{subtitle}</p>}
      </div>
      <span className="text-3xl">{icon}</span>
    </div>
  </div>
)

const RecentForecast = ({ forecast }) => (
  <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
    <div className="flex items-center gap-3">
      <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
        <span className="text-blue-600 font-bold text-sm">📦</span>
      </div>
      <div>
        <p className="font-medium text-sm">{forecast.item_id}</p>
        <p className="text-xs text-gray-500">{forecast.store_id}</p>
      </div>
    </div>
    <div className="text-right">
      <p className="font-semibold text-sm">{forecast.predicted_demand?.toFixed(0)} units</p>
      <p className="text-xs text-gray-500">{(forecast.confidence * 100).toFixed(0)}% conf.</p>
    </div>
  </div>
)

export default function Dashboard() {
  const navigate = useNavigate()
  const [analytics, setAnalytics] = useState(null)
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    try {
      const [analyticsData, historyData] = await Promise.all([
        getAnalytics(),
        getHistory(5),
      ])
      setAnalytics(analyticsData)
      setHistory(historyData)
    } catch (err) {
      console.error('Failed to load dashboard data:', err)
      setAnalytics({
        total_products: 50,
        total_stores: 50,
        avg_daily_sales: 45.2,
        total_forecasts: 0,
        latest_forecast: null,
        inventory_health: { healthy: 0, warning: 0, critical: 0 },
      })
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl p-8 text-white">
        <h2 className="text-3xl font-bold mb-2">AI Retail Demand Forecaster</h2>
        <p className="text-blue-100 mb-6 max-w-2xl">
          Predict demand, optimize inventory, and maximize revenue with AI-powered analytics.
          Powered by Temporal Fusion Transformer and Reinforcement Learning.
        </p>
        <button
          onClick={() => navigate('/forecast')}
          className="bg-white text-blue-600 px-6 py-3 rounded-lg font-semibold hover:bg-blue-50 transition-colors"
        >
          Start Forecasting →
        </button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Products"
          value={analytics.total_products}
          icon="📦"
          color="text-blue-600"
          subtitle="Unique products in dataset"
        />
        <StatCard
          title="Available Stores"
          value={analytics.total_stores}
          icon="🏪"
          color="text-green-600"
          subtitle="Retail locations"
        />
        <StatCard
          title="Avg Daily Sales"
          value={analytics.avg_daily_sales.toFixed(1)}
          icon="📊"
          color="text-purple-600"
          subtitle="Units per product"
        />
        <StatCard
          title="Total Forecasts"
          value={analytics.total_forecasts}
          icon="🔮"
          color="text-orange-600"
          subtitle="Predictions made"
        />
      </div>

      {/* Recent Forecasts & Inventory Health */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl shadow-sm border p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-bold text-lg">Recent Forecasts</h3>
            <button
              onClick={() => navigate('/forecast')}
              className="text-sm text-blue-600 hover:text-blue-800"
            >
              View All →
            </button>
          </div>
          <div className="space-y-2">
            {history.length > 0 ? (
              history.map((f, i) => <RecentForecast key={i} forecast={f} />)
            ) : (
              <div className="text-center py-8 text-gray-400">
                <p className="text-4xl mb-2">📊</p>
                <p>No forecasts yet. Start by predicting demand!</p>
              </div>
            )}
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border p-6">
          <h3 className="font-bold text-lg mb-4">Inventory Health</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-green-50 rounded-lg">
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                <span className="font-medium">Healthy Stock</span>
              </div>
              <span className="font-bold text-green-600">
                {analytics.inventory_health.healthy}
              </span>
            </div>
            <div className="flex items-center justify-between p-4 bg-yellow-50 rounded-lg">
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                <span className="font-medium">Low Stock Warning</span>
              </div>
              <span className="font-bold text-yellow-600">
                {analytics.inventory_health.warning}
              </span>
            </div>
            <div className="flex items-center justify-between p-4 bg-red-50 rounded-lg">
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                <span className="font-medium">Critical Stockout</span>
              </div>
              <span className="font-bold text-red-600">
                {analytics.inventory_health.critical}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Features Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-xl shadow-sm border p-6">
          <div className="text-3xl mb-3">🤖</div>
          <h3 className="font-bold mb-2">Temporal Fusion Transformer</h3>
          <p className="text-sm text-gray-600">
            State-of-the-art deep learning model for multi-horizon time series forecasting.
          </p>
        </div>
        <div className="bg-white rounded-xl shadow-sm border p-6">
          <div className="text-3xl mb-3">💰</div>
          <h3 className="font-bold mb-2">Dynamic Pricing (PPO)</h3>
          <p className="text-sm text-gray-600">
            Reinforcement learning agent that optimizes pricing for maximum revenue.
          </p>
        </div>
        <div className="bg-white rounded-xl shadow-sm border p-6">
          <div className="text-3xl mb-3">🧠</div>
          <h3 className="font-bold mb-2">Gemini AI Insights</h3>
          <p className="text-sm text-gray-600">
            Natural language business recommendations powered by Google Gemini.
          </p>
        </div>
      </div>
    </div>
  )
}
