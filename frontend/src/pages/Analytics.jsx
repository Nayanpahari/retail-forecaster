import { useState, useEffect } from 'react'
import { Line, Bar, Doughnut } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, ArcElement, Title, Tooltip, Legend, Filler,
} from 'chart.js'
import { getHistory, getAnalytics, getHeatmap } from '../api'

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, ArcElement, Title, Tooltip, Legend, Filler
)

export default function Analytics() {
  const [history, setHistory] = useState([])
  const [analytics, setAnalytics] = useState(null)
  const [heatmap, setHeatmap] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    try {
      const [hist, ana, heat] = await Promise.all([getHistory(50), getAnalytics(), getHeatmap()])
      setHistory(hist)
      setAnalytics(ana)
      setHeatmap(heat)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const salesTrendData = {
    labels: history.slice().reverse().map((h, i) => `#${i + 1}`),
    datasets: [
      {
        label: 'Predicted Demand',
        data: history.slice().reverse().map((h) => h.predicted_demand),
        borderColor: '#3B82F6',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        fill: true,
        tension: 0.4,
      },
    ],
  }

  const revenueData = {
    labels: history.slice().reverse().map((h, i) => `#${i + 1}`),
    datasets: [
      {
        label: 'Revenue ($)',
        data: history.slice().reverse().map((h) => h.revenue),
        backgroundColor: 'rgba(16, 185, 129, 0.7)',
        borderColor: '#10B981',
        borderWidth: 1,
      },
    ],
  }

  const confidenceData = {
    labels: history.slice().reverse().map((h, i) => `#${i + 1}`),
    datasets: [
      {
        label: 'Confidence (%)',
        data: history.slice().reverse().map((h) => (h.confidence * 100)),
        borderColor: '#8B5CF6',
        backgroundColor: 'rgba(139, 92, 246, 0.1)',
        fill: true,
        tension: 0.4,
      },
    ],
  }

  const inventoryHealthData = analytics ? {
    labels: ['Healthy', 'Warning', 'Critical'],
    datasets: [
      {
        data: [
          analytics.inventory_health.healthy,
          analytics.inventory_health.warning,
          analytics.inventory_health.critical,
        ],
        backgroundColor: ['#10B981', '#F59E0B', '#EF4444'],
        borderWidth: 0,
      },
    ],
  } : null

  const priceData = {
    labels: history.slice().reverse().map((h, i) => `#${i + 1}`),
    datasets: [
      {
        label: 'Suggested Price ($)',
        data: history.slice().reverse().map((h) => h.suggested_price),
        borderColor: '#F59E0B',
        backgroundColor: 'rgba(245, 158, 11, 0.1)',
        fill: true,
        tension: 0.4,
      },
    ],
  }

  const lineOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      y: { beginAtZero: true },
    },
  }

  const doughnutOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'bottom' },
    },
    cutout: '60%',
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
      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-xl shadow-sm border p-4 text-center">
          <p className="text-sm text-gray-500">Total Forecasts</p>
          <p className="text-2xl font-bold text-blue-600">{analytics?.total_forecasts || 0}</p>
        </div>
        <div className="bg-white rounded-xl shadow-sm border p-4 text-center">
          <p className="text-sm text-gray-500">Avg Demand</p>
          <p className="text-2xl font-bold text-green-600">
            {analytics?.avg_daily_sales?.toFixed(1) || 0}
          </p>
        </div>
        <div className="bg-white rounded-xl shadow-sm border p-4 text-center">
          <p className="text-sm text-gray-500">Products Tracked</p>
          <p className="text-2xl font-bold text-purple-600">{analytics?.total_products || 0}</p>
        </div>
        <div className="bg-white rounded-xl shadow-sm border p-4 text-center">
          <p className="text-sm text-gray-500">Stores Active</p>
          <p className="text-2xl font-bold text-orange-600">{analytics?.total_stores || 0}</p>
        </div>
      </div>

      {/* Charts Row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl shadow-sm border p-6">
          <h3 className="font-bold text-lg mb-4">Sales Trend</h3>
          <div className="h-64">
            <Line data={salesTrendData} options={lineOptions} />
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border p-6">
          <h3 className="font-bold text-lg mb-4">Revenue Trend</h3>
          <div className="h-64">
            <Bar data={revenueData} options={lineOptions} />
          </div>
        </div>
      </div>

      {/* Charts Row 2 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-white rounded-xl shadow-sm border p-6">
          <h3 className="font-bold text-lg mb-4">Confidence Scores</h3>
          <div className="h-64">
            <Line data={confidenceData} options={lineOptions} />
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border p-6">
          <h3 className="font-bold text-lg mb-4">Price Trends</h3>
          <div className="h-64">
            <Line data={priceData} options={lineOptions} />
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border p-6">
          <h3 className="font-bold text-lg mb-4">Inventory Health</h3>
          <div className="h-64">
            {inventoryHealthData && (
              <Doughnut data={inventoryHealthData} options={doughnutOptions} />
            )}
          </div>
        </div>
      </div>

      {/* Seasonality Heatmap */}
      <div className="bg-white rounded-xl shadow-sm border p-6">
        <h3 className="font-bold text-lg mb-4">Demand Heatmap (Day of Week)</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr>
                <th className="p-2 text-left">Day</th>
                {heatmap?.stores?.map((s) => (
                  <th key={s} className="p-2 text-center">{s}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {heatmap?.days?.map((day, i) => (
                <tr key={day}>
                  <td className="p-2 font-medium">{day}</td>
                  {heatmap?.stores?.map((s) => {
                    const val = heatmap.data?.[s]?.[i] ?? 0
                    const maxVal = Math.max(...(heatmap.stores || []).map(st => Math.max(...(heatmap.data?.[st] || [1]))))
                    const intensity = maxVal > 0 ? Math.min(val / maxVal, 1) : 0
                    return (
                      <td
                        key={s}
                        className="p-2 text-center text-xs font-medium"
                        style={{
                          backgroundColor: `rgba(59, 130, 246, ${intensity})`,
                          color: intensity > 0.5 ? 'white' : '#1f2937',
                        }}
                      >
                        {val.toFixed(0)}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Recent Forecasts Table */}
      <div className="bg-white rounded-xl shadow-sm border p-6">
        <h3 className="font-bold text-lg mb-4">Forecast History</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b">
                <th className="p-2 text-left">Product</th>
                <th className="p-2 text-left">Store</th>
                <th className="p-2 text-center">Days</th>
                <th className="p-2 text-center">Demand</th>
                <th className="p-2 text-center">Confidence</th>
                <th className="p-2 text-center">Revenue</th>
                <th className="p-2 text-center">Price</th>
              </tr>
            </thead>
            <tbody>
              {history.map((h) => (
                <tr key={h.prediction_id} className="border-b hover:bg-gray-50">
                  <td className="p-2">{h.demo_name || h.item_id}</td>
                  <td className="p-2">{h.store_id}</td>
                  <td className="p-2 text-center">{h.forecast_days}</td>
                  <td className="p-2 text-center font-medium">{h.predicted_demand.toFixed(0)}</td>
                  <td className="p-2 text-center">
                    <span className={`px-2 py-1 rounded-full text-xs ${
                      h.confidence > 0.8 ? 'bg-green-100 text-green-700' :
                      h.confidence > 0.6 ? 'bg-yellow-100 text-yellow-700' :
                      'bg-red-100 text-red-700'
                    }`}>
                      {(h.confidence * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td className="p-2 text-center">${h.revenue.toFixed(2)}</td>
                  <td className="p-2 text-center">${h.suggested_price.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
