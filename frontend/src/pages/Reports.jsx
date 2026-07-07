import { useState, useEffect } from 'react'
import { generateReport, getAllProducts, getStores } from '../api'

export default function Reports() {
  const [products, setProducts] = useState([])
  const [stores, setStores] = useState([])
  const [formData, setFormData] = useState({
    item_id: '',
    store_id: '',
    forecast_days: 7,
    format: 'pdf',
  })
  const [generating, setGenerating] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [dataLoading, setDataLoading] = useState(true)

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

  const handleGenerate = async () => {
    setGenerating(true)
    setError(null)
    setResult(null)
    try {
      const data = await generateReport(formData)
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to generate report.')
    } finally {
      setGenerating(false)
    }
  }

  const handleDownloadCSV = async () => {
    setGenerating(true)
    setError(null)
    try {
      const data = await generateReport({ ...formData, format: 'csv' })
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to generate CSV.')
    } finally {
      setGenerating(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Report Generator */}
      <div className="bg-white rounded-xl shadow-sm border p-6">
        <h2 className="text-lg font-bold mb-4">Generate Report</h2>
        {dataLoading ? (
          <div className="flex items-center gap-2 text-gray-500">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
            Loading products and stores...
          </div>
        ) : (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Product ({products.length})</label>
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
            <label className="block text-sm font-medium text-gray-700 mb-1">Store ({stores.length})</label>
            <select
              value={formData.store_id}
              onChange={(e) => setFormData({ ...formData, store_id: e.target.value })}
              className="w-full border rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 outline-none"
            >
              {stores.map((s) => (
                <option key={s.store_id} value={s.store_id}>{s.store_id}</option>
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
            <label className="block text-sm font-medium text-gray-700 mb-1">Format</label>
            <select
              value={formData.format}
              onChange={(e) => setFormData({ ...formData, format: e.target.value })}
              className="w-full border rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 outline-none"
            >
              <option value="pdf">PDF Report</option>
              <option value="csv">CSV Data</option>
            </select>
          </div>
        </div>
        )}

        <div className="flex gap-3 mt-4">
          <button
            onClick={handleGenerate}
            disabled={generating}
            className="bg-blue-600 text-white px-6 py-2.5 rounded-lg font-semibold hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {generating ? '⏳ Generating...' : '📄 Generate PDF'}
          </button>
          <button
            onClick={handleDownloadCSV}
            disabled={generating}
            className="bg-green-600 text-white px-6 py-2.5 rounded-lg font-semibold hover:bg-green-700 disabled:opacity-50 transition-colors"
          >
            {generating ? '⏳ Generating...' : '📊 Export CSV'}
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
          {error}
        </div>
      )}

      {result && (
        <div className="bg-green-50 border border-green-200 rounded-xl p-6">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center">
              <span className="text-2xl">✅</span>
            </div>
            <div>
              <h3 className="font-bold text-green-800">Report Generated Successfully</h3>
              <p className="text-sm text-green-600">
                Format: {result.format?.toUpperCase()} | File: {result.file_path}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Report Types Info */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl shadow-sm border p-6">
          <h3 className="font-bold text-lg mb-4">📄 PDF Report Includes</h3>
          <ul className="space-y-2">
            {[
              'Executive Summary',
              'Demand Forecast with Chart',
              'Key Metrics Table',
              'Business Recommendations',
              'Risk Analysis',
              'Inventory Recommendations',
              'Pricing Strategy',
            ].map((item, i) => (
              <li key={i} className="flex items-center gap-2 text-sm text-gray-700">
                <span className="text-green-500">✓</span>
                {item}
              </li>
            ))}
          </ul>
        </div>

        <div className="bg-white rounded-xl shadow-sm border p-6">
          <h3 className="font-bold text-lg mb-4">📊 CSV Export Includes</h3>
          <ul className="space-y-2">
            {[
              'Date-wise Forecast',
              'Predicted Demand',
              'Product & Store Info',
              'Confidence Scores',
              'Revenue Projections',
              'Raw Data for Analysis',
            ].map((item, i) => (
              <li key={i} className="flex items-center gap-2 text-sm text-gray-700">
                <span className="text-green-500">✓</span>
                {item}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  )
}
