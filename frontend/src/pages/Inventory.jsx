import { useState, useEffect } from 'react'
import { getHistory } from '../api'

const StockBar = ({ current, safety, max }) => {
  const percentage = Math.min((current / max) * 100, 100)
  const isLow = current < safety * 1.5
  const isCritical = current < safety

  return (
    <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
      <div
        className={`h-full rounded-full transition-all duration-500 ${
          isCritical ? 'bg-red-500' : isLow ? 'bg-yellow-500' : 'bg-green-500'
        }`}
        style={{ width: `${percentage}%` }}
      />
    </div>
  )
}

const InventoryItem = ({ item }) => {
  const safetyStock = item.safety_stock || 20
  const currentStock = item.current_inventory || 100
  const expectedSales = item.predicted_demand || 0
  const daysUntilStockout = item.days_until_stockout || 999
  const reorderQty = item.reorder_quantity || 0

  const maxStock = safetyStock * 5
  const isCritical = daysUntilStockout < 7
  const isWarning = daysUntilStockout < 14

  return (
    <div className={`bg-white rounded-xl shadow-sm border p-5 ${
      isCritical ? 'border-red-200 bg-red-50' :
      isWarning ? 'border-yellow-200 bg-yellow-50' : ''
    }`}>
      <div className="flex items-start justify-between mb-3">
        <div>
          <h4 className="font-bold text-sm">{item.item_id}</h4>
          <p className="text-xs text-gray-500">{item.store_id}</p>
        </div>
        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
          isCritical ? 'bg-red-100 text-red-700' :
          isWarning ? 'bg-yellow-100 text-yellow-700' :
          'bg-green-100 text-green-700'
        }`}>
          {isCritical ? 'Critical' : isWarning ? 'Warning' : 'Healthy'}
        </span>
      </div>

      <StockBar current={currentStock} safety={safetyStock} max={maxStock} />

      <div className="grid grid-cols-2 gap-2 mt-3">
        <div className="bg-gray-50 rounded-lg p-2">
          <p className="text-xs text-gray-500">Current Stock</p>
          <p className="font-bold text-sm">{currentStock} units</p>
        </div>
        <div className="bg-gray-50 rounded-lg p-2">
          <p className="text-xs text-gray-500">Safety Stock</p>
          <p className="font-bold text-sm">{safetyStock} units</p>
        </div>
        <div className="bg-gray-50 rounded-lg p-2">
          <p className="text-xs text-gray-500">Expected Sales</p>
          <p className="font-bold text-sm">{expectedSales.toFixed(0)} units</p>
        </div>
        <div className="bg-gray-50 rounded-lg p-2">
          <p className="text-xs text-gray-500">Days to Stockout</p>
          <p className={`font-bold text-sm ${isCritical ? 'text-red-600' : ''}`}>
            {daysUntilStockout} days
          </p>
        </div>
      </div>

      {reorderQty > 0 && (
        <div className="mt-3 p-2 bg-blue-50 rounded-lg">
          <p className="text-xs text-blue-700 font-medium">
            📦 Reorder: {reorderQty} units
          </p>
        </div>
      )}
    </div>
  )
}

export default function Inventory() {
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('all')

  useEffect(() => {
    loadInventory()
  }, [])

  const loadInventory = async () => {
    try {
      const history = await getHistory(50)
      const inventoryMap = new Map()

      history.forEach((h) => {
        const key = `${h.item_id}_${h.store_id}`
        if (!inventoryMap.has(key)) {
          const predicted = h.predicted_demand || 0
          const safetyStock = h.safety_stock ?? Math.floor(predicted * 0.3)
          const currentInventory = h.current_inventory ?? Math.max(safetyStock, Math.floor(predicted * 2))
          const daysUntilStockout = h.days_until_stockout ?? (currentInventory > 0 && predicted > 0 ? Math.floor(currentInventory / (predicted / 7)) : 999)
          const reorderQty = h.reorder_quantity ?? Math.max(0, Math.floor(predicted * 0.5))
          inventoryMap.set(key, {
            item_id: h.item_id,
            store_id: h.store_id,
            demo_name: h.demo_name,
            predicted_demand: h.predicted_demand,
            current_inventory: currentInventory,
            safety_stock: safetyStock,
            days_until_stockout: daysUntilStockout,
            reorder_quantity: reorderQty,
          })
        }
      })

      setItems(Array.from(inventoryMap.values()))
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const filteredItems = items.filter((item) => {
    if (filter === 'critical') return item.days_until_stockout < 7
    if (filter === 'warning') return item.days_until_stockout >= 7 && item.days_until_stockout < 14
    if (filter === 'healthy') return item.days_until_stockout >= 14
    return true
  })

  const stats = {
    total: items.length,
    critical: items.filter((i) => i.days_until_stockout < 7).length,
    warning: items.filter((i) => i.days_until_stockout >= 7 && i.days_until_stockout < 14).length,
    healthy: items.filter((i) => i.days_until_stockout >= 14).length,
    totalStock: items.reduce((sum, i) => sum + i.current_inventory, 0),
    totalReorder: items.reduce((sum, i) => sum + i.reorder_quantity, 0),
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
      <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
        <div className="bg-white rounded-xl shadow-sm border p-4 text-center">
          <p className="text-xs text-gray-500">Total Items</p>
          <p className="text-xl font-bold">{stats.total}</p>
        </div>
        <div className="bg-red-50 rounded-xl border border-red-200 p-4 text-center">
          <p className="text-xs text-red-500">Critical</p>
          <p className="text-xl font-bold text-red-600">{stats.critical}</p>
        </div>
        <div className="bg-yellow-50 rounded-xl border border-yellow-200 p-4 text-center">
          <p className="text-xs text-yellow-500">Warning</p>
          <p className="text-xl font-bold text-yellow-600">{stats.warning}</p>
        </div>
        <div className="bg-green-50 rounded-xl border border-green-200 p-4 text-center">
          <p className="text-xs text-green-500">Healthy</p>
          <p className="text-xl font-bold text-green-600">{stats.healthy}</p>
        </div>
        <div className="bg-white rounded-xl shadow-sm border p-4 text-center">
          <p className="text-xs text-gray-500">Total Stock</p>
          <p className="text-xl font-bold">{stats.totalStock.toLocaleString()}</p>
        </div>
        <div className="bg-blue-50 rounded-xl border border-blue-200 p-4 text-center">
          <p className="text-xs text-blue-500">Need Reorder</p>
          <p className="text-xl font-bold text-blue-600">{stats.totalReorder.toLocaleString()}</p>
        </div>
      </div>

      {/* Filters */}
      <div className="flex gap-2">
        {[
          { key: 'all', label: 'All Items' },
          { key: 'critical', label: 'Critical' },
          { key: 'warning', label: 'Warning' },
          { key: 'healthy', label: 'Healthy' },
        ].map((f) => (
          <button
            key={f.key}
            onClick={() => setFilter(f.key)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              filter === f.key
                ? 'bg-blue-600 text-white'
                : 'bg-white border text-gray-600 hover:bg-gray-50'
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {/* Inventory Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredItems.map((item, i) => (
          <InventoryItem key={`${item.item_id}_${item.store_id}_${i}`} item={item} />
        ))}
      </div>

      {filteredItems.length === 0 && (
        <div className="text-center py-12 bg-white rounded-xl border">
          <p className="text-4xl mb-2">📦</p>
          <p className="text-gray-500">No items match the selected filter.</p>
        </div>
      )}
    </div>
  )
}
