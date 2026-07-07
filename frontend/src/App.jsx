import { useState } from 'react'
import { Routes, Route, Link, useLocation } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Forecast from './pages/Forecast'
import Analytics from './pages/Analytics'
import Inventory from './pages/Inventory'
import Reports from './pages/Reports'

const navItems = [
  { path: '/', label: 'Home', icon: '🏠' },
  { path: '/forecast', label: 'Forecast', icon: '📊' },
  { path: '/analytics', label: 'Analytics', icon: '📈' },
  { path: '/inventory', label: 'Inventory', icon: '📦' },
  { path: '/reports', label: 'Reports', icon: '📄' },
]

function App() {
  const location = useLocation()
  const [sidebarOpen, setSidebarOpen] = useState(true)

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <aside className={`${sidebarOpen ? 'w-64' : 'w-16'} bg-gray-900 text-white transition-all duration-300 flex flex-col`}>
        <div className="p-4 border-b border-gray-700">
          <div className="flex items-center gap-2">
            <span className="text-2xl">🛒</span>
            {sidebarOpen && (
              <span className="font-bold text-lg">Retail Forecaster</span>
            )}
          </div>
        </div>

        <nav className="flex-1 p-2">
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg mb-1 transition-colors ${
                location.pathname === item.path
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-300 hover:bg-gray-800 hover:text-white'
              }`}
            >
              <span className="text-xl">{item.icon}</span>
              {sidebarOpen && <span>{item.label}</span>}
            </Link>
          ))}
        </nav>

        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="p-4 border-t border-gray-700 hover:bg-gray-800 transition-colors"
        >
          {sidebarOpen ? '◀' : '▶'}
        </button>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <header className="bg-white border-b px-6 py-4 flex items-center justify-between sticky top-0 z-10">
          <div>
            <h1 className="text-xl font-bold text-gray-800">
              {navItems.find((n) => n.path === location.pathname)?.label || 'Dashboard'}
            </h1>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-sm text-gray-500">
              AI-Powered Retail Analytics
            </div>
            <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white text-sm font-bold">
              AI
            </div>
          </div>
        </header>

        <div className="p-6">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/forecast" element={<Forecast />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/inventory" element={<Inventory />} />
            <Route path="/reports" element={<Reports />} />
          </Routes>
        </div>
      </main>
    </div>
  )
}

export default App
