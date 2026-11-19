
import { useLocation } from 'react-router-dom'
import { Construction } from 'lucide-react'

const pageNames = {
  '/investigations': 'Investigations',
  '/map': 'Map View',
  '/reports': 'Reports',
  '/settings': 'Settings',
}

function PlaceholderPage() {
  const location = useLocation()
  const pageName = pageNames[location.pathname] || 'Page'

  return (
    <div className="flex min-h-[70vh] flex-col items-center justify-center px-6 text-center">
      <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
        <Construction size={32} className="text-cyan-400" />
      </div>

      <h1 className="mt-5 text-2xl font-semibold text-white">
        {pageName}
      </h1>

      <p className="mt-2 max-w-md text-sm text-slate-400">
        This workspace is part of the GEI development roadmap and will be implemented in the upcoming stages.
      </p>
    </div>
  )
}

export default PlaceholderPage
