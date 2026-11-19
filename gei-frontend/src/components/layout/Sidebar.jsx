
import { NavLink } from 'react-router-dom'
import {
  Globe2,
  LayoutDashboard,
  Radar,
  Map,
  Layers3,
  ScanSearch,
  FileText,
  Database,
  Settings,
  ChevronRight,
  CircleHelp,
} from 'lucide-react'

const sections = [
  {
    title: 'WORKSPACE',
    items: [
      { name: 'Overview', path: '/', icon: LayoutDashboard },
      { name: 'Investigations', path: '/investigations', icon: Radar },
      { name: 'Geospatial Map', path: '/map', icon: Map },
    ],
  },
  {
    title: 'ANALYSIS',
    items: [
      { name: 'Change Detection', path: '/detection', icon: ScanSearch },
      { name: 'Analysis Layers', path: '/layers', icon: Layers3 },
      { name: 'Reports', path: '/reports', icon: FileText },
    ],
  },
  {
    title: 'SYSTEM',
    items: [
      { name: 'Data Sources', path: '/data-sources', icon: Database },
      { name: 'Settings', path: '/settings', icon: Settings },
    ],
  },
]

function Sidebar() {
  return (
    <aside className="flex h-screen w-[250px] shrink-0 flex-col border-r border-[#202b3b] bg-[#0c121d]">

      <div className="flex h-[76px] items-center gap-3 border-b border-[#202b3b] px-5">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-cyan-400/20 bg-cyan-400/10">
          <Globe2 size={23} className="text-cyan-400" />
        </div>

        <div>
          <h1 className="text-lg font-bold tracking-[2px] text-white">
            GEI
          </h1>
          <p className="text-[9px] tracking-[1.8px] text-slate-500">
            INTELLIGENCE ENGINE
          </p>
        </div>
      </div>

      <div className="border-b border-[#202b3b] px-4 py-4">
        <button className="flex w-full items-center justify-between rounded-lg border border-[#29364a] bg-[#131d2b] px-3 py-2.5 text-left transition hover:border-cyan-500/30">
          <div>
            <p className="text-[10px] text-slate-500">ACTIVE WORKSPACE</p>
            <p className="mt-1 text-xs font-medium text-slate-200">
              Default Project
            </p>
          </div>
          <ChevronRight size={15} className="text-slate-500" />
        </button>
      </div>

      <nav className="flex-1 space-y-7 overflow-y-auto px-3 py-6">
        {sections.map((section) => (
          <div key={section.title}>
            <p className="mb-3 px-3 text-[10px] font-semibold tracking-[1.5px] text-slate-500">
              {section.title}
            </p>

            <div className="space-y-1">
              {section.items.map((item) => {
                const Icon = item.icon

                return (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    end={item.path === '/'}
                    className={({ isActive }) =>
                      `group flex items-center gap-3 rounded-lg border px-3 py-2.5 text-[13px] transition ${
                        isActive
                          ? 'border-cyan-500/20 bg-cyan-500/10 text-cyan-400'
                          : 'border-transparent text-slate-400 hover:bg-[#172232] hover:text-slate-100'
                      }`
                    }
                  >
                    <Icon size={17} />
                    <span>{item.name}</span>
                  </NavLink>
                )
              })}
            </div>
          </div>
        ))}
      </nav>

      <div className="space-y-3 border-t border-[#202b3b] p-4">
        <div className="rounded-lg border border-emerald-500/15 bg-emerald-500/5 p-3">
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            <span className="text-xs font-medium text-emerald-400">
              Engine Configured
            </span>
          </div>
          <p className="mt-2 text-[10px] leading-4 text-slate-500">
            Backend connection will be verified during integration.
          </p>
        </div>

        <button className="flex w-full items-center gap-3 px-3 py-2 text-xs text-slate-500 transition hover:text-white">
          <CircleHelp size={16} />
          Help and documentation
        </button>
      </div>
    </aside>
  )
}

export default Sidebar
