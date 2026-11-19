
import {
  Search,
  Bell,
  ChevronDown,
  Command,
  Plus,
  Activity,
} from 'lucide-react'

function Header() {
  return (
    <header className="flex h-[76px] shrink-0 items-center justify-between border-b border-[#202b3b] bg-[#0c121d] px-7">

      <div className="flex items-center gap-3">
        <div>
          <p className="text-[10px] tracking-[1.5px] text-slate-500">
            GEOSPATIAL INVESTIGATION ENGINE
          </p>
          <h2 className="mt-1 text-sm font-semibold text-slate-200">
            Intelligence Workspace
          </h2>
        </div>
      </div>

      <div className="flex items-center gap-4">

        <div className="hidden items-center gap-2 rounded-lg border border-[#29364a] bg-[#111a28] px-3 py-2.5 lg:flex">
          <Search size={15} className="text-slate-500" />
          <input
            type="text"
            placeholder="Search investigations..."
            className="w-44 bg-transparent text-xs text-slate-200 outline-none placeholder:text-slate-500"
          />
          <span className="flex items-center gap-1 rounded border border-[#354258] px-1.5 py-0.5 text-[10px] text-slate-500">
            <Command size={10} /> K
          </span>
        </div>

        <div className="hidden items-center gap-2 rounded-lg border border-emerald-500/15 bg-emerald-500/5 px-3 py-2 md:flex">
          <Activity size={14} className="text-emerald-400" />
          <span className="text-[11px] text-emerald-400">
            Workspace ready
          </span>
        </div>

        <button
          type="button"
          aria-label="Create investigation"
          className="hidden items-center gap-2 rounded-lg bg-cyan-500 px-3.5 py-2.5 text-xs font-semibold text-slate-950 transition hover:bg-cyan-400 sm:flex"
        >
          <Plus size={15} />
          New Investigation
        </button>

        <button
          type="button"
          aria-label="Notifications"
          className="relative rounded-lg p-2.5 text-slate-400 transition hover:bg-slate-800 hover:text-white"
        >
          <Bell size={18} />
          <span className="absolute right-2 top-2 h-1.5 w-1.5 rounded-full bg-cyan-400" />
        </button>

        <div className="flex items-center gap-2 border-l border-[#29364a] pl-4">
          <div className="flex h-9 w-9 items-center justify-center rounded-full border border-cyan-500/20 bg-cyan-500/10 text-sm font-semibold text-cyan-400">
            B
          </div>

          <div className="hidden xl:block">
            <p className="text-xs font-medium text-slate-200">
              Investigator
            </p>
            <p className="text-[10px] text-slate-500">
              Administrator
            </p>
          </div>

          <ChevronDown size={14} className="text-slate-500" />
        </div>
      </div>
    </header>
  )
}

export default Header
