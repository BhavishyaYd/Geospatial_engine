
import {
  Radar,
  ScanSearch,
  Layers3,
  Clock3,
  BrainCircuit,
  Image,
  Activity,
  ArrowUpRight,
  CircleDot,
  FileText,
  Database,
  ChevronRight,
  Clock,
} from 'lucide-react'

import MapPanel from '../components/dashboard/MapPanel'

const metrics = [
  {
    title: 'Total Investigations',
    value: '—',
    detail: 'Awaiting backend data',
    icon: Radar,
  },
  {
    title: 'Detected Regions',
    value: '—',
    detail: 'Awaiting analysis',
    icon: ScanSearch,
  },
  {
    title: 'Area Analysed',
    value: '—',
    detail: 'No AOI loaded',
    icon: Layers3,
  },
  {
    title: 'Processing Time',
    value: '—',
    detail: 'No pipeline run yet',
    icon: Clock3,
  },
]

const models = [
  {
    name: 'CSD-Net',
    description: 'Temporal change detection',
    detail: 'Change masks and pixel-level evidence',
    icon: ScanSearch,
  },
  {
    name: 'RemoteCLIP',
    description: 'Semantic image analysis',
    detail: 'Image embeddings and semantic comparison',
    icon: BrainCircuit,
  },
  {
    name: 'Florence-2',
    description: 'Visual region descriptions',
    detail: 'Descriptions of extracted regions',
    icon: Image,
  },
]

function SectionHeading({ title, subtitle, action }) {
  return (
    <div className="mb-4 flex items-center justify-between gap-3">
      <div>
        <h3 className="text-sm font-semibold text-slate-100">
          {title}
        </h3>
        {subtitle && (
          <p className="mt-1 text-[11px] text-slate-500">
            {subtitle}
          </p>
        )}
      </div>

      {action}
    </div>
  )
}

function Dashboard() {
  return (
    <div className="min-h-full space-y-6 p-5 md:p-7">

      {/* PAGE INTRO */}
      <section className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="mb-2 flex items-center gap-2 text-[10px] font-semibold tracking-[1.8px] text-cyan-400">
            <CircleDot size={12} />
            GEOSPATIAL INTELLIGENCE / OVERVIEW
          </div>

          <h1 className="text-2xl font-semibold tracking-tight text-white">
            Intelligence Overview
          </h1>

          <p className="mt-2 max-w-2xl text-xs leading-5 text-slate-400">
            Monitor geographical investigations, analyse temporal imagery
            and review AI-generated change evidence.
          </p>
        </div>

        <div className="flex items-center gap-2 rounded-lg border border-amber-500/20 bg-amber-500/5 px-3 py-2">
          <span className="h-2 w-2 rounded-full bg-amber-400" />
          <span className="text-[10px] font-medium text-amber-300">
            BACKEND INTEGRATION PENDING
          </span>
        </div>
      </section>

      {/* METRICS */}
      <section className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {metrics.map((metric) => {
          const Icon = metric.icon

          return (
            <div
              key={metric.title}
              className="rounded-xl border border-[#263448] bg-[#101925] p-4 transition hover:border-cyan-500/30"
            >
              <div className="flex items-center justify-between">
                <p className="text-xs text-slate-400">
                  {metric.title}
                </p>

                <div className="rounded-lg bg-cyan-500/10 p-2">
                  <Icon size={17} className="text-cyan-400" />
                </div>
              </div>

              <h2 className="mt-4 text-2xl font-semibold tabular-nums text-white">
                {metric.value}
              </h2>

              <p className="mt-2 text-[10px] text-slate-500">
                {metric.detail}
              </p>
            </div>
          )
        })}
      </section>

      {/* MAIN INVESTIGATION WORKSPACE */}
      <section className="grid grid-cols-1 gap-5 2xl:grid-cols-[minmax(0,1fr)_310px]">

        <div className="min-w-0">
          <MapPanel />
        </div>

        {/* AI INTELLIGENCE PANEL */}
        <aside className="space-y-4">

          <div className="rounded-xl border border-[#263448] bg-[#101925] p-4">
            <SectionHeading
              title="AI Intelligence Pipeline"
              subtitle="Analysis modules used by GEI"
            />

            <div className="space-y-3">
              {models.map((model, index) => {
                const Icon = model.icon

                return (
                  <div
                    key={model.name}
                    className="rounded-lg border border-[#29374a] bg-[#131f2e] p-3"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="rounded-md bg-cyan-500/10 p-2">
                          <Icon size={15} className="text-cyan-400" />
                        </div>

                        <div>
                          <p className="text-xs font-semibold text-slate-200">
                            {model.name}
                          </p>
                          <p className="mt-0.5 text-[10px] text-slate-500">
                            {model.description}
                          </p>
                        </div>
                      </div>

                      <span className="rounded-md border border-amber-500/20 bg-amber-500/5 px-2 py-1 text-[9px] text-amber-300">
                        NOT RUNNING
                      </span>
                    </div>

                    <p className="mt-3 text-[10px] leading-4 text-slate-500">
                      {model.detail}
                    </p>
                  </div>
                )
              })}
            </div>
          </div>

          <div className="rounded-xl border border-[#263448] bg-[#101925] p-4">
            <SectionHeading
              title="Investigation Controls"
              subtitle="Start a new analysis"
            />

            <div className="space-y-3">
              <div className="rounded-lg border border-dashed border-[#35465d] bg-[#131f2e] p-4 text-center">
                <div className="mx-auto flex h-10 w-10 items-center justify-center rounded-lg bg-cyan-500/10">
                  <Database size={19} className="text-cyan-400" />
                </div>

                <p className="mt-3 text-xs font-medium text-slate-200">
                  No imagery selected
                </p>

                <p className="mt-1 text-[10px] leading-4 text-slate-500">
                  Select a temporal image pair to begin an investigation.
                </p>
              </div>

              <button
                type="button"
                disabled
                className="flex w-full items-center justify-center gap-2 rounded-lg border border-[#344256] bg-[#1b293a] px-3 py-2.5 text-xs font-medium text-slate-400 opacity-70"
              >
                <ScanSearch size={15} />
                Start Investigation
                <ArrowUpRight size={14} />
              </button>

              <p className="text-center text-[9px] text-slate-600">
                Upload workflow will be enabled in a later milestone.
              </p>
            </div>
          </div>
        </aside>
      </section>

      {/* LOWER DASHBOARD */}
      <section className="grid grid-cols-1 gap-5 xl:grid-cols-2">

        <div className="rounded-xl border border-[#263448] bg-[#101925] p-4">
          <SectionHeading
            title="Recent Investigations"
            subtitle="Previously executed analysis runs"
            action={
              <button
                type="button"
                className="flex items-center gap-1 text-[10px] text-cyan-400 hover:text-cyan-300"
              >
                View all <ChevronRight size={13} />
              </button>
            }
          />

          <div className="flex min-h-[150px] flex-col items-center justify-center text-center">
            <div className="rounded-xl bg-[#172334] p-3">
              <Radar size={21} className="text-slate-500" />
            </div>

            <p className="mt-3 text-xs font-medium text-slate-300">
              No investigation records yet
            </p>

            <p className="mt-1 max-w-xs text-[10px] leading-4 text-slate-500">
              Completed pipeline runs will appear here when the backend is connected.
            </p>
          </div>
        </div>

        <div className="rounded-xl border border-[#263448] bg-[#101925] p-4">
          <SectionHeading
            title="System Activity"
            subtitle="Application and processing status"
          />

          <div className="space-y-4">
            <div className="flex items-start gap-3">
              <div className="rounded-lg bg-cyan-500/10 p-2">
                <Activity size={15} className="text-cyan-400" />
              </div>

              <div>
                <p className="text-xs font-medium text-slate-200">
                  Frontend workspace initialized
                </p>
                <p className="mt-1 text-[10px] text-slate-500">
                  Dashboard interface is available.
                </p>
              </div>

              <span className="ml-auto text-[9px] text-slate-600">
                CURRENT
              </span>
            </div>

            <div className="flex items-start gap-3">
              <div className="rounded-lg bg-amber-500/10 p-2">
                <Database size={15} className="text-amber-400" />
              </div>

              <div>
                <p className="text-xs font-medium text-slate-200">
                  Backend connection
                </p>
                <p className="mt-1 text-[10px] text-slate-500">
                  API integration has not been configured.
                </p>
              </div>

              <span className="ml-auto text-[9px] text-amber-400">
                PENDING
              </span>
            </div>

            <div className="flex items-start gap-3">
              <div className="rounded-lg bg-slate-700/30 p-2">
                <Clock size={15} className="text-slate-400" />
              </div>

              <div>
                <p className="text-xs font-medium text-slate-200">
                  Waiting for first analysis
                </p>
                <p className="mt-1 text-[10px] text-slate-500">
                  No pipeline execution has been initiated from this dashboard.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="flex flex-wrap items-center justify-between gap-2 border-t border-[#202b3b] pt-4 text-[10px] text-slate-600">
        <p>GEI · Geospatial Investigation Engine</p>
        <p>Intelligence Workspace · Development Build</p>
      </footer>
    </div>
  )
}

export default Dashboard
