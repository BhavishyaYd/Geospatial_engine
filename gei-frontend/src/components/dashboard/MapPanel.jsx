
import { useEffect, useRef, useState } from 'react'

import * as maplibregl from 'maplibre-gl'
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url'

import { Geoman } from '@geoman-io/maplibre-geoman-free'

import {
  area,
  bbox,
  booleanValid,
} from '@turf/turf'

import {
  Maximize2,
  Crosshair,
  Plus,
  Minus,
  MapPin,
  Download,
  Pencil,
  Activity,
  CheckCircle2,
  AlertTriangle,
} from 'lucide-react'

maplibregl.setWorkerUrl(workerUrl)

// Calculate AOI metadata from GeoJSON.

function calculateAOIStats(geojson) {
  const features = (geojson?.features || []).filter(
    (feature) =>
      feature.geometry &&
      ['Polygon', 'MultiPolygon'].includes(feature.geometry.type)
  )

  if (features.length === 0) {
    return {
      count: 0,
      areaKm2: 0,
      bounds: null,
      valid: false,
      invalidCount: 0,
    }
  }

  let totalArea = 0
  let invalidCount = 0

  features.forEach((feature) => {
    try {
      const isValid = booleanValid(feature)

      if (!isValid) {
        invalidCount++
      }

      totalArea += area(feature)
    } catch (error) {
      console.error('AOI validation error:', error)
      invalidCount++
    }
  })

  let bounds = null

  try {
    bounds = bbox({
      type: 'FeatureCollection',
      features,
    })
  } catch (error) {
    console.error('AOI bounds calculation error:', error)
  }

  return {
    count: features.length,
    areaKm2: totalArea / 1000000,
    bounds,
    valid: invalidCount === 0,
    invalidCount,
  }
}


function MapPanel() {
  const mapContainer = useRef(null)
  const mapInstance = useRef(null)
  const geomanInstance = useRef(null)

  const [mapStatus, setMapStatus] = useState('Loading map...')
  const [mapReady, setMapReady] = useState(false)
  const [activeLayer, setActiveLayer] = useState('satellite')

  const [aoiGeoJSON, setAoiGeoJSON] = useState(null)

  const [aoiStats, setAoiStats] = useState({
    count: 0,
    areaKm2: 0,
    bounds: null,
    valid: false,
    invalidCount: 0,
  })

  // Initialize MapLibre and Geoman.
  useEffect(() => {
    if (!mapContainer.current || mapInstance.current) return

    let cancelled = false

    const map = new maplibregl.Map({
      container: mapContainer.current,

      style: {
        version: 8,

        sources: {
          satellite: {
            type: 'raster',
            tiles: [
              'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            ],
            tileSize: 256,
            attribution: 'Tiles © Esri',
          },

          streets: {
            type: 'raster',
            tiles: [
              'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
            ],
            tileSize: 256,
            attribution: '© OpenStreetMap contributors',
          },
        },

        layers: [
          {
            id: 'satellite-layer',
            type: 'raster',
            source: 'satellite',
            layout: {
              visibility: 'visible',
            },
            minzoom: 0,
            maxzoom: 19,
          },

          {
            id: 'streets-layer',
            type: 'raster',
            source: 'streets',
            layout: {
              visibility: 'none',
            },
            minzoom: 0,
            maxzoom: 19,
          },
        ],
      },

      center: [0, 20],
      zoom: 2,
      minZoom: 1,
      maxZoom: 18,
    })

    mapInstance.current = map

    map.addControl(
      new maplibregl.NavigationControl({
        showCompass: true,
        showZoom: false,
      }),
      'top-right'
    )

    // Initialize Geoman after map loading.
    const initializeGeoman = async () => {
      if (cancelled) return

      try {
        const geoman = new Geoman(map)

        geomanInstance.current = geoman

        await geoman.waitForGeomanLoaded()

        if (cancelled) {
          await geoman.destroy({ removeSources: true })
          return
        }

        await geoman.addControls()

        if (cancelled) {
          await geoman.destroy({ removeSources: true })
          return
        }

        // Refresh AOI metadata after every geometry change.
        const updateAOIData = () => {
          const geojson = geoman.features.exportGeoJson()

          const stats = calculateAOIStats(geojson)

          setAoiGeoJSON(geojson)
          setAoiStats(stats)

          console.log('GEI AOI GeoJSON:', geojson)
          console.log('GEI AOI Statistics:', stats)
        }

        // Feature created.
        geoman.mapAdapter.on('gm:create', updateAOIData)

        // Update coordinates while polygon vertices are being edited.
        geoman.mapAdapter.on('gm:edit', updateAOIData)

        // Final update after editing.
        geoman.mapAdapter.on('gm:editend', updateAOIData)

        // Update after moving or rotating features.
        geoman.mapAdapter.on('gm:drag', updateAOIData)
        geoman.mapAdapter.on('gm:dragend', updateAOIData)
        geoman.mapAdapter.on('gm:rotate', updateAOIData)
        geoman.mapAdapter.on('gm:rotateend', updateAOIData)

        // Update after deleting features.
        geoman.mapAdapter.on('gm:remove', updateAOIData)

        setMapReady(true)
        setMapStatus('Drawing tools ready')

        console.log('GEI Geoman: Drawing tools initialized')

      } catch (error) {
        if (!cancelled) {
          console.error('GEI Geoman initialization failed:', error)
          setMapStatus('Drawing tools unavailable')
        }
      }
    }

    map.on('load', () => {
      map.resize()

      console.log('GEI MapLibre: Map initialized')

      initializeGeoman()
    })

    map.on('error', (event) => {
      console.error('GEI MapLibre resource error:', {
        message: event.error?.message,
        sourceId: event.sourceId,
        tile: event.tile,
      })
    })

    const resizeObserver = new ResizeObserver(() => {
      map.resize()
    })

    resizeObserver.observe(mapContainer.current)

    return () => {
      cancelled = true

      resizeObserver.disconnect()

      if (geomanInstance.current) {
        const geoman = geomanInstance.current
        geomanInstance.current = null

        void geoman.destroy({ removeSources: true })
      }

      map.remove()
      mapInstance.current = null
    }
  }, [])

  // Switch map layers.
  const switchLayer = (layer) => {
    const map = mapInstance.current

    if (!map || !mapReady || !map.isStyleLoaded()) return

    map.setLayoutProperty(
      'satellite-layer',
      'visibility',
      layer === 'satellite' ? 'visible' : 'none'
    )

    map.setLayoutProperty(
      'streets-layer',
      'visibility',
      layer === 'streets' ? 'visible' : 'none'
    )

    setActiveLayer(layer)
  }

  // Start polygon drawing.
  const startAOIDrawing = async () => {
    const geoman = geomanInstance.current

    if (!geoman) return

    try {
      await geoman.enableDraw('polygon')
      setMapStatus('Drawing investigation boundary')
    } catch (error) {
      console.error('Unable to start AOI drawing:', error)
    }
  }

  // Export AOI GeoJSON.
  const exportAOI = () => {
    if (!aoiGeoJSON || aoiStats.count === 0) return

    const blob = new Blob(
      [JSON.stringify(aoiGeoJSON, null, 2)],
      { type: 'application/geo+json' }
    )

    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')

    link.href = url
    link.download = 'gei_aoi.geojson'

    document.body.appendChild(link)
    link.click()
    link.remove()

    URL.revokeObjectURL(url)
  }

  // Map controls.
  const zoomIn = () => mapInstance.current?.zoomIn()

  const zoomOut = () => mapInstance.current?.zoomOut()

  const resetView = () => {
    mapInstance.current?.flyTo({
      center: [0, 20],
      zoom: 2,
      essential: true,
    })
  }

  const fullscreen = () => {
    mapContainer.current?.requestFullscreen?.()
  }

  const formatCoordinate = (value) => {
    return Number(value).toFixed(5)
  }

  return (
    <section className="overflow-hidden rounded-xl border border-[#263448] bg-[#101925]">

      {/* MAP HEADER */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#263448] px-4 py-3">

        <div className="flex items-center gap-2">

          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-cyan-500/10">
            <MapPin size={16} className="text-cyan-400" />
          </div>

          <div>
            <h3 className="text-sm font-semibold text-slate-100">
              Geospatial Workspace
            </h3>

            <p className="text-[10px] text-slate-500">
              {mapStatus} · Interactive map
            </p>
          </div>

        </div>

        <div className="flex items-center gap-2">

          <span className="rounded-md border border-[#344256] bg-[#172334] px-2 py-1 text-[10px] text-slate-400">
            WORLD VIEW
          </span>

          <button
            type="button"
            onClick={fullscreen}
            title="Fullscreen"
            className="rounded-md border border-[#344256] p-2 text-slate-400 hover:bg-[#25354a] hover:text-white"
          >
            <Maximize2 size={15} />
          </button>

        </div>
      </div>

      {/* MAP */}
      <div className="relative h-[520px] w-full bg-[#111b29]">

        <div
          ref={mapContainer}
          className="absolute inset-0 h-full w-full"
        />

        {/* AOI DRAWING BUTTONS */}
        <div className="absolute left-1/2 top-4 z-10 flex -translate-x-1/2 flex-wrap justify-center gap-2">

          <button
            type="button"
            onClick={startAOIDrawing}
            disabled={!mapReady}
            className="flex items-center gap-2 rounded-lg border border-cyan-400/30 bg-cyan-500 px-4 py-2 text-xs font-semibold text-slate-950 shadow-lg transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Pencil size={14} />
            Draw Investigation AOI
          </button>

          {aoiStats.count > 0 && (
            <button
              type="button"
              onClick={exportAOI}
              className="flex items-center gap-2 rounded-lg border border-white/10 bg-[#0b1420]/95 px-3 py-2 text-xs font-medium text-cyan-300 shadow-lg transition hover:bg-[#1b2b40]"
            >
              <Download size={14} />
              Export GeoJSON
            </button>
          )}

        </div>

        {/* MAP LAYER SWITCHER */}
        <div className="absolute right-4 top-16 z-10 rounded-lg border border-white/10 bg-[#0b1420]/95 p-2 shadow-lg backdrop-blur">

          <p className="mb-2 px-1 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
            Map Layers
          </p>

          <div className="flex gap-1">

            <button
              type="button"
              disabled={!mapReady}
              onClick={() => switchLayer('satellite')}
              className={`rounded-md px-3 py-2 text-xs transition ${
                activeLayer === 'satellite'
                  ? 'bg-cyan-500/15 text-cyan-300 ring-1 ring-cyan-500/40'
                  : 'text-slate-400 hover:bg-[#1b2b40] hover:text-white'
              } disabled:cursor-not-allowed disabled:opacity-50`}
            >
              Satellite
            </button>

            <button
              type="button"
              disabled={!mapReady}
              onClick={() => switchLayer('streets')}
              className={`rounded-md px-3 py-2 text-xs transition ${
                activeLayer === 'streets'
                  ? 'bg-cyan-500/15 text-cyan-300 ring-1 ring-cyan-500/40'
                  : 'text-slate-400 hover:bg-[#1b2b40] hover:text-white'
              } disabled:cursor-not-allowed disabled:opacity-50`}
            >
              Streets
            </button>

          </div>
        </div>

        {/* AOI METADATA PANEL */}
        <div className="absolute bottom-4 left-16 z-10 w-[270px] rounded-lg border border-white/10 bg-[#0b1420]/95 p-3 shadow-lg backdrop-blur">

          <div className="mb-3 flex items-center justify-between">

            <div className="flex items-center gap-2">
              <Crosshair size={15} className="text-cyan-400" />

              <p className="text-xs font-semibold text-slate-200">
                AOI Intelligence
              </p>
            </div>

            {aoiStats.count > 0 && (
                <span
                    className={`flex items-center gap-1 text-[10px] ${
                    aoiStats.valid ? 'text-emerald-400' : 'text-amber-400'
                    }`}
                >
                    {aoiStats.valid
                    ? <CheckCircle2 size={12} />
                    : <AlertTriangle size={12} />}

                    {aoiStats.valid
                    ? 'All geometries valid'
                    : `${aoiStats.invalidCount} invalid`}
                </span>
            )}

          </div>

          {aoiStats.count === 0 ? (

            <p className="text-[10px] leading-4 text-slate-400">
              No investigation area selected. Draw a polygon to calculate its geographic metadata.
            </p>

          ) : (

            <div className="space-y-3">

              <div className="grid grid-cols-2 gap-2">

                <div className="rounded-md bg-[#172334] p-2">
                  <p className="text-[9px] uppercase tracking-wider text-slate-500">
                    Total Area
                  </p>

                  <p className="mt-1 text-sm font-semibold text-cyan-300">
                    {aoiStats.areaKm2.toFixed(4)}
                  </p>

                  <p className="text-[9px] text-slate-500">
                    Square kilometres
                  </p>
                </div>

                <div className="rounded-md bg-[#172334] p-2">
                  <p className="text-[9px] uppercase tracking-wider text-slate-500">
                    Features
                  </p>

                  <p className="mt-1 text-sm font-semibold text-white">
                    {aoiStats.count}
                  </p>

                  <p className="text-[9px] text-slate-500">
                    Geographic regions
                  </p>
                </div>

              </div>

              {aoiStats.bounds && (
                <div>
                  <div className="mb-2 flex items-center gap-2">
                    <Activity size={12} className="text-cyan-400" />

                    <p className="text-[10px] font-medium text-slate-300">
                      Geographic Bounds
                    </p>
                  </div>

                  <div className="grid grid-cols-2 gap-x-3 gap-y-2">

                    {[
                      ['West', aoiStats.bounds[0]],
                      ['South', aoiStats.bounds[1]],
                      ['East', aoiStats.bounds[2]],
                      ['North', aoiStats.bounds[3]],
                    ].map(([label, value]) => (
                      <div key={label}>
                        <p className="text-[9px] text-slate-500">
                          {label}
                        </p>

                        <p className="font-mono text-[10px] text-slate-300">
                          {formatCoordinate(value)}°
                        </p>
                      </div>
                    ))}

                  </div>
                </div>
              )}

            </div>

          )}

        </div>

        {/* MAP CONTROLS */}
        <div className="absolute bottom-4 right-4 z-10 flex flex-col gap-1">

          <button
            type="button"
            onClick={zoomIn}
            aria-label="Zoom in"
            className="rounded-md border border-white/10 bg-[#0b1420]/90 p-2 text-white hover:bg-[#1b2b40]"
          >
            <Plus size={16} />
          </button>

          <button
            type="button"
            onClick={zoomOut}
            aria-label="Zoom out"
            className="rounded-md border border-white/10 bg-[#0b1420]/90 p-2 text-white hover:bg-[#1b2b40]"
          >
            <Minus size={16} />
          </button>

          <button
            type="button"
            onClick={resetView}
            aria-label="Reset map view"
            className="rounded-md border border-white/10 bg-[#0b1420]/90 p-2 text-white hover:bg-[#1b2b40]"
          >
            <Crosshair size={16} />
          </button>

        </div>

      </div>
    </section>
  )
}

export default MapPanel
