import { Suspense, lazy, useCallback, useEffect, useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import Nav, { useTheme } from '../components/Nav.jsx'
import ErrorBoundary from '../components/ErrorBoundary.jsx'
import AudioPlayer, { Attribution } from '../components/AudioPlayer.jsx'
import { SPHERE_RADIUS as RADIUS, hexToRgba } from '../theme.js'

const SphereView = lazy(() => import('../components/Sphere.jsx'))

const DATASETS = [
  { key: 'mtg_jamendo', label: 'MTG-Jamendo' },
  { key: 'music4all', label: 'Music4All' },
]

/** Rebuild the exporter's compact audio ref into what AudioPlayer expects. */
function audioOf(tracks, i) {
  const kind = tracks.audio_kind[i]
  if (kind === 1) {
    const id = tracks.audio_id[i]
    const lic = tracks.licenses[tracks.license_ref[i]] ?? { name: 'Creative Commons', url: '' }
    return {
      kind: 'jamendo',
      url: `https://mp3d.jamendo.com/?trackid=${id}&format=mp31`,
      page: `https://www.jamendo.com/track/${id}`,
      license: lic.name,
      license_url: lic.url,
    }
  }
  if (kind === 2) return { kind: 'spotify', id: tracks.audio_id[i] }
  return { kind: 'none' }
}

function trackOf(tracks, i) {
  return {
    clip_id: tracks.clip_id[i],
    title: tracks.title[i],
    artist: tracks.artist[i],
    tags: tracks.tags[i] ?? [],
    audio: audioOf(tracks, i),
  }
}

function useExplorerData(key) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    setData(null)
    setError(null)
    const base = `${import.meta.env.BASE_URL}data/explorer/${key}/`
    Promise.all([
      fetch(`${base}tracks.json`).then((r) => (r.ok ? r.json() : Promise.reject(new Error('tracks.json')))),
      fetch(`${base}chains.json`).then((r) => (r.ok ? r.json() : Promise.reject(new Error('chains.json')))),
      fetch(`${base}pos.bin`).then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error('pos.bin')))),
    ])
      .then(([tracks, chains, buf]) => {
        if (cancelled) return
        // int16 -> unit sphere -> render radius
        const raw = new Int16Array(buf)
        const pos = new Float32Array(raw.length)
        for (let i = 0; i < raw.length; i++) pos[i] = (raw[i] / 32767) * RADIUS
        setData({ tracks, chains, positions: pos })
      })
      .catch((e) => !cancelled && setError(e))
    return () => {
      cancelled = true
    }
  }, [key])

  return { data, error }
}

function GlassCard({ children, className = '' }) {
  return (
    <div
      className={`rounded-2xl border border-white/60 bg-white/70 shadow-xl backdrop-blur-2xl dark:border-white/10 dark:bg-neutral-900/60 ${className}`}
    >
      {children}
    </div>
  )
}

function TrackPane({ track, role, color }) {
  return (
    <div className="p-4">
      <div className="flex items-center gap-2">
        <span className="chip font-semibold text-white" style={{ backgroundColor: color }}>
          {role}
        </span>
        <p className="truncate text-sm font-semibold text-neutral-900 dark:text-neutral-100">{track.title}</p>
      </div>
      <p className="mt-0.5 truncate text-xs text-neutral-600 dark:text-neutral-400">{track.artist}</p>
      {track.tags?.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {track.tags.map((t) => (
            <span
              key={t}
              className="chip bg-neutral-100 text-[10px] text-neutral-600 dark:bg-neutral-800 dark:text-neutral-400"
            >
              {t}
            </span>
          ))}
        </div>
      )}
      <div className="mt-3">
        <AudioPlayer audio={track.audio} accent={color} compact />
      </div>
      <div className="mt-1.5">
        <Attribution track={track} />
      </div>
    </div>
  )
}

export default function Explore() {
  const [dark, setDark] = useTheme()
  const [dsIdx, setDsIdx] = useState(0)
  const [chainIdx, setChainIdx] = useState(0)
  const [stepIdx, setStepIdx] = useState(0)

  const ds = DATASETS[dsIdx]
  const { data, error } = useExplorerData(ds.key)

  const chain = data?.chains?.[chainIdx]
  const steps = chain?.st ?? []
  const step = steps[stepIdx]

  // Node indices along the chain: source of each step, plus the final target.
  const nodeIdx = useMemo(() => {
    if (!steps.length) return []
    return [...steps.map((s) => s.s), steps[steps.length - 1].t]
  }, [steps])

  const nodePositions = useMemo(() => {
    if (!data) return []
    return nodeIdx.map((i) => [data.positions[i * 3], data.positions[i * 3 + 1], data.positions[i * 3 + 2]])
  }, [data, nodeIdx])

  // Camera looks at the midpoint of the active edge.
  const focus = useMemo(() => {
    if (nodePositions.length < 2) return null
    const a = nodePositions[Math.min(stepIdx, nodePositions.length - 2)]
    const b = nodePositions[Math.min(stepIdx + 1, nodePositions.length - 1)]
    return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2]
  }, [nodePositions, stepIdx])

  const move = useCallback(
    (delta) => {
      if (!steps.length) return
      const next = stepIdx + delta
      if (next < 0) {
        setChainIdx((c) => {
          const n = Math.max(0, c - 1)
          setStepIdx(0)
          return n
        })
      } else if (next >= steps.length) {
        setChainIdx((c) => {
          const n = Math.min((data?.chains?.length ?? 1) - 1, c + 1)
          setStepIdx(0)
          return n
        })
      } else {
        setStepIdx(next)
      }
    },
    [stepIdx, steps.length, data],
  )

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'ArrowRight') move(1)
      else if (e.key === 'ArrowLeft') move(-1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [move])

  const pickChain = (i) => {
    setChainIdx(i)
    setStepIdx(0)
  }

  const src = data && step ? trackOf(data.tracks, step.s) : null
  const tgt = data && step ? trackOf(data.tracks, step.t) : null

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-white text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100">
      <Nav dark={dark} setDark={setDark} />

      <div className="relative flex-1 overflow-hidden">
        {/* Sphere */}
        <div className="absolute inset-0">
          {error ? (
            <div className="flex h-full items-center justify-center px-6 text-center text-sm text-neutral-500">
              Could not load explorer data ({String(error.message)}). Run
              <code className="mx-1 rounded bg-neutral-100 px-1 dark:bg-neutral-800">
                scripts/export_explorer_data.py
              </code>
              first.
            </div>
          ) : !data ? (
            <div className="flex h-full items-center justify-center">
              <div className="h-8 w-8 animate-spin rounded-full border-2 border-neutral-300 border-t-stage-validate" />
            </div>
          ) : (
            <ErrorBoundary label="Sphere">
              <Suspense fallback={null}>
                <SphereView
                  positions={data.positions}
                  chainNodes={nodePositions}
                  activeStep={stepIdx}
                  focus={focus}
                  dark={dark}
                />
              </Suspense>
            </ErrorBoundary>
          )}
        </div>

        {/* Dataset + chain picker */}
        <div className="pointer-events-none absolute left-0 right-0 top-0 p-4">
          <div className="pointer-events-auto flex flex-wrap items-center gap-3">
            <GlassCard className="inline-flex p-1">
              {DATASETS.map((d, i) => (
                <button
                  key={d.key}
                  type="button"
                  onClick={() => {
                    setDsIdx(i)
                    setChainIdx(0)
                    setStepIdx(0)
                  }}
                  className={`rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${
                    i === dsIdx
                      ? 'bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900'
                      : 'text-neutral-600 hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-neutral-100'
                  }`}
                >
                  {d.label}
                </button>
              ))}
            </GlassCard>

            {data && (
              <GlassCard className="px-3 py-1.5">
                <span className="text-xs text-neutral-600 dark:text-neutral-400">
                  {data.chains.length.toLocaleString()} chains · {data.tracks.clip_id.length.toLocaleString()} tracks
                </span>
              </GlassCard>
            )}
          </div>
        </div>

        {/* Chain navigation */}
        {data && (
          <div className="pointer-events-none absolute bottom-0 left-0 right-0 flex justify-center p-4">
            <GlassCard className="pointer-events-auto flex items-center gap-3 px-3 py-2">
              <button type="button" onClick={() => move(-1)} className="pill px-3 py-1 text-xs" aria-label="Previous">
                ←
              </button>
              <div className="flex items-center gap-1.5">
                {steps.map((_, i) => (
                  <button
                    key={i}
                    type="button"
                    onClick={() => setStepIdx(i)}
                    aria-label={`Turn ${i + 1}`}
                    className="h-1.5 rounded-full transition-all duration-300"
                    style={{
                      width: i === stepIdx ? 22 : 7,
                      backgroundColor: i === stepIdx ? '#FB8B24' : hexToRgba('#FB8B24', 0.3),
                    }}
                  />
                ))}
              </div>
              <span className="whitespace-nowrap text-xs text-neutral-500 dark:text-neutral-400">
                chain {chainIdx + 1} · turn {stepIdx + 1}/{steps.length}
              </span>
              <button type="button" onClick={() => move(1)} className="pill px-3 py-1 text-xs" aria-label="Next">
                →
              </button>
              <input
                type="range"
                min={0}
                max={Math.max(0, data.chains.length - 1)}
                value={chainIdx}
                onChange={(e) => pickChain(Number(e.target.value))}
                className="ml-1 h-1 w-28 cursor-pointer accent-stage-validate"
                aria-label="Jump to chain"
              />
            </GlassCard>
          </div>
        )}

        {/* The chain card: source -> instruction -> target */}
        <AnimatePresence mode="wait">
          {src && tgt && (
            <motion.div
              key={`${chainIdx}-${stepIdx}`}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.25, ease: 'easeOut' }}
              className="pointer-events-none absolute right-0 top-14 bottom-20 flex w-full max-w-sm items-center p-4"
            >
              <GlassCard className="pointer-events-auto max-h-full w-full overflow-y-auto">
                <TrackPane track={src} role={stepIdx === 0 ? 'Start' : `Turn ${stepIdx}`} color="#1FA347" />

                {/* The instruction is the bridge — same orange as the active arc on the sphere. */}
                <div className="relative px-4">
                  <div
                    className="rounded-xl border px-3 py-2.5"
                    style={{
                      borderColor: hexToRgba('#FB8B24', 0.5),
                      backgroundColor: hexToRgba('#FB8B24', 0.08),
                    }}
                  >
                    <p className="text-sm font-medium leading-snug text-neutral-900 dark:text-neutral-100">
                      “{step.i}”
                    </p>
                    {step.ax?.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1">
                        {step.ax.map((a) => (
                          <span
                            key={a}
                            className="chip text-[10px] font-medium"
                            style={{ backgroundColor: hexToRgba('#FB8B24', 0.16), color: '#B45309' }}
                          >
                            {a.replace(/_/g, ' ')}
                          </span>
                        ))}
                      </div>
                    )}
                    <p className="mt-1.5 text-[10px] text-neutral-500 dark:text-neutral-400">
                      similarity {step.sc}
                    </p>
                  </div>
                </div>

                <TrackPane
                  track={tgt}
                  role={stepIdx === steps.length - 1 ? 'End' : `Turn ${stepIdx + 1}`}
                  color={stepIdx === steps.length - 1 ? '#7B3FF2' : '#2E6FD6'}
                />
              </GlassCard>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="pointer-events-none absolute bottom-4 left-4 hidden text-[11px] text-neutral-400 md:block">
          ← → to move through turns · drag to orbit · scroll to zoom
        </div>
      </div>
    </div>
  )
}
