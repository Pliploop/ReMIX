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

const START = '#1FA347'
const MID = '#2E6FD6'
const END = '#7B3FF2'
const INSTRUCT = '#FB8B24'

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

/**
 * The instruction variants for a step.
 *
 * `v` (all surviving variants) only exists once export_explorer_data.py has been
 * re-run with --all-variants; until then a step carries just the single best
 * variant it was exported with, and the dropdown collapses to one entry.
 */
function variantsOf(step) {
  if (Array.isArray(step?.v) && step.v.length) return step.v
  return step ? [{ i: step.i, c: step.c }] : []
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

function IconButton({ onClick, label, children, disabled }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={label}
      className="flex h-7 w-7 items-center justify-center rounded-full text-neutral-600 transition-colors hover:bg-neutral-900/5 hover:text-neutral-900 disabled:opacity-25 dark:text-neutral-400 dark:hover:bg-white/10 dark:hover:text-neutral-100"
    >
      {children}
    </button>
  )
}

/** A track pinned to its own node on the sphere. Deliberately small: it floats over the cloud. */
function SphereCard({ track, role, color }) {
  return (
    <div className="w-56 rounded-xl border border-white/60 bg-white/80 p-2.5 shadow-2xl backdrop-blur-xl dark:border-white/10 dark:bg-neutral-900/80">
      <div className="flex items-center gap-1.5">
        <span
          className="rounded-full px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wide text-white"
          style={{ backgroundColor: color }}
        >
          {role}
        </span>
        <p className="truncate text-[11px] font-semibold text-neutral-900 dark:text-neutral-100">{track.title}</p>
      </div>
      <p className="mt-0.5 truncate text-[10px] text-neutral-500 dark:text-neutral-400">{track.artist}</p>
      <div className="mt-1.5">
        {track.audio.kind === 'jamendo' ? (
          <AudioPlayer audio={track.audio} accent={color} clipId={track.clip_id} compact />
        ) : (
          /* A 152px Spotify iframe cannot float on a sphere; the panel has the player. */
          <p className="text-[10px] text-neutral-500 dark:text-neutral-400">
            Spotify player in the panel →
          </p>
        )}
      </div>
    </div>
  )
}

/**
 * The panel half of a track.
 *
 * Each track gets exactly one player, and where it lives depends on what will fit
 * where. A Jamendo waveform is small and pins nicely to its node, so it plays on
 * the sphere and this pane carries only metadata. A Spotify embed cannot float on
 * a node -- it is a fixed 152px iframe -- so it plays here instead. Rendering a
 * player in both places would give one track two unsynchronised play buttons that
 * happily talk over each other, and would decode the audio twice.
 */
function TrackPane({ track, role, color }) {
  const onSphere = track.audio.kind === 'jamendo'
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
      {!onSphere && (
        <div className="mt-3">
          <AudioPlayer audio={track.audio} accent={color} clipId={track.clip_id} />
        </div>
      )}
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
  const [variantIdx, setVariantIdx] = useState(0)
  const [contextual, setContextual] = useState(false)
  const [panelOpen, setPanelOpen] = useState(true)
  const [homeTick, setHomeTick] = useState(0)

  const ds = DATASETS[dsIdx]
  const { data, error } = useExplorerData(ds.key)

  const chain = data?.chains?.[chainIdx]
  const steps = chain?.st ?? []
  const step = steps[stepIdx]

  const variants = variantsOf(step)
  const variant = variants[Math.min(variantIdx, variants.length - 1)] ?? null
  const instruction = variant ? (contextual && variant.c ? variant.c : variant.i) : ''

  // Which chain (and turn) each track belongs to, so a click on the cloud can open it.
  const trackIndex = useMemo(() => {
    if (!data) return null
    const m = new Map()
    data.chains.forEach((ch, ci) => {
      ch.st.forEach((s, si) => {
        if (!m.has(s.s)) m.set(s.s, { chain: ci, step: si })
        if (!m.has(s.t)) m.set(s.t, { chain: ci, step: si })
      })
    })
    return m
  }, [data])

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

  // Only a new chain — or Home — flies the camera. Stepping within a chain moves
  // a short distance, and yanking the view on every arrow key made the sphere
  // impossible to explore.
  const flyKey = `${ds.key}:${chainIdx}:${homeTick}`

  const goChain = useCallback((i, step = 0) => {
    setChainIdx(i)
    setStepIdx(step)
    setVariantIdx(0)
  }, [])

  /** Arrow keys walk turns, and run on into the neighbouring chain at either end. */
  const move = useCallback(
    (delta) => {
      if (!steps.length) return
      const next = stepIdx + delta
      if (next < 0) {
        if (chainIdx > 0) goChain(chainIdx - 1, 0)
      } else if (next >= steps.length) {
        if (chainIdx < (data?.chains?.length ?? 1) - 1) goChain(chainIdx + 1, 0)
      } else {
        setStepIdx(next)
        setVariantIdx(0)
      }
    },
    [stepIdx, steps.length, chainIdx, data, goChain],
  )

  useEffect(() => {
    const onKey = (e) => {
      if (e.target?.tagName === 'INPUT' || e.target?.tagName === 'SELECT') return
      if (e.key === 'ArrowRight') move(1)
      else if (e.key === 'ArrowLeft') move(-1)
      else if (e.key === 'h' || e.key === 'H') setHomeTick((t) => t + 1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [move])

  const onPick = useCallback(
    (i) => {
      const hit = trackIndex?.get(i)
      if (hit) {
        goChain(hit.chain, hit.step)
        setHomeTick((t) => t + 1)
      }
    },
    [trackIndex, goChain],
  )

  const tooltipFor = useCallback(
    (i) => {
      if (!data) return null
      const hit = trackIndex?.get(i)
      return {
        title: data.tracks.title[i],
        artist: data.tracks.artist[i],
        hint: hit
          ? hit.chain === chainIdx
            ? 'in this chain'
            : `click → chain ${hit.chain + 1}`
          : 'not in a chain',
      }
    },
    [data, trackIndex, chainIdx],
  )

  const src = data && step ? trackOf(data.tracks, step.s) : null
  const tgt = data && step ? trackOf(data.tracks, step.t) : null

  // Cards pinned in 3D: the two endpoints of the current turn, and the
  // instruction floating over the arc that joins them.
  const cards = useMemo(() => {
    if (!src || !tgt || nodePositions.length < 2) return []
    const a = nodePositions[stepIdx]
    const b = nodePositions[stepIdx + 1]
    if (!a || !b) return []
    const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2]
    return [
      {
        key: `s-${chainIdx}-${stepIdx}`,
        position: a,
        k: 1.06,
        node: <SphereCard track={src} role={stepIdx === 0 ? 'Start' : `Turn ${stepIdx}`} color={stepIdx === 0 ? START : MID} />,
      },
      {
        key: `t-${chainIdx}-${stepIdx}`,
        position: b,
        k: 1.06,
        node: (
          <SphereCard
            track={tgt}
            role={stepIdx === steps.length - 1 ? 'End' : `Turn ${stepIdx + 1}`}
            color={stepIdx === steps.length - 1 ? END : MID}
          />
        ),
      },
      {
        key: `i-${chainIdx}-${stepIdx}`,
        position: mid,
        k: 1.3,
        node: (
          <div
            className="max-w-[15rem] rounded-full border px-3 py-1.5 text-center shadow-xl backdrop-blur-xl"
            style={{ borderColor: hexToRgba(INSTRUCT, 0.6), backgroundColor: hexToRgba(INSTRUCT, 0.14) }}
          >
            <p className="truncate text-[11px] font-semibold text-neutral-900 dark:text-neutral-100">
              “{instruction}”
            </p>
          </div>
        ),
      },
    ]
  }, [src, tgt, nodePositions, stepIdx, chainIdx, steps.length, instruction])

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
                  flyKey={flyKey}
                  dark={dark}
                  onPick={onPick}
                  tooltipFor={tooltipFor}
                  cards={cards}
                />
              </Suspense>
            </ErrorBoundary>
          )}
        </div>

        {/* Dataset picker + corpus size */}
        <div className="pointer-events-none absolute left-0 right-0 top-0 p-4">
          <div className="pointer-events-auto flex flex-wrap items-center gap-3">
            <GlassCard className="inline-flex p-1">
              {DATASETS.map((d, i) => (
                <button
                  key={d.key}
                  type="button"
                  onClick={() => {
                    setDsIdx(i)
                    goChain(0)
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

            <GlassCard className="inline-flex items-center gap-1 px-1 py-1">
              <button
                type="button"
                onClick={() => setHomeTick((t) => t + 1)}
                className="flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium text-neutral-600 transition-colors hover:bg-neutral-900/5 hover:text-neutral-900 dark:text-neutral-400 dark:hover:bg-white/10 dark:hover:text-neutral-100"
                title="Recentre on the current turn (H)"
              >
                <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                  <path strokeLinecap="round" strokeLinejoin="round" d="m3 11 9-8 9 8M5 9v11h14V9" />
                </svg>
                Home
              </button>
            </GlassCard>
          </div>
        </div>

        {/* Navigation: turns on top, chains beneath. Two separate axes, two rows. */}
        {data && steps.length > 0 && (
          <div className="pointer-events-none absolute bottom-0 left-0 right-0 flex justify-center p-4">
            <GlassCard className="pointer-events-auto divide-y divide-neutral-200/60 px-3 py-2 dark:divide-white/10">
              {/* Turn within the chain — what the arrow keys drive. */}
              <div className="flex items-center gap-3 pb-2">
                <span className="w-10 text-[10px] font-semibold uppercase tracking-wide text-neutral-400">turn</span>
                <IconButton onClick={() => move(-1)} label="Previous turn">←</IconButton>
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
                        backgroundColor: i === stepIdx ? INSTRUCT : hexToRgba(INSTRUCT, 0.3),
                      }}
                    />
                  ))}
                </div>
                <IconButton onClick={() => move(1)} label="Next turn">→</IconButton>
                <span className="whitespace-nowrap text-xs tabular-nums text-neutral-500 dark:text-neutral-400">
                  {stepIdx + 1}/{steps.length}
                </span>
              </div>

              {/* Chain — its own control, so the arrow keys never jump you off a chain by accident. */}
              <div className="flex items-center gap-3 pt-2">
                <span className="w-10 text-[10px] font-semibold uppercase tracking-wide text-neutral-400">chain</span>
                <IconButton onClick={() => goChain(Math.max(0, chainIdx - 1))} label="Previous chain" disabled={chainIdx === 0}>
                  ←
                </IconButton>
                <input
                  type="range"
                  min={0}
                  max={Math.max(0, data.chains.length - 1)}
                  value={chainIdx}
                  onChange={(e) => goChain(Number(e.target.value))}
                  className="h-1 w-40 cursor-pointer accent-stage-validate"
                  aria-label="Jump to chain"
                />
                <IconButton
                  onClick={() => goChain(Math.min(data.chains.length - 1, chainIdx + 1))}
                  label="Next chain"
                  disabled={chainIdx >= data.chains.length - 1}
                >
                  →
                </IconButton>
                <span className="whitespace-nowrap text-xs tabular-nums text-neutral-500 dark:text-neutral-400">
                  {(chainIdx + 1).toLocaleString()}/{data.chains.length.toLocaleString()}
                </span>
              </div>
            </GlassCard>
          </div>
        )}

        {/* Detail panel */}
        {src && tgt && (
          <div className="pointer-events-none absolute right-3 top-16 bottom-24 flex w-full max-w-[21rem] items-start justify-end">
            <AnimatePresence initial={false} mode="wait">
              {panelOpen ? (
                <motion.div
                  key="panel"
                  initial={{ opacity: 0, x: 24 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 24 }}
                  transition={{ duration: 0.22, ease: 'easeOut' }}
                  className="pointer-events-auto max-h-full w-full"
                >
                  <GlassCard className="flex max-h-full w-full flex-col overflow-hidden">
                    <div className="flex items-center justify-between gap-2 border-b border-neutral-200/60 px-3 py-2 dark:border-white/10">
                      <span className="text-[10px] font-semibold uppercase tracking-wide text-neutral-400">
                        chain {chainIdx + 1} · turn {stepIdx + 1}
                      </span>
                      <IconButton onClick={() => setPanelOpen(false)} label="Collapse panel">
                        <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                          <path strokeLinecap="round" strokeLinejoin="round" d="m9 6 6 6-6 6" />
                        </svg>
                      </IconButton>
                    </div>

                    <div className="min-h-0 flex-1 overflow-y-auto">
                      <TrackPane track={src} role={stepIdx === 0 ? 'Start' : `Turn ${stepIdx}`} color={stepIdx === 0 ? START : MID} />

                      {/* The instruction is the bridge — same orange as the active arc. */}
                      <div className="px-4">
                        <div
                          className="rounded-xl border px-3 py-2.5"
                          style={{
                            borderColor: hexToRgba(INSTRUCT, 0.5),
                            backgroundColor: hexToRgba(INSTRUCT, 0.08),
                          }}
                        >
                          <div className="flex items-center gap-1.5">
                            <select
                              value={Math.min(variantIdx, variants.length - 1)}
                              onChange={(e) => setVariantIdx(Number(e.target.value))}
                              disabled={variants.length < 2}
                              aria-label="Instruction variant"
                              className="rounded-md border border-neutral-300/70 bg-white/70 px-1.5 py-0.5 text-[10px] font-medium text-neutral-700 disabled:opacity-50 dark:border-white/10 dark:bg-neutral-800/70 dark:text-neutral-300"
                            >
                              {variants.map((v, i) => (
                                <option key={i} value={i}>
                                  Variant {i + 1}
                                  {v.vb ? ` · ${v.vb}` : ''}
                                </option>
                              ))}
                            </select>

                            {variant?.c && variant.c !== variant.i && (
                              <button
                                type="button"
                                onClick={() => setContextual((c) => !c)}
                                className="rounded-md px-1.5 py-0.5 text-[10px] font-medium transition-colors"
                                style={{
                                  backgroundColor: contextual ? hexToRgba(INSTRUCT, 0.2) : 'transparent',
                                  color: contextual ? '#B45309' : undefined,
                                }}
                                title="Show the phrasing that may refer back to earlier turns"
                              >
                                contextual
                              </button>
                            )}
                          </div>

                          <p className="mt-2 text-sm font-medium leading-snug text-neutral-900 dark:text-neutral-100">
                            “{instruction}”
                          </p>

                          {step.ax?.length > 0 && (
                            <div className="mt-2 flex flex-wrap gap-1">
                              {step.ax.map((a) => (
                                <span
                                  key={a}
                                  className="chip text-[10px] font-medium"
                                  style={{ backgroundColor: hexToRgba(INSTRUCT, 0.16), color: '#B45309' }}
                                >
                                  {a.replace(/_/g, ' ')}
                                </span>
                              ))}
                            </div>
                          )}
                          {step.e && (
                            <p className="mt-1.5 text-[11px] leading-snug text-neutral-500 dark:text-neutral-400">
                              {step.e}
                            </p>
                          )}
                          <p className="mt-1.5 text-[10px] text-neutral-500 dark:text-neutral-400">
                            similarity {step.sc}
                          </p>
                        </div>
                      </div>

                      <TrackPane
                        track={tgt}
                        role={stepIdx === steps.length - 1 ? 'End' : `Turn ${stepIdx + 1}`}
                        color={stepIdx === steps.length - 1 ? END : MID}
                      />
                    </div>
                  </GlassCard>
                </motion.div>
              ) : (
                <motion.button
                  key="tab"
                  type="button"
                  initial={{ opacity: 0, x: 24 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 24 }}
                  onClick={() => setPanelOpen(true)}
                  className="pointer-events-auto flex items-center gap-1.5 rounded-full border border-white/60 bg-white/70 px-3 py-2 text-xs font-medium shadow-xl backdrop-blur-2xl dark:border-white/10 dark:bg-neutral-900/60"
                >
                  <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                    <path strokeLinecap="round" strokeLinejoin="round" d="m15 6-6 6 6 6" />
                  </svg>
                  Details
                </motion.button>
              )}
            </AnimatePresence>
          </div>
        )}

        <div className="pointer-events-none absolute bottom-4 left-4 hidden text-[11px] leading-relaxed text-neutral-400 md:block">
          ← → turns · drag to orbit · scroll to zoom
          <br />
          hover a track for its title · click to open its chain · H to recentre
        </div>
      </div>
    </div>
  )
}
