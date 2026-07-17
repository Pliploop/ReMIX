import { useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import AudioPlayer, { Attribution } from './AudioPlayer.jsx'
import { hexToRgba } from '../theme.js'

const INSTRUCT = '#FB8B24'
const CHAIN = '#1FA347'

function TrackCard({ track, badge, accent }) {
  return (
    <div
      className="stage-card"
      style={{ borderColor: hexToRgba(accent, 0.35), backgroundColor: hexToRgba(accent, 0.05) }}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span
              className="chip shrink-0 font-semibold text-white"
              style={{ backgroundColor: accent }}
            >
              {badge}
            </span>
            <h4 className="truncate text-sm font-semibold text-neutral-900 dark:text-neutral-100">
              {track.title}
            </h4>
          </div>
          <p className="mt-1 truncate text-xs text-neutral-600 dark:text-neutral-400">{track.artist}</p>
        </div>
      </div>

      {track.tags?.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {track.tags.slice(0, 4).map((t) => (
            <span
              key={t}
              className="chip bg-white/70 text-neutral-700 ring-1 ring-inset ring-neutral-200 dark:bg-neutral-800/60 dark:text-neutral-300 dark:ring-neutral-700"
            >
              {t}
            </span>
          ))}
        </div>
      )}

      <div className="mt-4">
        <AudioPlayer audio={track.audio} accent={accent} clipId={track.clip_id} compact />
      </div>
      <div className="mt-2">
        <Attribution track={track} />
      </div>
    </div>
  )
}

/** The instruction is the hero: it sits on the arrow between the two tracks. */
function InstructionBridge({ step, open, onToggle }) {
  return (
    <div className="relative py-3 pl-6">
      <span
        className="absolute left-[7px] top-0 h-full w-px"
        style={{ backgroundColor: hexToRgba(CHAIN, 0.3) }}
        aria-hidden
      />
      <span
        className="absolute left-0 top-1/2 h-3.5 w-3.5 -translate-y-1/2 rounded-full border-2 border-white dark:border-neutral-950"
        style={{ backgroundColor: CHAIN }}
        aria-hidden
      />

      <button
        type="button"
        onClick={onToggle}
        className="w-full rounded-2xl border px-4 py-3 text-left transition-shadow hover:shadow-md"
        style={{ borderColor: hexToRgba(INSTRUCT, 0.45), backgroundColor: hexToRgba(INSTRUCT, 0.07) }}
      >
        <div className="flex items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-2.5">
            <svg className="h-4 w-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke={INSTRUCT} strokeWidth="2">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M8 10h8M8 14h5M21 12a8 8 0 0 1-8 8H7l-4 3v-6.5A8 8 0 0 1 11 4h2a8 8 0 0 1 8 8z"
              />
            </svg>
            <p className="truncate text-sm font-medium text-neutral-900 dark:text-neutral-100">
              “{step.instruction}”
            </p>
          </div>
          <svg
            className={`h-4 w-4 shrink-0 text-neutral-400 transition-transform ${open ? 'rotate-180' : ''}`}
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="m6 9 6 6 6-6" />
          </svg>
        </div>

        <AnimatePresence initial={false}>
          {open && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.22, ease: 'easeOut' }}
              className="overflow-hidden"
            >
              <div className="space-y-3 pt-3">
                {step.instruction_contextual && step.instruction_contextual !== step.instruction && (
                  <Detail label="Contextual phrasing">“{step.instruction_contextual}”</Detail>
                )}
                <div className="grid gap-2 sm:grid-cols-3">
                  <DeltaList title="Dropped" items={step.lost} color="#E23B34" />
                  <DeltaList title="Introduced" items={step.new} color="#1FA347" />
                  <DeltaList title="Kept" items={step.preserved} color="#2E6FD6" />
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </button>
    </div>
  )
}

function Detail({ label, children }) {
  return (
    <div>
      <p className="text-[11px] font-semibold uppercase tracking-wide text-neutral-500 dark:text-neutral-400">
        {label}
      </p>
      <p className="mt-0.5 text-sm text-neutral-700 dark:text-neutral-300">{children}</p>
    </div>
  )
}

function DeltaList({ title, items, color }) {
  if (!items?.length) return null
  return (
    <div>
      <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide" style={{ color }}>
        {title}
      </p>
      <ul className="space-y-1">
        {items.slice(0, 4).map((it) => (
          <li key={it} className="text-xs leading-snug text-neutral-600 dark:text-neutral-400">
            {it}
          </li>
        ))}
      </ul>
    </div>
  )
}

export default function ChainViewer({ data }) {
  const datasets = data?.datasets ?? []
  // Open on Music4All — its chains are the stronger showcase — wherever it sits
  // in the exported order.
  const defaultIdx = Math.max(0, datasets.findIndex((d) => d.key === 'music4all'))
  const [dsIdx, setDsIdx] = useState(defaultIdx)
  const [chainIdx, setChainIdx] = useState(0)
  const [openStep, setOpenStep] = useState(0)

  const dataset = datasets[dsIdx]
  const chain = dataset?.chains?.[chainIdx]

  const tracks = useMemo(() => {
    if (!chain) return []
    return [chain.steps[0].source, ...chain.steps.map((s) => s.target)]
  }, [chain])

  if (!dataset || !chain) return null

  const pickDataset = (i) => {
    setDsIdx(i)
    setChainIdx(0)
    setOpenStep(0)
  }

  return (
    <div>
      <div className="mb-5 flex flex-wrap items-center gap-3">
        <div className="inline-flex rounded-full border border-neutral-200 p-1 dark:border-neutral-700">
          {datasets.map((d, i) => (
            <button
              key={d.key}
              type="button"
              onClick={() => pickDataset(i)}
              className={`rounded-full px-3.5 py-1.5 text-xs font-medium transition-colors ${
                i === dsIdx
                  ? 'bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900'
                  : 'text-neutral-600 hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-neutral-100'
              }`}
            >
              {d.label}
            </button>
          ))}
        </div>

        <div className="flex flex-wrap gap-1.5">
          {dataset.chains.map((c, i) => (
            <button
              key={c.chain_id}
              type="button"
              onClick={() => {
                setChainIdx(i)
                setOpenStep(0)
              }}
              className={`h-7 w-7 rounded-md text-xs font-medium transition-colors ${
                i === chainIdx
                  ? 'text-white'
                  : 'text-neutral-500 ring-1 ring-inset ring-neutral-200 hover:text-neutral-900 dark:ring-neutral-700 dark:hover:text-neutral-100'
              }`}
              style={i === chainIdx ? { backgroundColor: CHAIN } : undefined}
              aria-label={`Chain ${i + 1}`}
            >
              {i + 1}
            </button>
          ))}
        </div>

        <span className="ml-auto text-xs text-neutral-500 dark:text-neutral-400">
          {chain.steps.length} turns · {chain.split} split
        </span>
      </div>

      <motion.div key={chain.chain_id} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}>
        {tracks.map((track, i) => (
          <div key={`${chain.chain_id}-${i}`}>
            <TrackCard
              track={track}
              badge={i === 0 ? 'Start' : i === tracks.length - 1 ? 'End' : `Turn ${i}`}
              accent={i === 0 ? CHAIN : i === tracks.length - 1 ? '#7B3FF2' : '#2E6FD6'}
            />
            {i < chain.steps.length && (
              <InstructionBridge
                step={chain.steps[i]}
                open={openStep === i}
                onToggle={() => setOpenStep(openStep === i ? -1 : i)}
              />
            )}
          </div>
        ))}
      </motion.div>
    </div>
  )
}
