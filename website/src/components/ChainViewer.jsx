import { useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import AudioPlayer, { Attribution } from './AudioPlayer.jsx'
import { STAGE_BY_ID } from '../theme.js'

const INSTRUCT = STAGE_BY_ID.instruct.color
const CHAIN = STAGE_BY_ID.chain.color

function TrackCard({ track, badge, accent }) {
  return (
    <div className="surface p-6">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="eyebrow flex items-center gap-2">
            <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: accent }} />
            {badge}
          </p>
          <h4 className="mt-2 truncate text-lg font-medium tracking-[-0.01em] text-ink dark:text-neutral-100">
            {track.title}
          </h4>
          <p className="mt-0.5 truncate text-sm text-ink-2 dark:text-neutral-400">{track.artist}</p>
        </div>
        {track.tags?.length > 0 && (
          <div className="hidden flex-wrap justify-end gap-1.5 sm:flex">
            {track.tags.slice(0, 4).map((t) => (
              <span key={t} className="chip">
                {t}
              </span>
            ))}
          </div>
        )}
      </div>

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
    <div className="relative py-4 pl-10">
      <span className="absolute left-[15px] top-0 h-full w-px bg-line dark:bg-white/10" aria-hidden />
      <span
        className="absolute left-[10px] top-1/2 h-[11px] w-[11px] -translate-y-1/2 rounded-full bg-ground ring-2 dark:bg-ground-dark"
        style={{ '--tw-ring-color': INSTRUCT }}
        aria-hidden
      />

      <button
        type="button"
        onClick={onToggle}
        className="relative w-full overflow-hidden rounded-xl border border-line bg-white/60 py-3.5 pl-5 pr-4 text-left transition-colors hover:bg-white dark:border-white/10 dark:bg-white/[0.03] dark:hover:bg-white/[0.06]"
      >
        <span className="absolute inset-y-0 left-0 w-[3px]" style={{ backgroundColor: INSTRUCT }} aria-hidden />
        <div className="flex items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-2.5">
            <svg className="h-4 w-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke={INSTRUCT} strokeWidth="2">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M8 10h8M8 14h5M21 12a8 8 0 0 1-8 8H7l-4 3v-6.5A8 8 0 0 1 11 4h2a8 8 0 0 1 8 8z"
              />
            </svg>
            <p className="truncate text-[15px] text-ink dark:text-neutral-100">
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
                  <DeltaList title="Dropped" items={step.lost} color={STAGE_BY_ID.enrich.color} />
                  <DeltaList title="Introduced" items={step.new} color={STAGE_BY_ID.chain.color} />
                  <DeltaList title="Kept" items={step.preserved} color={STAGE_BY_ID.neighbour.color} />
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
      <p className="eyebrow">{label}</p>
      <p className="mt-1 text-sm text-ink dark:text-neutral-300">{children}</p>
    </div>
  )
}

function DeltaList({ title, items, color }) {
  if (!items?.length) return null
  return (
    <div>
      <p className="eyebrow mb-2 flex items-center gap-1.5">
        <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: color }} />
        {title}
      </p>
      <ul className="space-y-1">
        {items.slice(0, 4).map((it) => (
          <li key={it} className="text-[13px] leading-snug text-ink-2 dark:text-neutral-400">
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
      <div className="mb-8 flex flex-wrap items-center gap-4">
        <div className="segmented">
          {datasets.map((d, i) => (
            <button key={d.key} type="button" data-on={i === dsIdx} onClick={() => pickDataset(i)}>
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
              className={`h-8 w-8 rounded-full text-xs font-medium tabular-nums transition-colors ${
                i === chainIdx
                  ? 'bg-ink text-white dark:bg-neutral-100 dark:text-neutral-900'
                  : 'text-ink-2 ring-1 ring-inset ring-line hover:text-ink dark:ring-white/10 dark:hover:text-neutral-100'
              }`}
              aria-label={`Chain ${i + 1}`}
            >
              {i + 1}
            </button>
          ))}
        </div>

        <span className="ml-auto text-xs tabular-nums text-ink-3">
          {chain.steps.length} turns · {chain.split} split
        </span>
      </div>

      <motion.div key={chain.chain_id} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}>
        {tracks.map((track, i) => (
          <div key={`${chain.chain_id}-${i}`}>
            <TrackCard
              track={track}
              badge={i === 0 ? 'Start' : i === tracks.length - 1 ? 'End' : `Turn ${i}`}
              accent={i === 0 ? CHAIN : i === tracks.length - 1 ? STAGE_BY_ID.validate.color : STAGE_BY_ID.neighbour.color}
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
