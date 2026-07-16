import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { STAGES, hexToRgba } from '../theme.js'
import StageArt from './StageArt.jsx'

export default function Pipeline() {
  const [active, setActive] = useState(0)
  const stage = STAGES[active]

  return (
    <div>
      {/* Stage rail — the five chips from the paper figure. */}
      <div className="mb-6 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
        {STAGES.map((s, i) => {
          const on = i === active
          return (
            <button
              key={s.id}
              type="button"
              onClick={() => setActive(i)}
              className="rounded-xl border px-3 py-2.5 text-left transition-all duration-200"
              style={{
                borderColor: on ? s.color : hexToRgba(s.color, 0.28),
                backgroundColor: on ? hexToRgba(s.color, 0.12) : 'transparent',
                boxShadow: on ? `0 1px 12px ${hexToRgba(s.color, 0.18)}` : 'none',
              }}
            >
              <span className="text-[11px] font-bold" style={{ color: s.color }}>
                {s.n}
              </span>
              <span
                className="mt-0.5 block text-xs font-semibold leading-tight"
                style={{ color: on ? s.color : undefined }}
              >
                {s.name}
              </span>
            </button>
          )
        })}
      </div>

      <div
        className="overflow-hidden rounded-2xl border"
        style={{ borderColor: hexToRgba(stage.color, 0.35), backgroundColor: hexToRgba(stage.color, 0.04) }}
      >
        <AnimatePresence mode="wait">
          <motion.div
            key={stage.id}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.24, ease: 'easeOut' }}
            className="grid items-center gap-6 p-6 md:grid-cols-2 md:p-8"
          >
            <div>
              <span
                className="chip font-semibold text-white"
                style={{ backgroundColor: stage.color }}
              >
                Stage {stage.n}
              </span>
              <h3 className="mt-3 text-xl font-semibold text-neutral-900 dark:text-neutral-100">
                {stage.name}
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-neutral-700 dark:text-neutral-300">
                {stage.blurb}
              </p>
              <p className="mt-3 text-sm leading-relaxed text-neutral-600 dark:text-neutral-400">
                {stage.detail}
              </p>
            </div>
            <div className="flex items-center justify-center">
              <StageArt id={stage.id} color={stage.color} />
            </div>
          </motion.div>
        </AnimatePresence>
      </div>

      <div className="mt-4 flex items-center justify-between">
        <button
          type="button"
          onClick={() => setActive((a) => Math.max(0, a - 1))}
          disabled={active === 0}
          className="pill text-xs disabled:pointer-events-none disabled:opacity-30"
        >
          ← Previous
        </button>
        <div className="flex gap-1.5">
          {STAGES.map((s, i) => (
            <button
              key={s.id}
              type="button"
              onClick={() => setActive(i)}
              aria-label={s.name}
              className="h-1.5 rounded-full transition-all duration-300"
              style={{
                width: i === active ? 24 : 8,
                backgroundColor: i === active ? s.color : hexToRgba(s.color, 0.25),
              }}
            />
          ))}
        </div>
        <button
          type="button"
          onClick={() => setActive((a) => Math.min(STAGES.length - 1, a + 1))}
          disabled={active === STAGES.length - 1}
          className="pill text-xs disabled:pointer-events-none disabled:opacity-30"
        >
          Next →
        </button>
      </div>
    </div>
  )
}
