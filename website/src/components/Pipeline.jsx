import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { STAGES } from '../theme.js'
import StageArt from './StageArt.jsx'

export default function Pipeline() {
  const [active, setActive] = useState(0)
  const stage = STAGES[active]

  return (
    <div>
      {/* Stepper: a hairline per stage that fills with the stage colour when active. */}
      <div className="mb-8 grid grid-cols-2 gap-x-4 gap-y-5 sm:grid-cols-3 lg:grid-cols-5">
        {STAGES.map((s, i) => {
          const on = i === active
          return (
            <button key={s.id} type="button" onClick={() => setActive(i)} className="group text-left">
              <span className="block h-[2px] w-full overflow-hidden rounded-full bg-line dark:bg-white/10">
                <span
                  className="block h-full rounded-full transition-all duration-500 ease-out"
                  style={{ width: on ? '100%' : '0%', backgroundColor: s.color }}
                />
              </span>
              <span
                className="mt-3 block text-[11px] font-semibold tabular-nums tracking-[0.18em] transition-colors"
                style={{ color: on ? s.color : '#8E8E96' }}
              >
                {String(s.n).padStart(2, '0')}
              </span>
              <span
                className={`mt-1 block text-sm leading-snug transition-colors ${
                  on ? 'font-medium text-ink dark:text-neutral-100' : 'text-ink-2 group-hover:text-ink dark:text-neutral-400'
                }`}
              >
                {s.name}
              </span>
            </button>
          )
        })}
      </div>

      <div className="surface overflow-hidden">
        <AnimatePresence mode="wait">
          <motion.div
            key={stage.id}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.24, ease: 'easeOut' }}
            className="grid items-center gap-10 p-8 md:grid-cols-2 md:p-12"
          >
            <div>
              <p className="eyebrow" style={{ color: stage.color }}>
                Stage {String(stage.n).padStart(2, '0')}
              </p>
              <h3 className="mt-3 text-2xl font-medium tracking-[-0.02em] text-ink dark:text-neutral-100">
                {stage.name}
              </h3>
              <p className="mt-4 text-[15px] leading-relaxed text-ink dark:text-neutral-200">{stage.blurb}</p>
              <p className="mt-3 text-[15px] leading-relaxed text-ink-2 dark:text-neutral-400">{stage.detail}</p>
            </div>
            <div className="flex items-center justify-center rounded-xl bg-ground p-6 dark:bg-white/[0.03]">
              <StageArt id={stage.id} color={stage.color} />
            </div>
          </motion.div>
        </AnimatePresence>
      </div>

      <div className="mt-5 flex items-center justify-between">
        <button
          type="button"
          onClick={() => setActive((a) => Math.max(0, a - 1))}
          disabled={active === 0}
          className="text-sm text-ink-2 transition-colors hover:text-ink disabled:pointer-events-none disabled:opacity-30 dark:text-neutral-400"
        >
          ← Previous
        </button>
        <span className="text-xs tabular-nums text-ink-3">
          {active + 1} / {STAGES.length}
        </span>
        <button
          type="button"
          onClick={() => setActive((a) => Math.min(STAGES.length - 1, a + 1))}
          disabled={active === STAGES.length - 1}
          className="text-sm text-ink-2 transition-colors hover:text-ink disabled:pointer-events-none disabled:opacity-30 dark:text-neutral-400"
        >
          Next →
        </button>
      </div>
    </div>
  )
}
