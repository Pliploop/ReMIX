import { useState } from 'react'
import { AgreementBar, AcceptByQuestion, AxesBar, ChainLengthBar, GenreDonut, TransitionArea } from './Charts.jsx'
import { STAGE_BY_ID } from '../theme.js'

const DS_COLOR = { music4all: STAGE_BY_ID.neighbour.color, mtg_jamendo: STAGE_BY_ID.instruct.color }

export function DatasetToggle({ datasets, index, onChange }) {
  return (
    <div className="segmented">
      {datasets.map((d, i) => (
        <button key={d.key} type="button" data-on={i === index} onClick={() => onChange(i)}>
          {d.label}
        </button>
      ))}
    </div>
  )
}

function Figure({ value, label, color }) {
  return (
    <div className="surface px-5 py-5">
      <span className="block h-[3px] w-4 rounded-full" style={{ backgroundColor: color }} />
      <p className="mt-3 text-3xl font-medium tabular-nums tracking-[-0.03em] text-ink dark:text-neutral-100">{value}</p>
      <p className="mt-1 text-xs text-ink-2 dark:text-neutral-400">{label}</p>
    </div>
  )
}

const fmt = (n) => (n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : `${n}`)

export function DatasetStats({ stats, dark }) {
  const [i, setI] = useState(0)
  const ds = stats.datasets[i]
  if (!ds) return null
  const color = DS_COLOR[ds.key] ?? '#8E4EC6'
  const o = ds.overview

  return (
    <div>
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <DatasetToggle datasets={stats.datasets} index={i} onChange={setI} />
        <p className="text-xs tabular-nums text-ink-3">
          {o.clips.toLocaleString()} clips · {o.artists.toLocaleString()} artists
        </p>
      </div>

      <div className="mb-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Figure value={fmt(o.chains)} label="Chains" color={STAGE_BY_ID.chain.color} />
        <Figure value={fmt(o.steps)} label="Steps" color={STAGE_BY_ID.neighbour.color} />
        <Figure value={fmt(o.variants)} label="Instruction variants" color={STAGE_BY_ID.instruct.color} />
        <Figure value={`${o.median_instruction_words}`} label="Median words per instruction" color={STAGE_BY_ID.validate.color} />
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <GenreDonut data={ds.genre} dark={dark} />
        <ChainLengthBar data={ds.chain_length} dark={dark} color={color} />
        <AxesBar data={ds.axes} dark={dark} />
        <TransitionArea data={ds.transition_score} dark={dark} color={color} />
      </div>
    </div>
  )
}

export function ValidationStats({ stats, dark }) {
  const [i, setI] = useState(0)
  const v = stats.validation?.[i]
  if (!v) return null

  const overall = v.accept_by_question.find((r) => r.question === 'Overall valid')
  const meanAc1 = v.agreement.length
    ? v.agreement.reduce((a, r) => a + (r.ac1 ?? 0), 0) / v.agreement.length
    : 0

  return (
    <div>
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <DatasetToggle datasets={stats.validation} index={i} onChange={setI} />
        <p className="text-xs tabular-nums text-ink-3">
          {v.judges.join(' vs ')} · accept = score ≥ {stats.accept_threshold}
        </p>
      </div>

      <div className="mb-5 grid grid-cols-2 gap-3 sm:grid-cols-3">
        {v.judges.map((j, k) => (
          <Figure
            key={j}
            value={overall?.[j] != null ? `${overall[j]}%` : '—'}
            label={`${j} accepts overall`}
            color={k === 0 ? STAGE_BY_ID.validate.color : STAGE_BY_ID.chain.color}
          />
        ))}
        <Figure value={meanAc1.toFixed(2)} label="Mean AC1 across questions" color={STAGE_BY_ID.neighbour.color} />
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <AcceptByQuestion data={v.accept_by_question} judges={v.judges} dark={dark} />
        <AgreementBar data={v.agreement} dark={dark} />
      </div>
    </div>
  )
}
