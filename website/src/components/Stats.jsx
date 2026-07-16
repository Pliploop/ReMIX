import { useState } from 'react'
import { AgreementBar, AcceptByQuestion, AxesBar, ChainLengthBar, GenreDonut, TransitionArea } from './Charts.jsx'
import { hexToRgba } from '../theme.js'

const DS_COLOR = { music4all: '#2E6FD6', mtg_jamendo: '#FB8B24' }

export function DatasetToggle({ datasets, index, onChange }) {
  return (
    <div className="inline-flex rounded-full border border-neutral-200 p-1 dark:border-neutral-700">
      {datasets.map((d, i) => (
        <button
          key={d.key}
          type="button"
          onClick={() => onChange(i)}
          className={`rounded-full px-3.5 py-1.5 text-xs font-medium transition-colors ${
            i === index
              ? 'bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900'
              : 'text-neutral-600 hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-neutral-100'
          }`}
        >
          {d.label}
        </button>
      ))}
    </div>
  )
}

function Figure({ value, label, color }) {
  return (
    <div
      className="rounded-xl border px-4 py-3"
      style={{ borderColor: hexToRgba(color, 0.3), backgroundColor: hexToRgba(color, 0.05) }}
    >
      <p className="text-xl font-semibold tracking-tight" style={{ color }}>
        {value}
      </p>
      <p className="mt-0.5 text-[11px] text-neutral-600 dark:text-neutral-400">{label}</p>
    </div>
  )
}

const fmt = (n) => (n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : `${n}`)

export function DatasetStats({ stats, dark }) {
  const [i, setI] = useState(0)
  const ds = stats.datasets[i]
  if (!ds) return null
  const color = DS_COLOR[ds.key] ?? '#7B3FF2'
  const o = ds.overview

  return (
    <div>
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <DatasetToggle datasets={stats.datasets} index={i} onChange={setI} />
        <p className="text-xs text-neutral-500 dark:text-neutral-400">
          {o.clips.toLocaleString()} clips · {o.artists.toLocaleString()} artists
        </p>
      </div>

      <div className="mb-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Figure value={fmt(o.chains)} label="Chains" color="#1FA347" />
        <Figure value={fmt(o.steps)} label="Steps" color="#2E6FD6" />
        <Figure value={fmt(o.variants)} label="Instruction variants" color="#FB8B24" />
        <Figure value={`${o.median_instruction_words}`} label="Median words per instruction" color="#7B3FF2" />
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
        <p className="text-xs text-neutral-500 dark:text-neutral-400">
          {v.judges.join(' vs ')} · accept = score ≥ {stats.accept_threshold}
        </p>
      </div>

      <div className="mb-5 grid grid-cols-2 gap-3 sm:grid-cols-3">
        {v.judges.map((j, k) => (
          <Figure
            key={j}
            value={overall?.[j] != null ? `${overall[j]}%` : '—'}
            label={`${j} accepts overall`}
            color={k === 0 ? '#7B3FF2' : '#1FA347'}
          />
        ))}
        <Figure value={meanAc1.toFixed(2)} label="Mean AC1 across questions" color="#2E6FD6" />
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <AcceptByQuestion data={v.accept_by_question} judges={v.judges} dark={dark} />
        <AgreementBar data={v.agreement} dark={dark} />
      </div>
    </div>
  )
}
