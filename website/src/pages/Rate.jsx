import { useCallback, useEffect, useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import Nav, { useTheme } from '../components/Nav.jsx'
import ErrorBoundary from '../components/ErrorBoundary.jsx'
import AudioPlayer, { Attribution } from '../components/AudioPlayer.jsx'
import { hexToRgba } from '../theme.js'

// Empty on the GitHub Pages build (static, no backend); "/api" on the Space.
const API = import.meta.env.VITE_API_BASE ?? ''

/**
 * Rater identity: a random id kept in localStorage. Deliberately not an IP --
 * an IP is personal data under UK GDPR, and this needs no personal data at all.
 * It only has to tell raters apart for inter-rater agreement, which a stable
 * per-browser id does. It cannot link one person across devices; that is fine.
 */
function useRaterId() {
  const [id, setId] = useState('')
  useEffect(() => {
    let v = localStorage.getItem('remix-rater-id')
    if (!v) {
      v = `r_${(crypto.randomUUID?.() ?? Math.random().toString(36).slice(2)).replace(/-/g, '').slice(0, 16)}`
      localStorage.setItem('remix-rater-id', v)
    }
    setId(v)
  }, [])
  return id
}

function useDone(key) {
  const [done, setDone] = useState(() => new Set())
  useEffect(() => {
    try {
      setDone(new Set(JSON.parse(localStorage.getItem(`remix-done-${key}`) || '[]')))
    } catch {
      setDone(new Set())
    }
  }, [key])
  const mark = useCallback(
    (id) => {
      setDone((prev) => {
        const next = new Set(prev).add(id)
        localStorage.setItem(`remix-done-${key}`, JSON.stringify([...next]))
        return next
      })
    },
    [key],
  )
  return [done, mark]
}

function Segmented({ options, value, onChange, extra = [] }) {
  const all = [...options.map((o) => o.label), ...extra]
  return (
    <div className="flex flex-wrap gap-1.5">
      {all.map((label) => {
        const on = value === label
        const isExtra = extra.includes(label)
        return (
          <button
            key={label}
            type="button"
            onClick={() => onChange(on ? null : label)}
            className={`rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-all ${
              on
                ? 'border-transparent text-white shadow-sm'
                : isExtra
                  ? 'border-neutral-200 text-neutral-400 hover:border-neutral-400 dark:border-neutral-700 dark:text-neutral-500'
                  : 'border-neutral-200 text-neutral-600 hover:border-neutral-900 hover:text-neutral-900 dark:border-neutral-700 dark:text-neutral-400 dark:hover:border-neutral-300 dark:hover:text-neutral-100'
            }`}
            style={on ? { backgroundColor: isExtra ? '#71717a' : '#7B3FF2' } : undefined}
          >
            {label}
          </button>
        )
      })}
    </div>
  )
}

function TrackPanel({ track, role, color }) {
  const [showCaption, setShowCaption] = useState(false)
  return (
    <div
      className="rounded-2xl border p-4"
      style={{ borderColor: hexToRgba(color, 0.35), backgroundColor: hexToRgba(color, 0.05) }}
    >
      <div className="flex items-center gap-2">
        <span className="chip font-semibold text-white" style={{ backgroundColor: color }}>
          {role}
        </span>
        <p className="truncate text-sm font-semibold text-neutral-900 dark:text-neutral-100">{track.title}</p>
      </div>
      <p className="mt-0.5 truncate text-xs text-neutral-600 dark:text-neutral-400">{track.artist}</p>

      <div className="mt-3">
        <AudioPlayer audio={track.audio} accent={color} compact />
      </div>
      <div className="mt-1.5">
        <Attribution track={track} />
      </div>

      {track.tags?.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1">
          {track.tags.map((t) => (
            <span key={t} className="chip bg-white/70 text-[10px] text-neutral-700 ring-1 ring-inset ring-neutral-200 dark:bg-neutral-800/60 dark:text-neutral-300 dark:ring-neutral-700">
              {t}
            </span>
          ))}
        </div>
      )}

      {track.caption && (
        <div className="mt-3">
          <button
            type="button"
            onClick={() => setShowCaption((v) => !v)}
            className="text-[11px] font-medium text-neutral-500 underline-offset-2 hover:underline dark:text-neutral-400"
          >
            {showCaption ? 'Hide' : 'Show'} description
          </button>
          <AnimatePresence>
            {showCaption && (
              <motion.p
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="overflow-hidden text-[11px] leading-relaxed text-neutral-600 dark:text-neutral-400"
              >
                <span className="mt-1.5 block">{track.caption}</span>
              </motion.p>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  )
}

export default function Rate() {
  const [dark, setDark] = useTheme()
  const raterId = useRaterId()
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [dsIdx, setDsIdx] = useState(0)
  const [idx, setIdx] = useState(0)
  const [answers, setAnswers] = useState({})
  const [issues, setIssues] = useState([])
  const [notes, setNotes] = useState('')
  const [sending, setSending] = useState(false)
  const [nickname, setNickname] = useState('')

  const dataset = data?.datasets?.[dsIdx]
  const [done, markDone] = useDone(dataset?.key ?? 'none')

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/validation_tasks.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setData)
      .catch(setError)
  }, [])

  useEffect(() => {
    setNickname(localStorage.getItem('remix-rater-name') || '')
  }, [])

  const items = dataset?.items ?? []

  /**
   * Queue order. The two calibration buckets want opposite treatment:
   *
   *   core_overlap - every rater must rate all 50 or there is no inter-rater
   *                  agreement to compute, so these go first.
   *   sentinel     - attention checks only work if a rater cannot tell when one
   *                  is coming, so they are spread through the run rather than
   *                  front-loaded (all-early checks miss a rater who drifts).
   *   full         - the bulk, filling the gaps.
   *
   * Rated items are not dropped, just moved to the back.
   */
  const order = useMemo(() => {
    const pick = (b) => items.map((_, i) => i).filter((i) => items[i].bucket === b)
    const overlap = pick('core_overlap')
    const sentinels = pick('sentinel')
    const full = pick('full')

    const queue = [...overlap]
    const every = Math.max(1, Math.ceil(full.length / Math.max(sentinels.length, 1)))
    let s = 0
    full.forEach((idx, n) => {
      queue.push(idx)
      if (s < sentinels.length && (n + 1) % every === 0) queue.push(sentinels[s++])
    })
    while (s < sentinels.length) queue.push(sentinels[s++])

    const pending = queue.filter((i) => !done.has(items[i].assignment_id))
    const rated = queue.filter((i) => done.has(items[i].assignment_id))
    return [...pending, ...rated]
  }, [items, done])

  const item = items[order[idx]]
  const rubric = data?.rubric
  const variant = item?.variants?.[0]

  const reset = () => {
    setAnswers({})
    setIssues([])
    setNotes('')
  }

  const complete = rubric?.questions.every((q) => answers[q.id]) ?? false

  const submit = async () => {
    if (!item || !variant) return
    setSending(true)
    const payload = {
      annotation_type: 'human_single_variant_rating',
      annotator_id: raterId,
      annotator_name: nickname || null,
      annotated_at_utc: new Date().toISOString(),
      dataset: dataset.key,
      assignment_id: item.assignment_id,
      chain_id: item.chain_id,
      turn_index: item.turn_index,
      variant_index: variant.variant_index,
      instruction_field: item.instruction_field,
      instruction: variant.instruction,
      bucket: item.bucket,
      is_sentinel: item.is_sentinel,
      split: item.split,
      stable_hash: item.stable_hash,
      source_clip_id: item.source.clip_id,
      target_clip_id: item.target.clip_id,
      modality: 'audio_and_text',
      audio_available: item.source.audio.kind !== 'none' && item.target.audio.kind !== 'none',
      answers: Object.fromEntries(
        rubric.questions.map((q) => {
          const label = answers[q.id]
          const opt = q.options.find((o) => o.label === label)
          return [
            q.id,
            {
              label: label ?? null,
              score: opt?.score ?? null,
              cannot_judge: label === rubric.cannot_judge_label,
              not_applicable: label === rubric.not_applicable_label,
            },
          ]
        }),
      ),
      issue_tags: issues,
      notes: notes.trim() || null,
    }

    try {
      // No localStorage fallback: this route only exists when a backend is
      // configured (see main.jsx). A "saved" rating that never leaves the
      // browser is indistinguishable from a real one until the data is missing.
      const res = await fetch(`${API}/ratings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      markDone(item.assignment_id)
      reset()
      setIdx((i) => Math.min(i + 1, items.length - 1))
    } catch (e) {
      setError(e)
    } finally {
      setSending(false)
    }
  }

  if (error) {
    return (
      <Shell dark={dark} setDark={setDark}>
        <div className="rounded-2xl border border-dashed border-red-300 p-8 text-center text-sm text-red-600 dark:border-red-900">
          Could not load rating tasks: {String(error.message ?? error)}
        </div>
      </Shell>
    )
  }
  if (!data || !item || !rubric) {
    return (
      <Shell dark={dark} setDark={setDark}>
        <div className="flex justify-center py-24">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-neutral-300 border-t-stage-validate" />
        </div>
      </Shell>
    )
  }

  const doneCount = items.filter((it) => done.has(it.assignment_id)).length

  return (
    <Shell dark={dark} setDark={setDark}>
      {/* header */}
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <div className="inline-flex rounded-full border border-neutral-200 p-1 dark:border-neutral-700">
          {data.datasets.map((d, i) => (
            <button
              key={d.key}
              type="button"
              onClick={() => {
                setDsIdx(i)
                setIdx(0)
                reset()
              }}
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
        <div className="flex items-center gap-3">
          <input
            value={nickname}
            onChange={(e) => {
              setNickname(e.target.value)
              localStorage.setItem('remix-rater-name', e.target.value)
            }}
            placeholder="name (optional)"
            className="w-32 rounded-lg border border-neutral-200 bg-transparent px-2.5 py-1.5 text-xs outline-none focus:border-stage-validate dark:border-neutral-700"
          />
          <span className="text-xs text-neutral-500 dark:text-neutral-400">
            {doneCount}/{items.length} rated
          </span>
        </div>
      </div>

      <div className="mb-6 h-1 w-full overflow-hidden rounded-full bg-neutral-100 dark:bg-neutral-800">
        <motion.div
          className="h-full rounded-full bg-stage-validate"
          animate={{ width: `${(doneCount / Math.max(items.length, 1)) * 100}%` }}
          transition={{ duration: 0.4 }}
        />
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={item.assignment_id}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.22 }}
        >
          {/* the pair + the instruction between them */}
          <div className="grid gap-3 md:grid-cols-2">
            <TrackPanel track={item.source} role="Source" color="#1FA347" />
            <TrackPanel track={item.target} role="Target" color="#2E6FD6" />
          </div>

          <div
            className="my-4 rounded-2xl border px-5 py-4"
            style={{ borderColor: hexToRgba('#FB8B24', 0.45), backgroundColor: hexToRgba('#FB8B24', 0.07) }}
          >
            <p className="text-[11px] font-semibold uppercase tracking-wide text-stage-instruct">
              The instruction
            </p>
            <p className="mt-1.5 text-lg font-medium leading-snug text-neutral-900 dark:text-neutral-100">
              “{variant.instruction}”
            </p>
          </div>

          {/* rubric */}
          <div className="space-y-5">
            {rubric.questions.map((q, i) => (
              <div key={q.id}>
                <p className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                  <span className="mr-1.5 text-xs text-neutral-400">{i + 1}.</span>
                  {q.statement}
                </p>
                {q.help && (
                  <p className="mb-2 mt-0.5 text-[11px] leading-relaxed text-neutral-500 dark:text-neutral-400">
                    {q.help}
                  </p>
                )}
                <Segmented
                  options={q.options}
                  value={answers[q.id] ?? null}
                  onChange={(v) => setAnswers((a) => ({ ...a, [q.id]: v }))}
                  extra={q.allow_na ? [rubric.not_applicable_label, rubric.cannot_judge_label] : [rubric.cannot_judge_label]}
                />
              </div>
            ))}

            <div>
              <p className="mb-2 text-sm font-medium text-neutral-900 dark:text-neutral-100">
                Anything wrong with it? <span className="text-xs font-normal text-neutral-400">(optional)</span>
              </p>
              <div className="flex flex-wrap gap-1.5">
                {rubric.issue_tags.map((t) => {
                  const on = issues.includes(t)
                  return (
                    <button
                      key={t}
                      type="button"
                      onClick={() => setIssues((v) => (on ? v.filter((x) => x !== t) : [...v, t]))}
                      className={`rounded-full border px-2.5 py-1 text-[11px] transition-colors ${
                        on
                          ? 'border-transparent bg-stage-enrich text-white'
                          : 'border-neutral-200 text-neutral-500 hover:border-neutral-400 dark:border-neutral-700 dark:text-neutral-400'
                      }`}
                    >
                      {t}
                    </button>
                  )
                })}
              </div>
            </div>

            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Notes (optional)"
              rows={2}
              className="w-full rounded-xl border border-neutral-200 bg-transparent px-3 py-2 text-sm outline-none focus:border-stage-validate dark:border-neutral-700"
            />
          </div>

          <div className="sticky bottom-0 mt-6 flex items-center justify-between gap-3 border-t border-neutral-200 bg-white/80 py-4 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
            <button
              type="button"
              onClick={() => {
                reset()
                setIdx((i) => Math.min(i + 1, items.length - 1))
              }}
              className="pill text-xs"
            >
              Skip
            </button>
            <div className="flex items-center gap-3">
              {!complete && (
                <span className="text-xs text-neutral-400">
                  {rubric.questions.filter((q) => !answers[q.id]).length} left
                </span>
              )}
              <button
                type="button"
                onClick={submit}
                disabled={!complete || sending}
                className="rounded-full bg-stage-validate px-5 py-2 text-sm font-medium text-white transition-transform hover:scale-[1.03] disabled:cursor-not-allowed disabled:opacity-30"
              >
                {sending ? 'Saving…' : 'Submit & next'}
              </button>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>

    </Shell>
  )
}

function Shell({ dark, setDark, children }) {
  return (
    <div className="min-h-screen bg-white text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100">
      <Nav dark={dark} setDark={setDark} />
      <main className="mx-auto max-w-3xl px-4 py-8 sm:px-6">
        <ErrorBoundary label="Rating app">{children}</ErrorBoundary>
      </main>
    </div>
  )
}
