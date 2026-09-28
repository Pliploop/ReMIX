import { Suspense, lazy, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import Nav, { useActiveSection, useTheme } from '../components/Nav.jsx'
import Pipeline from '../components/Pipeline.jsx'
import ChainViewer from '../components/ChainViewer.jsx'
import Logo from '../components/Logo.jsx'
import VideoFigure from '../components/VideoFigure.jsx'
import ErrorBoundary from '../components/ErrorBoundary.jsx'
import { LINKS, STAGES } from '../theme.js'

// Recharts is ~100KB and lives below the fold; keep it out of the landing chunk.
const DatasetStats = lazy(() =>
  import('../components/Stats.jsx').then((m) => ({ default: m.DatasetStats })),
)
const ValidationStats = lazy(() =>
  import('../components/Stats.jsx').then((m) => ({ default: m.ValidationStats })),
)

const SECTIONS = [
  { id: 'idea', label: 'Idea' },
  { id: 'pipeline', label: 'Pipeline' },
  { id: 'chains', label: 'Chains' },
  { id: 'stats', label: 'Dataset' },
  { id: 'validation', label: 'Validation' },
]

function Section({ id, n, eyebrow, title, lede, children }) {
  return (
    <section id={id} className="scroll-mt-20 border-t border-line py-20 dark:border-white/[0.08] md:py-28">
      <motion.div
        initial={{ opacity: 0, y: 14 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.15 }}
        transition={{ duration: 0.45, ease: 'easeOut' }}
      >
        <div className="grid gap-6 md:grid-cols-[180px_1fr] md:gap-10">
          <p className="eyebrow pt-2">
            <span className="tabular-nums text-ink-3">{String(n).padStart(2, '0')}</span>
            <span className="mx-2 text-ink-3">/</span>
            {eyebrow}
          </p>
          <div>
            <h2 className="text-3xl font-medium tracking-[-0.025em] text-ink dark:text-neutral-100 md:text-[44px] md:leading-[1.08]">
              {title}
            </h2>
            {lede && (
              <p className="mt-5 max-w-2xl text-[17px] leading-relaxed text-ink-2 dark:text-neutral-400">{lede}</p>
            )}
          </div>
        </div>
        <div className="mt-12">{children}</div>
      </motion.div>
    </section>
  )
}

function ChartFallback() {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {[0, 1].map((i) => (
        <div key={i} className="surface h-72 animate-pulse" />
      ))}
    </div>
  )
}

/**
 * A release link, or an honest disabled chip when there is nothing to link to yet.
 * `LINKS.paper`/`huggingface` are null until publication -- rendering them as
 * `href="#"` would look live and then dead-end the reader.
 */
function LinkPill({ href, children, soon }) {
  if (!href) {
    return (
      <span
        className="inline-flex cursor-not-allowed items-center gap-2 rounded-full border border-dashed border-line px-4 py-2.5 text-sm text-ink-3 dark:border-white/10 dark:text-neutral-500"
        title={`${soon} — not public yet`}
      >
        {children}
        <span className="text-[10px] font-semibold uppercase tracking-[0.14em]">soon</span>
      </span>
    )
  }
  return (
    <a href={href} target="_blank" rel="noreferrer" className="pill">
      {children}
    </a>
  )
}

/** One figure in the hero's row: accent tick, big ink number, quiet label. */
function Stat({ value, label, color }) {
  return (
    <div className="border-t border-line pt-5 dark:border-white/[0.08]">
      <span className="block h-[3px] w-5 rounded-full" style={{ backgroundColor: color }} />
      <p className="mt-4 text-4xl font-medium tabular-nums tracking-[-0.03em] text-ink dark:text-neutral-100 md:text-5xl">
        {value}
      </p>
      <p className="mt-2 text-sm text-ink-2 dark:text-neutral-400">{label}</p>
    </div>
  )
}

const compact = (n) =>
  n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}k` : `${n}`

export default function Home() {
  const [dark, setDark] = useTheme()
  const [chains, setChains] = useState(null)
  const [stats, setStats] = useState(null)
  const active = useActiveSection(SECTIONS.map((s) => s.id))
  const total = stats
    ? ['chains', 'steps', 'variants'].reduce(
        (o, k) => ({ ...o, [k]: stats.datasets.reduce((a, d) => a + d.overview[k], 0) }),
        {},
      )
    : null

  useEffect(() => {
    const get = (name) =>
      fetch(`${import.meta.env.BASE_URL}data/${name}`)
        .then((r) => (r.ok ? r.json() : null))
        .catch(() => null)
    get('chains.json').then(setChains)
    get('stats.json').then(setStats)
  }, [])

  return (
    <div className="min-h-screen bg-ground text-ink antialiased transition-colors duration-300 dark:bg-ground-dark dark:text-neutral-100">
      <Nav dark={dark} setDark={setDark} sections={SECTIONS} active={active} />

      <main className="mx-auto max-w-6xl px-5 sm:px-8">
        {/* Hero */}
        <section className="relative pb-16 pt-20 md:pb-24 md:pt-32">
          <div className="dot-field pointer-events-none absolute inset-x-[-10%] top-0 -z-10 h-[560px]" aria-hidden />
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, ease: 'easeOut' }}>
            <div className="inline-flex items-center gap-2.5 rounded-full border border-line bg-white/80 py-1 pl-1.5 pr-3.5 text-xs text-ink-2 shadow-soft backdrop-blur dark:border-white/10 dark:bg-neutral-900/70 dark:text-neutral-300">
              <span className="flex h-6 w-6 items-center justify-center rounded-full bg-ground dark:bg-neutral-800">
                <Logo size={15} className="text-ink dark:text-neutral-100" />
              </span>
              <span className="font-medium text-ink dark:text-neutral-100">ReMIX</span>
              <span className="text-ink-3">·</span>
              A dataset and benchmark for composed music retrieval
            </div>

            <h1 className="mt-8 max-w-4xl text-[44px] font-medium leading-[1.02] tracking-display text-ink dark:text-neutral-50 md:text-[80px]">
              Finding music is a conversation.
            </h1>
            <p className="mt-7 max-w-2xl text-lg leading-relaxed text-ink-2 dark:text-neutral-400 md:text-xl">
              ReMIX grounds every instruction in a real transition between two tracks, so a retriever can learn to
              follow edits like <span className="text-ink dark:text-neutral-100">“keep the vocals, make it heavier”</span> —
              one turn after another.
            </p>

            <div className="mt-10 flex flex-wrap items-center gap-3">
              <Link to="/explore" className="btn-primary">
                Explore the chains
                <span aria-hidden>→</span>
              </Link>
              <LinkPill href={LINKS.paper} soon="Paper">Paper</LinkPill>
              <LinkPill href={LINKS.huggingface} soon="Dataset">Dataset</LinkPill>
              <a href={LINKS.github} target="_blank" rel="noreferrer" className="pill">Code</a>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.15, ease: 'easeOut' }}
            className="mt-20 grid grid-cols-2 gap-x-8 gap-y-10 md:grid-cols-4"
          >
            <Stat value={total ? compact(total.chains) : '—'} label="conversational chains" color={STAGES[2].color} />
            <Stat value={total ? compact(total.steps) : '—'} label="grounded transitions" color={STAGES[1].color} />
            <Stat value={total ? compact(total.variants) : '—'} label="instruction variants" color={STAGES[3].color} />
            <Stat value="2" label="open catalogues" color={STAGES[0].color} />
          </motion.div>
        </section>

        <div className="pb-20 md:pb-28">
          <VideoFigure
            src={`${import.meta.env.BASE_URL}remix.mp4`}
            poster={`${import.meta.env.BASE_URL}remix-poster.jpg`}
          />
        </div>

        <Section
          id="idea"
          n={1}
          eyebrow="The idea"
          title="Nobody finds music in one shot."
          lede="You start somewhere close, then you steer. Make it punchier. Keep the vocals but brighten it. Actually, go back to that piano from before. Retrieval benchmarks almost never test this — they ask one question and stop."
        >
          <div className="grid gap-5 md:grid-cols-2">
            <div className="surface p-7">
              <p className="eyebrow">Single-shot retrieval</p>
              <p className="mt-4 text-[15px] leading-relaxed text-ink-2 dark:text-neutral-400">
                One query, one ranked list. If it is wrong, your only move is to write a different query and start over.
                Nothing carries across.
              </p>
            </div>
            <div className="surface relative overflow-hidden p-7">
              <span className="absolute inset-y-0 left-0 w-[3px] bg-stage-chain" aria-hidden />
              <p className="eyebrow text-stage-chain dark:text-stage-chain">ReMIX</p>
              <p className="mt-4 text-[15px] leading-relaxed text-ink dark:text-neutral-200">
                Each turn is an <span className="font-medium">edit</span> on the last result, and instructions may refer
                back to any earlier turn. The chain — not the query — is the unit of retrieval.
              </p>
            </div>
          </div>
        </Section>

        <Section
          id="pipeline"
          n={2}
          eyebrow="How it is built"
          title="Five stages, fully automatic."
          lede="From open catalogs to a validated benchmark, with no human in the generation loop — humans only check the result."
        >
          <Pipeline />
        </Section>

        <Section
          id="chains"
          n={3}
          eyebrow="See it"
          title="Real chains from the dataset."
          lede="Every chain below passed both LLM judges at every turn. Audio streams from Jamendo and Spotify — we redistribute none of it."
        >
          <ErrorBoundary label="Chain viewer">
            {chains ? (
              <ChainViewer data={chains} />
            ) : (
              <div className="rounded-2xl border border-dashed border-neutral-300 p-10 text-center text-sm text-neutral-500 dark:border-neutral-700">
                Loading chains…
              </div>
            )}
          </ErrorBoundary>
        </Section>

        <Section
          id="stats"
          n={4}
          eyebrow="What is inside"
          title="Two open catalogs, one recipe."
          lede="ReMIX is built over Music4All and MTG-Jamendo. Same pipeline, same instruction grammar, two very different musical distributions."
        >
          <ErrorBoundary label="Dataset charts">
            {stats ? (
              <Suspense fallback={<ChartFallback />}>
                <DatasetStats stats={stats} dark={dark} />
              </Suspense>
            ) : (
              <ChartFallback />
            )}
          </ErrorBoundary>
        </Section>

        <Section
          id="validation"
          n={5}
          eyebrow="Does it hold up?"
          title="Two judges, one rubric, one gate."
          lede="Every instruction variant is scored by Qwen3.6-27B and Gemma-4-31B against the same rubric a human rater sees. Only variants that pass become part of ReMIX."
        >
          <ErrorBoundary label="Validation charts">
            {stats ? (
              <Suspense fallback={<ChartFallback />}>
                <ValidationStats stats={stats} dark={dark} />
              </Suspense>
            ) : (
              <ChartFallback />
            )}
          </ErrorBoundary>
        </Section>

        <footer className="border-t border-line py-14 text-sm dark:border-white/[0.08]">
          <div className="flex flex-col gap-10 md:flex-row md:items-start md:justify-between">
            <div>
              <div className="flex items-center gap-2.5">
                <Logo size={22} className="text-ink dark:text-neutral-100" />
                <span className="font-medium text-ink dark:text-neutral-100">ReMIX</span>
              </div>
              <p className="mt-3 text-ink-2 dark:text-neutral-400">Queen Mary University of London</p>
            </div>
            <p className="max-w-md text-xs leading-relaxed text-ink-3 dark:text-neutral-500">
              Audio is streamed from Jamendo (Creative Commons) and Spotify. No audio is hosted or redistributed by this
              site. MTG-Jamendo tracks are credited to their artists under their individual licences.
            </p>
          </div>
        </footer>
      </main>
    </div>
  )
}
