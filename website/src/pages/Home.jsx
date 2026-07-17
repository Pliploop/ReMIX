import { Suspense, lazy, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import Nav, { useActiveSection, useTheme } from '../components/Nav.jsx'
import Pipeline from '../components/Pipeline.jsx'
import ChainViewer from '../components/ChainViewer.jsx'
import Logo from '../components/Logo.jsx'
import VideoFigure from '../components/VideoFigure.jsx'
import ErrorBoundary from '../components/ErrorBoundary.jsx'
import { LINKS, hexToRgba } from '../theme.js'

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

function Section({ id, eyebrow, title, lede, children }) {
  return (
    <section id={id} className="scroll-mt-20 py-16 md:py-20">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.2 }}
        transition={{ duration: 0.45, ease: 'easeOut' }}
      >
        {eyebrow && (
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-stage-validate">{eyebrow}</p>
        )}
        <h2 className="mt-2 text-2xl font-semibold tracking-tight text-neutral-900 dark:text-neutral-100 md:text-3xl">
          {title}
        </h2>
        {lede && <p className="mt-3 max-w-2xl text-base leading-relaxed text-neutral-600 dark:text-neutral-400">{lede}</p>}
        <div className="mt-8">{children}</div>
      </motion.div>
    </section>
  )
}

function ChartFallback() {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {[0, 1].map((i) => (
        <div key={i} className="h-64 animate-pulse rounded-2xl border border-neutral-200 bg-neutral-50 dark:border-neutral-800 dark:bg-neutral-900/50" />
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
        className="inline-flex cursor-not-allowed items-center gap-2 rounded-full border border-dashed border-neutral-300 px-4 py-2 text-sm text-neutral-400 dark:border-neutral-700 dark:text-neutral-500"
        title={`${soon} — not public yet`}
      >
        {children}
        <span className="text-[10px] uppercase tracking-wide">soon</span>
      </span>
    )
  }
  return (
    <a href={href} target="_blank" rel="noreferrer" className="pill">
      {children}
    </a>
  )
}

function Stat({ value, label, color }) {
  return (
    <div
      className="rounded-2xl border px-5 py-4"
      style={{ borderColor: hexToRgba(color, 0.3), backgroundColor: hexToRgba(color, 0.05) }}
    >
      <p className="text-2xl font-semibold tracking-tight" style={{ color }}>{value}</p>
      <p className="mt-0.5 text-xs text-neutral-600 dark:text-neutral-400">{label}</p>
    </div>
  )
}

export default function Home() {
  const [dark, setDark] = useTheme()
  const [chains, setChains] = useState(null)
  const [stats, setStats] = useState(null)
  const active = useActiveSection(SECTIONS.map((s) => s.id))

  useEffect(() => {
    const get = (name) =>
      fetch(`${import.meta.env.BASE_URL}data/${name}`)
        .then((r) => (r.ok ? r.json() : null))
        .catch(() => null)
    get('chains.json').then(setChains)
    get('stats.json').then(setStats)
  }, [])

  return (
    <div className="min-h-screen bg-white text-neutral-900 antialiased transition-colors duration-300 dark:bg-neutral-950 dark:text-neutral-100">
      <Nav dark={dark} setDark={setDark} sections={SECTIONS} active={active} />

      <main className="mx-auto max-w-5xl px-4 sm:px-6">
        {/* Hero */}
        <section className="relative py-20 md:py-28">
          <div
            className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-72 opacity-70"
            style={{ background: 'radial-gradient(60% 60% at 50% 0%, rgba(123,63,242,0.12), transparent 70%)' }}
            aria-hidden
          />
          <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
            <Logo size={64} className="text-neutral-900 dark:text-neutral-100" />
            <h1 className="mt-6 text-4xl font-semibold tracking-tight md:text-6xl">
              Re<span className="text-stage-validate">MIX</span>
            </h1>
            <p className="mt-4 max-w-2xl text-lg leading-relaxed text-neutral-600 dark:text-neutral-300">
              A large-scale dataset of <strong className="font-semibold text-neutral-900 dark:text-neutral-100">multi-turn, compositional</strong> music
              retrieval — where finding a track is a conversation, not a single query.
            </p>

            <div className="mt-8 flex flex-wrap gap-2.5">
              <LinkPill href={LINKS.paper} soon="Paper">Paper</LinkPill>
              <LinkPill href={LINKS.huggingface} soon="Dataset">Dataset</LinkPill>
              <a href={LINKS.github} target="_blank" rel="noreferrer" className="pill">Code</a>
              <Link
                to="/explore"
                className="inline-flex items-center gap-2 rounded-full bg-neutral-900 px-4 py-2 text-sm font-medium text-white transition-transform hover:scale-[1.03] dark:bg-neutral-100 dark:text-neutral-900"
              >
                Explore the chains →
              </Link>
            </div>

            <div className="mt-12 grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Stat
                value={
                  stats
                    ? `${(stats.datasets.reduce((a, d) => a + d.overview.chains, 0) / 1000).toFixed(1)}k`
                    : '—'
                }
                label="Sampled chains"
                color="#1FA347"
              />
              <Stat value="2" label="Open catalogs" color="#E23B34" />
              <Stat value="5" label="Variants per step" color="#FB8B24" />
              <Stat value="2" label="LLM judges" color="#7B3FF2" />
            </div>
          </motion.div>
        </section>

        <div className="pb-4">
          <VideoFigure
            src={`${import.meta.env.BASE_URL}remix.mp4`}
            poster={`${import.meta.env.BASE_URL}remix-poster.jpg`}
          />
        </div>

        <Section
          id="idea"
          eyebrow="The idea"
          title="Nobody finds music in one shot."
          lede="You start somewhere close, then you steer. Make it punchier. Keep the vocals but brighten it. Actually, go back to that piano from before. Retrieval benchmarks almost never test this — they ask one question and stop."
        >
          <div className="grid gap-4 md:grid-cols-2">
            <div className="stage-card border-neutral-200 bg-neutral-50 dark:border-neutral-800 dark:bg-neutral-900/50">
              <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">Single-shot retrieval</p>
              <p className="mt-2 text-sm leading-relaxed text-neutral-600 dark:text-neutral-400">
                One query, one ranked list. If it is wrong, your only move is to write a different query and start over.
                Nothing carries across.
              </p>
            </div>
            <div
              className="stage-card"
              style={{ borderColor: hexToRgba('#1FA347', 0.35), backgroundColor: hexToRgba('#1FA347', 0.06) }}
            >
              <p className="text-xs font-semibold uppercase tracking-wide text-stage-chain">ReMIX</p>
              <p className="mt-2 text-sm leading-relaxed text-neutral-700 dark:text-neutral-300">
                Each turn is an <strong>edit</strong> on the last result, and instructions may refer back to any earlier turn.
                The chain — not the query — is the unit of retrieval.
              </p>
            </div>
          </div>
        </Section>

        <Section
          id="pipeline"
          eyebrow="How it is built"
          title="Five stages, fully automatic."
          lede="From open catalogs to a validated benchmark, with no human in the generation loop — humans only check the result."
        >
          <Pipeline />
        </Section>

        <Section
          id="chains"
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

        <footer className="border-t border-neutral-200 py-10 text-sm dark:border-neutral-800">
          <div className="flex flex-col gap-6 md:flex-row md:items-start md:justify-between">
            <div className="flex items-center gap-2.5">
              <Logo size={22} className="text-neutral-900 dark:text-neutral-100" />
              <span className="text-neutral-500 dark:text-neutral-400">
                ReMIX · Queen Mary University of London
              </span>
            </div>
            <div className="max-w-md text-xs leading-relaxed text-neutral-500 dark:text-neutral-400">
              <p>
                Audio is streamed from Jamendo (Creative Commons) and Spotify. No audio is hosted or redistributed by this
                site. MTG-Jamendo tracks are credited to their artists under their individual licences.
              </p>
            </div>
          </div>
        </footer>
      </main>
    </div>
  )
}
