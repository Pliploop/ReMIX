import { useEffect, useState } from 'react'

/**
 * A LaTeX formula, typeset by KaTeX.
 *
 * KaTeX plus its fonts is ~300KB and only two formulas on the site need it, so it
 * is imported dynamically rather than pulled into the landing chunk. Until it
 * lands -- and under SSR, where the effect never runs -- we render `fallback`,
 * a plain-text spelling of the same expression. That keeps the meaning present
 * with no layout jump and no hard dependency on the load succeeding.
 */
export default function Formula({ tex, fallback, className = '' }) {
  const [html, setHtml] = useState(null)

  useEffect(() => {
    let cancelled = false
    Promise.all([import('katex'), import('katex/dist/katex.min.css')])
      .then(([katex]) => {
        if (cancelled) return
        setHtml(
          katex.default.renderToString(tex, {
            throwOnError: false,
            displayMode: false,
          }),
        )
      })
      .catch(() => {
        /* keep the fallback */
      })
    return () => {
      cancelled = true
    }
  }, [tex])

  if (!html) {
    return <span className={`font-mono text-[0.9em] ${className}`}>{fallback}</span>
  }
  return <span className={className} dangerouslySetInnerHTML={{ __html: html }} />
}
