import { useEffect, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import Logo, { Wordmark } from './Logo.jsx'

export function useTheme() {
  const [dark, setDark] = useState(false)

  useEffect(() => {
    const saved = localStorage.getItem('remix-theme')
    const prefers = window.matchMedia('(prefers-color-scheme: dark)').matches
    setDark(saved ? saved === 'dark' : prefers)
  }, [])

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    localStorage.setItem('remix-theme', dark ? 'dark' : 'light')
  }, [dark])

  return [dark, setDark]
}

/** Highlights whichever section is currently under the nav. */
export function useActiveSection(ids) {
  const [active, setActive] = useState(ids[0])
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)
        if (visible[0]) setActive(visible[0].target.id)
      },
      { rootMargin: '-20% 0px -60% 0px', threshold: [0.1, 0.4, 0.8] },
    )
    ids.forEach((id) => {
      const el = document.getElementById(id)
      if (el) observer.observe(el)
    })
    return () => observer.disconnect()
  }, [ids.join(',')])
  return active
}

/**
 * Scroll to a section without touching the URL hash.
 *
 * A plain `href="#pipeline"` cannot work here: HashRouter keeps the route in the
 * hash (`#/`), so the browser rewriting it to `#pipeline` is read back as the
 * route "/pipeline", which matches nothing and blanks the page. So we intercept
 * the click and scroll ourselves. `html { scroll-behavior: smooth }` in index.css
 * does the easing; scroll-mt-20 on each section clears the sticky nav.
 */
export function scrollToSection(e, id) {
  const el = document.getElementById(id)
  if (!el) return
  e.preventDefault()
  el.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

export default function Nav({ dark, setDark, sections = [], active }) {
  const { pathname } = useLocation()

  return (
    <header className="sticky top-0 z-50 border-b border-line/80 bg-ground/75 backdrop-blur-xl dark:border-white/[0.08] dark:bg-ground-dark/75">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-5 sm:px-8">
        <Link to="/" className="group flex items-center gap-2.5">
          <Logo size={26} className="text-neutral-900 dark:text-neutral-100" />
          <Wordmark className="text-[17px] text-ink dark:text-neutral-100" />
        </Link>

        <nav className="hidden items-center gap-1 text-sm md:flex">
          {sections.map((s) => (
            <a
              key={s.id}
              href={`#${s.id}`}
              onClick={(e) => scrollToSection(e, s.id)}
              className={`rounded-full px-3.5 py-1.5 transition-colors ${
                active === s.id
                  ? 'bg-white font-medium text-ink shadow-soft ring-1 ring-line dark:bg-neutral-800 dark:text-neutral-100 dark:ring-white/10'
                  : 'text-ink-2 hover:text-ink dark:text-neutral-400 dark:hover:text-neutral-100'
              }`}
            >
              {s.label}
            </a>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          <Link
            to={pathname === '/explore' ? '/' : '/explore'}
            className="rounded-full bg-ink px-4 py-1.5 text-xs font-medium text-white transition-colors hover:bg-neutral-800 dark:bg-neutral-100 dark:text-neutral-900"
          >
            {pathname === '/explore' ? 'Home' : 'Explore'}
          </Link>
          <button
            type="button"
            onClick={() => setDark(!dark)}
            aria-label="Toggle theme"
            className="flex h-8 w-8 items-center justify-center rounded-full border border-line bg-white text-ink-2 transition-colors hover:text-ink dark:border-white/10 dark:bg-neutral-900 dark:text-neutral-300"
          >
            {dark ? (
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.36 6.36-.7-.7M6.34 6.34l-.7-.7m12.72 0-.7.7M6.34 17.66l-.7.7M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0z" />
              </svg>
            ) : (
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M20.354 15.354A9 9 0 0 1 8.646 3.646 9.003 9.003 0 0 0 12 21a9.003 9.003 0 0 0 8.354-5.646z" />
              </svg>
            )}
          </button>
        </div>
      </div>
    </header>
  )
}
