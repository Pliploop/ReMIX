import { useRef, useState } from 'react'
import { motion } from 'framer-motion'

/**
 * The pipeline film, in the same visual grammar as the paper's main figure.
 *
 * Deliberately not autoplaying: the render is a few MB and carries no audio
 * track, so autoplay would spend a phone's data on a silent loop nobody asked
 * for. `preload="metadata"` fetches only enough to size the player and show the
 * poster frame; the bytes arrive when someone presses play.
 */
export default function VideoFigure({ src, poster }) {
  const video = useRef(null)
  const [started, setStarted] = useState(false)

  const play = () => {
    setStarted(true)
    video.current?.play()
  }

  return (
    <motion.figure
      initial={{ opacity: 0, y: 14 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.25 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className="group relative"
    >
      <div className="relative overflow-hidden rounded-[28px] border border-line bg-white p-1.5 shadow-lift dark:border-white/10 dark:bg-neutral-900 dark:shadow-black/40">
        <video
          ref={video}
          src={src}
          poster={poster}
          controls={started}
          preload="metadata"
          playsInline
          onPlay={() => setStarted(true)}
          className="block aspect-video w-full rounded-[22px] bg-ground dark:bg-neutral-950"
        />

        {!started && (
          <button
            type="button"
            onClick={play}
            aria-label="Play the pipeline walkthrough"
            className="absolute inset-0 flex items-center justify-center bg-neutral-950/[0.04] transition-colors hover:bg-neutral-950/10 dark:bg-neutral-950/20 dark:hover:bg-neutral-950/30"
          >
            <span className="flex h-[72px] w-[72px] items-center justify-center rounded-full bg-ink shadow-lift transition-transform duration-200 group-hover:scale-105 dark:bg-white">
              <svg className="ml-1 h-6 w-6 text-white dark:text-neutral-900" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z" />
              </svg>
            </span>
          </button>
        )}
      </div>

      <figcaption className="mt-4 flex items-center justify-between text-xs text-ink-3">
        <span>The full pipeline, from raw catalogue to validated benchmark.</span>
        <span className="tabular-nums">1:42 · no sound</span>
      </figcaption>
    </motion.figure>
  )
}
