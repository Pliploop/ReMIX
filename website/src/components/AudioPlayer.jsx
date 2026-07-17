import { useEffect, useRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'
import { clipWindow, hexToRgba } from '../theme.js'

/**
 * We never host audio. Two licence-clean paths, chosen per track by the exporter:
 *
 *   jamendo  -> the Jamendo CDN serves the CC-licensed original. It sends
 *               `access-control-allow-origin: *`, so we can fetch the bytes and
 *               draw a real waveform.
 *   spotify  -> Music4All is commercial music under a signed non-redistribution
 *               agreement, so playback goes through Spotify's own embed. No
 *               waveform is possible: the iframe is opaque to us by design.
 */
export default function AudioPlayer({ audio, accent = '#7B3FF2', clipId, compact = false }) {
  if (!audio || audio.kind === 'none') {
    return (
      <div className="rounded-xl border border-dashed border-neutral-300 px-4 py-3 text-xs text-neutral-500 dark:border-neutral-700 dark:text-neutral-400">
        No licence-clean audio source for this track.
      </div>
    )
  }
  if (audio.kind === 'spotify') return <SpotifyEmbed id={audio.id} clipId={clipId} />
  return <JamendoWave audio={audio} accent={accent} clipId={clipId} compact={compact} />
}

/**
 * Spotify's embed has a hard minimum height: below ~152px it keeps its own layout
 * and grows an internal scrollbar instead of shrinking. So we give it the full
 * 152 and never a "compact" height -- the iframe is opaque, so we cannot restyle
 * our way out of it, we can only stop squeezing it.
 */
function SpotifyEmbed({ id, clipId }) {
  const win = clipWindow(clipId)
  return (
    <div>
      <iframe
        title={`Spotify track ${id}`}
        src={`https://open.spotify.com/embed/track/${id}`}
        width="100%"
        height="152"
        scrolling="no"
        loading="lazy"
        allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
        className="block w-full rounded-xl border-0"
      />
      {win && (
        <p className="mt-1.5 text-[10px] text-neutral-500 dark:text-neutral-400">
          The pipeline analysed {win.start}–{win.end}s. Spotify’s player is opaque to us, so the
          window cannot be drawn on it.
        </p>
      )}
    </div>
  )
}

function fmt(t) {
  if (!Number.isFinite(t)) return '0:00'
  const m = Math.floor(t / 60)
  const s = Math.floor(t % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}

function JamendoWave({ audio, accent, clipId, compact }) {
  const holder = useRef(null)
  const ws = useRef(null)
  const [playing, setPlaying] = useState(false)
  const [ready, setReady] = useState(false)
  const [failed, setFailed] = useState(false)
  const [duration, setDuration] = useState(0)

  const win = clipWindow(clipId)

  useEffect(() => {
    if (!holder.current) return undefined
    const instance = WaveSurfer.create({
      container: holder.current,
      height: compact ? 44 : 56,
      waveColor: '#d4d4d8',
      progressColor: accent,
      cursorColor: accent,
      cursorWidth: 1,
      barWidth: 2,
      barGap: 2,
      barRadius: 2,
      normalize: true,
    })
    ws.current = instance
    instance.on('ready', () => {
      setReady(true)
      setDuration(instance.getDuration())
    })
    instance.on('play', () => setPlaying(true))
    instance.on('pause', () => setPlaying(false))
    instance.on('finish', () => setPlaying(false))
    instance.on('error', () => setFailed(true))
    instance.load(audio.url)

    return () => {
      // Guard: destroy() throws if a fetch is still in flight for this instance.
      try {
        instance.destroy()
      } catch {
        /* already torn down */
      }
      ws.current = null
    }
  }, [audio.url, accent, compact])

  // Jamendo streams the whole track, but the pipeline only ever saw `win`. Shade
  // everything outside it so what the model was judged on is unmistakable.
  // wavesurfer fits the full duration to the container width, so time maps to a
  // straight percentage -- no need for the regions plugin to place this.
  const pct = (t) => `${Math.min(100, Math.max(0, (t / duration) * 100))}%`
  const showWindow = ready && win && duration > 0 && win.end < duration

  const seekTo = (t) => {
    if (ws.current && duration > 0) ws.current.seekTo(Math.min(1, Math.max(0, t / duration)))
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => ws.current?.playPause()}
          disabled={!ready || failed}
          aria-label={playing ? 'Pause' : 'Play'}
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-white shadow-sm transition-transform duration-150 hover:scale-105 disabled:opacity-40"
          style={{ backgroundColor: accent }}
        >
          {!ready && !failed ? (
            <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white border-t-transparent" />
          ) : playing ? (
            <svg className="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 24 24">
              <path d="M6 4h4v16H6zM14 4h4v16h-4z" />
            </svg>
          ) : (
            <svg className="ml-0.5 h-3.5 w-3.5" fill="currentColor" viewBox="0 0 24 24">
              <path d="M8 5v14l11-7z" />
            </svg>
          )}
        </button>

        <div className="relative min-w-0 flex-1">
          <div ref={holder} />

          {showWindow && (
            <div className="pointer-events-none absolute inset-0" aria-hidden>
              {/* Veil the un-analysed audio on both sides. */}
              <div
                className="absolute inset-y-0 left-0 rounded-l-sm bg-white/70 dark:bg-neutral-950/70"
                style={{ width: pct(win.start) }}
              />
              <div
                className="absolute inset-y-0 right-0 rounded-r-sm bg-white/70 dark:bg-neutral-950/70"
                style={{ left: pct(win.end) }}
              />
              {/* Bracket the window itself. */}
              <div
                className="absolute inset-y-0 rounded-sm border-x-2"
                style={{
                  left: pct(win.start),
                  width: pct(win.end - win.start),
                  borderColor: accent,
                  backgroundColor: hexToRgba(accent, 0.07),
                }}
              />
            </div>
          )}
        </div>
      </div>

      {showWindow && (
        <button
          type="button"
          onClick={() => seekTo(win.start)}
          className="flex items-center gap-1.5 text-[10px] text-neutral-500 transition-colors hover:text-neutral-800 dark:text-neutral-400 dark:hover:text-neutral-200"
        >
          <span className="inline-block h-2 w-2 rounded-[2px]" style={{ backgroundColor: hexToRgba(accent, 0.5) }} />
          analysed {win.start}–{win.end}s of {fmt(duration)} · jump to clip
        </button>
      )}

      {failed && (
        <p className="text-xs text-neutral-500 dark:text-neutral-400">
          Stream unavailable —{' '}
          <a href={audio.page} target="_blank" rel="noreferrer" className="underline">
            play on Jamendo
          </a>
          .
        </p>
      )}
    </div>
  )
}

/** CC licences require attribution, so every Jamendo track carries its credit. */
export function Attribution({ track }) {
  const a = track?.audio
  if (a?.kind === 'jamendo') {
    return (
      <p className="text-[11px] leading-relaxed text-neutral-500 dark:text-neutral-400">
        <a href={a.page} target="_blank" rel="noreferrer" className="underline hover:text-neutral-800 dark:hover:text-neutral-200">
          {track.title}
        </a>{' '}
        by {track.artist} on Jamendo
        {a.license_url ? (
          <>
            {' · '}
            <a href={a.license_url} target="_blank" rel="noreferrer" className="underline hover:text-neutral-800 dark:hover:text-neutral-200">
              {a.license}
            </a>
          </>
        ) : (
          ` · ${a.license}`
        )}
      </p>
    )
  }
  if (a?.kind === 'spotify') {
    return (
      <p className="text-[11px] text-neutral-500 dark:text-neutral-400">
        Played via Spotify. Music4All audio is not redistributed.
      </p>
    )
  }
  return null
}
