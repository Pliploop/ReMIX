import { useEffect, useRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'

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
export default function AudioPlayer({ audio, accent = '#7B3FF2', compact = false }) {
  if (!audio || audio.kind === 'none') {
    return (
      <div className="rounded-xl border border-dashed border-neutral-300 px-4 py-3 text-xs text-neutral-500 dark:border-neutral-700 dark:text-neutral-400">
        No licence-clean audio source for this track.
      </div>
    )
  }
  if (audio.kind === 'spotify') return <SpotifyEmbed id={audio.id} compact={compact} />
  return <JamendoWave audio={audio} accent={accent} />
}

function SpotifyEmbed({ id, compact }) {
  return (
    <iframe
      title={`Spotify track ${id}`}
      src={`https://open.spotify.com/embed/track/${id}`}
      width="100%"
      height={compact ? 80 : 152}
      frameBorder="0"
      loading="lazy"
      allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
      className="rounded-xl"
    />
  )
}

function JamendoWave({ audio, accent }) {
  const holder = useRef(null)
  const ws = useRef(null)
  const [playing, setPlaying] = useState(false)
  const [ready, setReady] = useState(false)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    if (!holder.current) return undefined
    const instance = WaveSurfer.create({
      container: holder.current,
      height: 48,
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
    instance.on('ready', () => setReady(true))
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
  }, [audio.url, accent])

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
        <div ref={holder} className="min-w-0 flex-1" />
      </div>
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
