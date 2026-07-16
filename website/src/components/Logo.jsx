export default function Logo({ size = 32, className = '' }) {
  return (
    <svg
      viewBox="0 0 64 64"
      width={size}
      height={size}
      className={className}
      role="img"
      aria-label="ReMIX"
    >
      <g>
        <rect x="8" y="44" width="6" height="12" rx="3" fill="#E23B34" />
        <rect x="17" y="36" width="6" height="20" rx="3" fill="#2E6FD6" />
        <rect x="26" y="30" width="6" height="26" rx="3" fill="#1FA347" />
        <rect x="35" y="38" width="6" height="18" rx="3" fill="#FB8B24" />
        <rect x="44" y="43" width="6" height="13" rx="3" fill="#7B3FF2" />
      </g>
      <g
        fill="none"
        stroke="currentColor"
        strokeWidth="2.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path
          d="M46 4h12a4 4 0 0 1 4 4v6a4 4 0 0 1-4 4H50l-4 4v-4a4 4 0 0 1-4-4V8a4 4 0 0 1 4-4z"
          className="fill-white dark:fill-neutral-900"
        />
        <path d="M44 20C31 23 16 27 11 36" />
      </g>
      <path d="M11 40.5 7.4 34.2h7.2z" fill="currentColor" />
      <g stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
        <path d="M48 9h8" />
        <path d="M48 13h5" />
      </g>
    </svg>
  )
}

export function Wordmark({ className = '' }) {
  return (
    <span className={`font-semibold tracking-tight ${className}`}>
      Re<span className="text-stage-validate">MIX</span>
    </span>
  )
}
