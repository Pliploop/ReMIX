/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'Helvetica Neue', 'Helvetica', 'Arial', 'ui-sans-serif', 'system-ui'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      colors: {
        // Same tokens as the film (video/remix_video/theme.py).
        ground: { DEFAULT: '#FAFAF8', dark: '#0B0B0C' },
        line: '#E7E5E4',
        ink: { DEFAULT: '#18181B', 2: '#52525B', 3: '#8E8E96' },
        // The five pipeline stages. Accents only.
        stage: {
          enrich: '#E5484D',
          neighbour: '#3E63DD',
          chain: '#30A46C',
          instruct: '#F76B15',
          validate: '#8E4EC6',
        },
      },
      boxShadow: {
        soft: '0 1px 2px rgba(24,24,27,0.04), 0 10px 30px -14px rgba(24,24,27,0.12)',
        lift: '0 1px 2px rgba(24,24,27,0.05), 0 24px 60px -24px rgba(24,24,27,0.22)',
      },
      letterSpacing: {
        display: '-0.035em',
      },
      keyframes: {
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        'fade-up': 'fade-up 0.5s ease-out both',
      },
    },
  },
  plugins: [],
}
