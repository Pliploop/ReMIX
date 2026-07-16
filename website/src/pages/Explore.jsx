import Nav, { useTheme } from '../components/Nav.jsx'

// Placeholder. The sphere explorer lands here next: an instanced point cloud of
// every clip's audio embedding projected to 3D, chains drawn as great-circle
// arcs, and a floating glass card with the players.
export default function Explore() {
  const [dark, setDark] = useTheme()

  return (
    <div className="min-h-screen bg-white text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100">
      <Nav dark={dark} setDark={setDark} />
      <div className="mx-auto flex max-w-5xl flex-col items-center justify-center px-6 py-32 text-center">
        <h1 className="text-2xl font-semibold tracking-tight">Explorer</h1>
        <p className="mt-3 max-w-md text-sm text-neutral-600 dark:text-neutral-400">
          The chain explorer is being built: every track placed on a sphere by its audio embedding, with chains traced
          across the surface.
        </p>
      </div>
    </div>
  )
}
