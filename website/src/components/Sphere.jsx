import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Html, OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { SPHERE_RADIUS as RADIUS } from '../theme.js'

/** Great-circle (slerp) arc between two points on the sphere, lifted slightly clear of the surface. */
export function arcPoints(a, b, segments = 48, lift = 1.035) {
  const va = new THREE.Vector3(...a).normalize()
  const vb = new THREE.Vector3(...b).normalize()
  const out = []
  const dot = THREE.MathUtils.clamp(va.dot(vb), -1, 1)
  const omega = Math.acos(dot)
  for (let i = 0; i <= segments; i++) {
    const t = i / segments
    let v
    if (omega < 1e-6) {
      v = va.clone()
    } else {
      // Spherical linear interpolation: keeps the path on the great circle.
      const s0 = Math.sin((1 - t) * omega) / Math.sin(omega)
      const s1 = Math.sin(t * omega) / Math.sin(omega)
      v = va.clone().multiplyScalar(s0).add(vb.clone().multiplyScalar(s1))
    }
    // Bow the middle outward so the arc reads as a link, not a surface scratch.
    const bow = lift + 0.10 * Math.sin(Math.PI * t)
    out.push(v.normalize().multiplyScalar(RADIUS * bow))
  }
  return out
}

const lift = (p, k = 1.035) => new THREE.Vector3(...p).normalize().multiplyScalar(RADIUS * k)

/** All tracks as one instanced point cloud — 22k DOM nodes is impossible, 22k GPU points is free. */
function Cloud({ positions, dark, onHover, onPick }) {
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    return g
  }, [positions])

  useEffect(() => () => geom.dispose(), [geom])

  const last = useRef(-1)

  return (
    <points
      geometry={geom}
      onPointerMove={(e) => {
        e.stopPropagation()
        const i = e.index ?? -1
        // The pointer sweeps many points per second; only re-render on change.
        if (i !== last.current) {
          last.current = i
          onHover(i)
        }
      }}
      onPointerOut={() => {
        last.current = -1
        onHover(-1)
      }}
      onClick={(e) => {
        e.stopPropagation()
        if (e.index != null) onPick(e.index)
      }}
    >
      <pointsMaterial
        size={0.05}
        sizeAttenuation
        color={dark ? '#71717a' : '#9aa1ad'}
        transparent
        opacity={dark ? 0.9 : 0.95}
        depthWrite={false}
      />
    </points>
  )
}

/**
 * A bare wireframe cage — no opaque fill. The fill used to hide the far half of
 * the cloud, which is most of it; without it the corpus reads as a real 3D shell
 * and the chain arcs stay legible through it.
 */
function Globe({ dark }) {
  return (
    <mesh>
      <sphereGeometry args={[RADIUS * 0.985, 36, 24]} />
      <meshBasicMaterial
        color={dark ? '#a1a1aa' : '#18181b'}
        wireframe
        transparent
        opacity={dark ? 0.1 : 0.13}
        depthWrite={false}
      />
    </mesh>
  )
}

function ChainPath({ nodes, activeStep, dark }) {
  const arcs = useMemo(() => {
    const out = []
    for (let i = 0; i < nodes.length - 1; i++) out.push(arcPoints(nodes[i], nodes[i + 1]))
    return out
  }, [nodes])

  const geoms = useMemo(() => arcs.map((pts) => new THREE.BufferGeometry().setFromPoints(pts)), [arcs])
  useEffect(() => () => geoms.forEach((g) => g.dispose()), [geoms])

  return (
    <>
      {geoms.map((g, i) => {
        const on = i === activeStep
        return (
          <line key={i} geometry={g}>
            <lineBasicMaterial
              color={on ? '#FB8B24' : dark ? '#52525b' : '#b8bdc6'}
              transparent
              opacity={on ? 1 : 0.5}
            />
          </line>
        )
      })}
      {nodes.map((p, i) => {
        const isEnd = i === nodes.length - 1
        const on = i === activeStep || i === activeStep + 1
        const color = i === 0 ? '#1FA347' : isEnd ? '#7B3FF2' : '#2E6FD6'
        return (
          <mesh key={i} position={lift(p)}>
            <sphereGeometry args={[on ? 0.085 : 0.058, 20, 20]} />
            <meshBasicMaterial color={color} transparent opacity={on ? 1 : 0.7} />
          </mesh>
        )
      })}
    </>
  )
}

/** Grows the point under the cursor — points themselves cannot be sized per-vertex here. */
function HoverMarker({ position }) {
  if (!position) return null
  return (
    <mesh position={lift(position, 1.0)} raycast={() => null}>
      <sphereGeometry args={[0.07, 16, 16]} />
      <meshBasicMaterial color="#FB8B24" transparent opacity={0.95} />
    </mesh>
  )
}

/**
 * Flies the camera to a target once, then hands control back.
 *
 * The previous version lerped toward the focus on *every* frame, so orbiting away
 * was impossible — the camera fought the drag and snapped back. Now a fly is a
 * one-shot: it runs when `flyKey` changes (a new chain, or the Home button) and
 * any interaction cancels it. Distance is taken from wherever the camera already
 * is, so a fly never undoes the viewer's zoom.
 */
function CameraRig({ target, flyKey, controls }) {
  const desired = useRef(null)
  const flying = useRef(false)

  // Read the target through a ref so it is not an effect dependency: `focus` is a
  // fresh array on every turn change, and depending on it would re-arm the fly on
  // every arrow key — the exact yanking this rig exists to stop. Only flyKey (a
  // new chain, or Home) may start one.
  const targetRef = useRef(target)
  targetRef.current = target

  useEffect(() => {
    const t = targetRef.current
    if (!t) return
    const c = controls.current?.object
    const radius = c ? THREE.MathUtils.clamp(c.position.length(), 3.2, 9) : 6.2
    desired.current = new THREE.Vector3(...t).normalize().multiplyScalar(radius)
    flying.current = true
  }, [flyKey, controls])

  useEffect(() => {
    const c = controls.current
    if (!c) return undefined
    const cancel = () => {
      flying.current = false
    }
    c.addEventListener('start', cancel)
    return () => c.removeEventListener('start', cancel)
  }, [controls])

  useFrame(({ camera }) => {
    if (!flying.current || !desired.current) return
    camera.position.lerp(desired.current, 0.07)
    controls.current?.update?.()
    if (camera.position.distanceTo(desired.current) < 0.02) flying.current = false
  })

  return null
}

function Scene({
  positions,
  chainNodes,
  activeStep,
  focus,
  flyKey,
  dark,
  onPick,
  tooltipFor,
  cards,
}) {
  const controls = useRef()
  const { gl, raycaster } = useThree()
  const [hover, setHover] = useState(-1)

  useEffect(() => {
    gl.setClearAlpha(0)
  }, [gl])

  // Default Points threshold is 1 — half our sphere's radius, so every ray would
  // hit something. Scale it to roughly the on-screen size of a point.
  useEffect(() => {
    raycaster.params.Points.threshold = 0.045
  }, [raycaster])

  const hoverPos = hover >= 0 ? [positions[hover * 3], positions[hover * 3 + 1], positions[hover * 3 + 2]] : null
  const tip = hover >= 0 ? tooltipFor(hover) : null

  return (
    <>
      <Globe dark={dark} />
      <Cloud positions={positions} dark={dark} onHover={setHover} onPick={onPick} />
      {chainNodes?.length > 1 && <ChainPath nodes={chainNodes} activeStep={activeStep} dark={dark} />}
      <HoverMarker position={hoverPos} />

      {hoverPos && tip && (
        <Html position={lift(hoverPos, 1.06)} center style={{ pointerEvents: 'none' }} zIndexRange={[40, 0]}>
          <div className="w-56 -translate-y-14 rounded-xl border border-white/60 bg-white/80 p-2.5 shadow-2xl backdrop-blur-xl dark:border-white/10 dark:bg-neutral-900/80">
            <p className="truncate text-xs font-semibold text-neutral-900 dark:text-neutral-100">{tip.title}</p>
            <p className="mt-0.5 truncate text-[11px] text-neutral-500 dark:text-neutral-400">{tip.artist}</p>
            <p className="mt-1.5 text-[10px] font-medium text-stage-instruct">{tip.hint}</p>
          </div>
        </Html>
      )}

      {cards.map((c) => (
        <Html
          key={c.key}
          position={lift(c.position, c.k ?? 1.035)}
          center
          zIndexRange={[30, 0]}
          style={{ pointerEvents: 'auto' }}
        >
          {c.node}
        </Html>
      ))}

      <CameraRig target={focus} flyKey={flyKey} controls={controls} />
      <OrbitControls
        ref={controls}
        enablePan={false}
        enableDamping
        dampingFactor={0.08}
        rotateSpeed={0.5}
        zoomSpeed={0.8}
        minDistance={2.6}
        maxDistance={9}
      />
    </>
  )
}

export default function SphereView(props) {
  return (
    <Canvas camera={{ position: [0, 0, 6.2], fov: 42 }} gl={{ antialias: true, alpha: true }} dpr={[1, 2]}>
      <Scene {...props} />
    </Canvas>
  )
}
