import { useEffect, useMemo, useRef } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
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

/** All tracks as one instanced point cloud — 22k DOM nodes is impossible, 22k GPU points is free. */
function Cloud({ positions, dark }) {
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    return g
  }, [positions])

  useEffect(() => () => geom.dispose(), [geom])

  return (
    <points geometry={geom}>
      <pointsMaterial
        size={0.019}
        sizeAttenuation
        color={dark ? '#52525b' : '#c8ccd4'}
        transparent
        opacity={dark ? 0.85 : 0.9}
        depthWrite={false}
      />
    </points>
  )
}

function Globe({ dark }) {
  return (
    <>
      <mesh>
        <sphereGeometry args={[RADIUS * 0.985, 64, 64]} />
        <meshBasicMaterial
          color={dark ? '#0a0a0a' : '#ffffff'}
          transparent
          opacity={dark ? 0.82 : 0.9}
          depthWrite
        />
      </mesh>
      <mesh>
        <sphereGeometry args={[RADIUS * 0.9852, 48, 48]} />
        <meshBasicMaterial
          color={dark ? '#27272a' : '#e8eaee'}
          wireframe
          transparent
          opacity={dark ? 0.35 : 0.5}
        />
      </mesh>
    </>
  )
}

function ChainPath({ nodes, activeStep, dark }) {
  const arcs = useMemo(() => {
    const out = []
    for (let i = 0; i < nodes.length - 1; i++) {
      out.push(arcPoints(nodes[i], nodes[i + 1]))
    }
    return out
  }, [nodes])

  return (
    <>
      {arcs.map((pts, i) => {
        const on = i === activeStep
        const g = new THREE.BufferGeometry().setFromPoints(pts)
        return (
          <line key={i} geometry={g}>
            <lineBasicMaterial
              color={on ? '#FB8B24' : dark ? '#3f3f46' : '#c7cbd2'}
              transparent
              opacity={on ? 1 : 0.55}
              linewidth={1}
            />
          </line>
        )
      })}
      {nodes.map((p, i) => {
        const v = new THREE.Vector3(...p).normalize().multiplyScalar(RADIUS * 1.035)
        const isEnd = i === nodes.length - 1
        const on = i === activeStep || i === activeStep + 1
        const color = i === 0 ? '#1FA347' : isEnd ? '#7B3FF2' : '#2E6FD6'
        return (
          <mesh key={i} position={v}>
            <sphereGeometry args={[on ? 0.062 : 0.042, 20, 20]} />
            <meshBasicMaterial color={color} transparent opacity={on ? 1 : 0.62} />
          </mesh>
        )
      })}
    </>
  )
}

/** Eases the camera so the current edge faces the viewer. */
function CameraRig({ target, controls }) {
  const desired = useRef(new THREE.Vector3(0, 0, 6.2))

  useEffect(() => {
    if (!target) return
    const v = new THREE.Vector3(...target).normalize()
    desired.current = v.multiplyScalar(6.2)
  }, [target])

  useFrame(({ camera }) => {
    if (!target) return
    camera.position.lerp(desired.current, 0.045)
    controls.current?.update?.()
  })
  return null
}

function Scene({ positions, chainNodes, activeStep, focus, dark }) {
  const controls = useRef()
  const { gl } = useThree()
  useEffect(() => {
    gl.setClearAlpha(0)
  }, [gl])

  return (
    <>
      <Globe dark={dark} />
      <Cloud positions={positions} dark={dark} />
      {chainNodes?.length > 1 && <ChainPath nodes={chainNodes} activeStep={activeStep} dark={dark} />}
      <CameraRig target={focus} controls={controls} />
      <OrbitControls
        ref={controls}
        enablePan={false}
        enableDamping
        dampingFactor={0.08}
        rotateSpeed={0.5}
        minDistance={3.2}
        maxDistance={9}
        autoRotate={!chainNodes?.length}
        autoRotateSpeed={0.35}
      />
    </>
  )
}

export default function SphereView({ positions, chainNodes, activeStep, focus, dark }) {
  return (
    <Canvas
      camera={{ position: [0, 0, 6.2], fov: 42 }}
      gl={{ antialias: true, alpha: true }}
      dpr={[1, 2]}
    >
      <Scene
        positions={positions}
        chainNodes={chainNodes}
        activeStep={activeStep}
        focus={focus}
        dark={dark}
      />
    </Canvas>
  )
}
