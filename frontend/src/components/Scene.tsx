"use client";

import { Canvas } from "@react-three/fiber";
import { 
  OrbitControls, 
  Environment, 
  Float,
  ContactShadows,
  TorusKnot
} from "@react-three/drei";
import { Suspense, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

function PremiumObject() {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.2) * 0.5;
      meshRef.current.rotation.y += 0.005;
    }
  });

  return (
    <Float speed={2} rotationIntensity={0.5} floatIntensity={2}>
      <TorusKnot ref={meshRef} args={[1, 0.3, 256, 64]} scale={1.2}>
        <meshPhysicalMaterial 
          color="#f2efe9"
          emissive="#2a241d"
          emissiveIntensity={0.05}
          roughness={0.1}
          metalness={0.8}
          transmission={0.9}
          thickness={1.5}
          ior={1.5}
          clearcoat={1}
          clearcoatRoughness={0.1}
        />
      </TorusKnot>
    </Float>
  );
}

export default function Scene() {
  return (
    <div className="absolute inset-0 z-0 h-screen w-full pointer-events-none">
      <Canvas camera={{ position: [0, 0, 6], fov: 45 }}>
        <color attach="background" args={["#fdfbf7"]} />
        <ambientLight intensity={0.6} color="#ffffff" />
        <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} intensity={1} castShadow />
        <pointLight position={[-10, -10, -10]} intensity={0.5} color="#c48c46" />
        
        <Suspense fallback={null}>
          <PremiumObject />
          <Environment preset="city" />
          <ContactShadows 
            position={[0, -2.5, 0]} 
            opacity={0.4} 
            scale={20} 
            blur={2} 
            far={4.5} 
            color="#2a241d" 
          />
        </Suspense>
        <OrbitControls 
          enableZoom={false} 
          enablePan={false} 
          maxPolarAngle={Math.PI / 2}
          minPolarAngle={Math.PI / 2}
        />
      </Canvas>
    </div>
  );
}
