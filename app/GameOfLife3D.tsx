"use client";

import { useEffect, useRef } from "react";
import { Scene } from "./scene";

export default function GameOfLife3D() {
  const mountRef = useRef<HTMLDivElement>(null);

  // Create the Three.js scene once on mount. The camera is fixed and there are
  // no controls — it simply runs and animates.
  useEffect(() => {
    if (!mountRef.current) return;
    const scene = new Scene(mountRef.current);
    scene.setPlaying(true);
    return () => scene.dispose();
  }, []);

  return (
    <div className="fixed inset-0 overflow-hidden bg-[#05060a] select-none">
      {/* Three.js canvas mounts here */}
      <div ref={mountRef} className="absolute inset-0" />
    </div>
  );
}
