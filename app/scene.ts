import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import { Life3D, RULES, type Rule } from "./life";

export interface Stats {
  generation: number;
  population: number;
  fps: number;
}

export interface SceneCallbacks {
  onStats?: (s: Stats) => void;
}

const DEFAULT_RULE_ID = "b4s567"; // Slow Bloom

const GRID_SIZE = 15;
const SPACING = 5; // 5× the former cell-to-cell spacing → wide gaps between cells
const CELL_SIZE = 0.82;

/**
 * Owns the entire Three.js world: renderer, bloom compositor, a fixed
 * corner-mounted camera, and the InstancedMesh that draws every live cell.
 * React only pokes it through the small imperative surface at the bottom
 * (play, step, randomize, …). The camera does not move.
 */
export class Scene {
  private container: HTMLElement;
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private composer: EffectComposer;
  private bloom: UnrealBloomPass;
  private mesh: THREE.InstancedMesh;
  private life: Life3D;
  private cb: SceneCallbacks;

  private raf = 0;
  private disposed = false;

  // Simulation timing.
  private playing = false;
  private stepsPerSecond = 0.5; // one generation every 2 seconds
  private stepAccumulator = 0;
  private rule: Rule = RULES.find((r) => r.id === DEFAULT_RULE_ID) ?? RULES[0];
  private density = this.rule.density;

  private clock = new THREE.Clock();
  private fpsSmooth = 60;
  private statAccum = 0;

  // Smooth generation transitions: each cell's change is animated over the step
  // interval instead of snapping. Buffers hold per-particle start/end grid
  // coordinates and kind (0 survivor, 1 newborn, 2 dying); colours are written
  // once per transition since they don't change mid-animation.
  private hasTransition = false;
  private tElapsed = 0;
  private tDuration = 1 / 6;
  private partCount = 0;
  private partStart = new Float32Array(GRID_SIZE * GRID_SIZE * GRID_SIZE * 3);
  private partEnd = new Float32Array(GRID_SIZE * GRID_SIZE * GRID_SIZE * 3);
  private partKind = new Uint8Array(GRID_SIZE * GRID_SIZE * GRID_SIZE);

  // Fixed camera + subtle mouse parallax. The camera rests at basePos aiming at
  // lookTarget; the cursor nudges it a little along its own right/up axes so the
  // view responds to the mouse without becoming free-look.
  private basePos = new THREE.Vector3();
  private lookTarget = new THREE.Vector3();
  private rightAxis = new THREE.Vector3();
  private upAxis = new THREE.Vector3();
  private parallaxAmp = 0;
  private mouseX = 0;
  private mouseY = 0;
  private mouseSmoothX = 0;
  private mouseSmoothY = 0;

  // Scratch objects reused every frame (avoid per-cell allocation).
  private dummy = new THREE.Object3D();
  private color = new THREE.Color();

  constructor(container: HTMLElement, cb: SceneCallbacks = {}) {
    this.container = container;
    this.cb = cb;

    const w = container.clientWidth || window.innerWidth;
    const h = container.clientHeight || window.innerHeight;

    // Renderer.
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    container.appendChild(this.renderer.domElement);

    // Scene + fog for depth cueing into the dark.
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x05060a);
    const worldExtent = GRID_SIZE * SPACING;
    this.scene.fog = new THREE.Fog(0x05060a, worldExtent * 0.9, worldExtent * 3.2);

    // Fixed camera anchored to the +X/+Y/+Z corner, inset 25% of the world
    // extent inward from that corner, looking 45° downward toward the opposite
    // (−X/−Y/−Z) corner. The camera never moves.
    this.camera = new THREE.PerspectiveCamera(65, w / h, 0.1, 1000);
    const inset = 0.25;
    const half = worldExtent / 2;
    this.camera.position.set(half - inset * worldExtent, half - inset * worldExtent, half - inset * worldExtent);
    // Azimuth toward the opposite corner across the XZ plane, pitched 45° down.
    const s = Math.SQRT1_2;
    const lookDir = new THREE.Vector3(-s, 0, -s).multiplyScalar(s).add(new THREE.Vector3(0, -s, 0));
    this.basePos.copy(this.camera.position);
    this.lookTarget.copy(this.basePos).addScaledVector(lookDir, worldExtent);
    this.rightAxis.crossVectors(lookDir, new THREE.Vector3(0, 1, 0)).normalize();
    this.upAxis.crossVectors(this.rightAxis, lookDir).normalize();
    this.parallaxAmp = worldExtent * 0.05; // slight — enough to feel responsive
    this.camera.lookAt(this.lookTarget);

    // Lights (cells are emissive, but standard material still needs some fill).
    this.scene.add(new THREE.AmbientLight(0x445566, 0.9));
    const hemi = new THREE.HemisphereLight(0x88aaff, 0x221133, 0.7);
    this.scene.add(hemi);
    const dir = new THREE.DirectionalLight(0xffffff, 1.4);
    dir.position.set(1, 1.5, 0.8);
    this.scene.add(dir);
    const fill = new THREE.DirectionalLight(0xaad4ff, 0.7);
    fill.position.set(-1, -0.6, -1);
    this.scene.add(fill);

    // The cell mesh: one instanced rounded box per possible cell.
    // (segments=3 keeps the corner tessellation cheap; radius ~1/5 of the cell.)
    const geo = new RoundedBoxGeometry(CELL_SIZE, CELL_SIZE, CELL_SIZE, 3, CELL_SIZE * 0.2);
    const mat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      metalness: 0.1,
      roughness: 0.45,
    });
    // MeshStandardMaterial has only one (uniform) emissive colour, which would
    // make every cell glow the same white. Inject the per-instance colour into
    // the emissive term (via our own uniform) so each cell glows in its own hue
    // for the bloom pass. vColor is a vec4 for instanced colour, hence .rgb.
    mat.onBeforeCompile = (shader) => {
      shader.uniforms.uEmissiveBoost = { value: 0.4 };
      shader.fragmentShader =
        "uniform float uEmissiveBoost;\n" +
        shader.fragmentShader.replace(
          "vec3 totalEmissiveRadiance = emissive;",
          "vec3 totalEmissiveRadiance = vColor.rgb * uEmissiveBoost;",
        );
    };
    const maxInstances = GRID_SIZE * GRID_SIZE * GRID_SIZE;
    this.mesh = new THREE.InstancedMesh(geo, mat, maxInstances);
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    // Prime the per-instance colour buffer so setColorAt works every frame.
    this.mesh.setColorAt(0, this.color);
    if (this.mesh.instanceColor) this.mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    this.mesh.count = 0;
    this.scene.add(this.mesh);

    // Post-processing: subtle bloom so live cells glow against the dark.
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(w, h), 1.0, 0.35, 0.62);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
    this.composer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.composer.setSize(w, h);

    // Simulation.
    this.life = new Life3D(GRID_SIZE, this.rule);
    this.life.randomize(this.density);
    this.updateInstances();

    // Events: viewport resize + cursor position for subtle parallax.
    window.addEventListener("resize", this.onResize);
    window.addEventListener("pointermove", this.onPointerMove);

    this.clock.start();
    this.loop();
  }

  // --- Event handlers -------------------------------------------------------

  private onPointerMove = (e: PointerEvent) => {
    // Normalised cursor position, −1..1 from the screen centre.
    this.mouseX = (e.clientX / window.innerWidth) * 2 - 1;
    this.mouseY = (e.clientY / window.innerHeight) * 2 - 1;
  };

  private onResize = () => {
    const w = this.container.clientWidth || window.innerWidth;
    const h = this.container.clientHeight || window.innerHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.composer.setSize(w, h);
  };

  // --- Rendering ------------------------------------------------------------

  /** Rebuild the instance matrices/colours from the current life grid. */
  private updateInstances() {
    const half = (GRID_SIZE - 1) / 2;
    const dummy = this.dummy;
    dummy.scale.setScalar(1);
    let i = 0;
    this.life.forEachLive((x, y, z, age) => {
      dummy.position.set((x - half) * SPACING, (y - half) * SPACING, (z - half) * SPACING);
      dummy.updateMatrix();
      this.mesh.setMatrixAt(i, dummy.matrix);
      this.color.setHSL(...this.cellHSL(y, age));
      this.mesh.setColorAt(i, this.color);
      i++;
    });
    this.mesh.count = i;
    this.mesh.instanceMatrix.needsUpdate = true;
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
  }

  /** Cell colour: hue runs bottom→top; younger cells are brighter. */
  private cellHSL(y: number, age: number): [number, number, number] {
    const hue = 0.62 - 0.62 * (y / (GRID_SIZE - 1));
    const t = Math.min(age, 8) / 8;
    return [(hue + 1) % 1, 0.95, 0.58 - 0.28 * t];
  }

  /** Snapshot the just-computed generation change into the particle buffers. */
  private buildTransition() {
    let i = 0;
    this.life.emitTransition((kind, sx, sy, sz, ex, ey, ez, age) => {
      const j = i * 3;
      this.partStart[j] = sx; this.partStart[j + 1] = sy; this.partStart[j + 2] = sz;
      this.partEnd[j] = ex; this.partEnd[j + 1] = ey; this.partEnd[j + 2] = ez;
      this.partKind[i] = kind;
      // Newborns colour by their destination height so they arrive in-palette.
      this.color.setHSL(...this.cellHSL(ey, age));
      this.mesh.setColorAt(i, this.color);
      i++;
    });
    this.partCount = i;
    this.mesh.count = i;
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
    this.hasTransition = true;
  }

  /** Draw the current transition interpolated by eased progress e ∈ [0,1]. */
  private drawTransition(e: number) {
    const half = (GRID_SIZE - 1) / 2;
    const dummy = this.dummy;
    const st = this.partStart;
    const en = this.partEnd;
    const kind = this.partKind;
    const n = this.partCount;
    for (let i = 0; i < n; i++) {
      const j = i * 3;
      const k = kind[i];
      let px: number, py: number, pz: number, scale: number;
      if (k === 1) {
        // Newborn: slide from parent into place. Start near full size so the
        // translation reads as motion rather than a pop-in.
        px = st[j] + (en[j] - st[j]) * e;
        py = st[j + 1] + (en[j + 1] - st[j + 1]) * e;
        pz = st[j + 2] + (en[j + 2] - st[j + 2]) * e;
        scale = 0.55 + 0.45 * e;
      } else if (k === 2) {
        // Dying: shrink away in place.
        px = en[j]; py = en[j + 1]; pz = en[j + 2];
        scale = 1 - e;
      } else {
        px = en[j]; py = en[j + 1]; pz = en[j + 2];
        scale = 1;
      }
      dummy.position.set((px - half) * SPACING, (py - half) * SPACING, (pz - half) * SPACING);
      dummy.scale.setScalar(scale > 0.0001 ? scale : 0.0001);
      dummy.updateMatrix();
      this.mesh.setMatrixAt(i, dummy.matrix);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
  }

  private loop = () => {
    if (this.disposed) return;
    this.raf = requestAnimationFrame(this.loop);
    const dt = Math.min(this.clock.getDelta(), 0.1);

    // Subtle mouse parallax: ease the camera a little toward the cursor along
    // its own right/up axes, then re-aim at the fixed target.
    const k = Math.min(1, dt * 4);
    this.mouseSmoothX += (this.mouseX - this.mouseSmoothX) * k;
    this.mouseSmoothY += (this.mouseY - this.mouseSmoothY) * k;
    this.camera.position
      .copy(this.basePos)
      .addScaledVector(this.rightAxis, this.mouseSmoothX * this.parallaxAmp)
      .addScaledVector(this.upAxis, -this.mouseSmoothY * this.parallaxAmp);
    this.camera.lookAt(this.lookTarget);

    const interval = 1 / this.stepsPerSecond;
    if (this.playing) {
      this.stepAccumulator += dt;
      // Advance to the next generation only once the current transition has
      // fully played, so every birth and death is animated end to end. When no
      // transition is in flight yet (fresh load / after reseed), step right
      // away — otherwise the first generation would never fire.
      if (
        this.stepAccumulator >= interval &&
        (!this.hasTransition || this.tElapsed >= this.tDuration)
      ) {
        this.life.step();
        this.buildTransition();
        this.tDuration = interval;
        this.tElapsed = 0;
        this.stepAccumulator = 0;
      }
    }

    // Progress the in-flight transition (this also plays out a single Step
    // taken while paused). Interpolation is LINEAR so motion runs at constant
    // velocity and flows seamlessly into the next generation's transition,
    // rather than easing to a stop at each generation boundary.
    if (this.hasTransition) {
      if (this.tElapsed < this.tDuration) {
        this.tElapsed = Math.min(this.tElapsed + dt, this.tDuration);
      }
      const e = this.tDuration > 0 ? this.tElapsed / this.tDuration : 1;
      this.drawTransition(e);
    }

    this.composer.render();

    // Throttled stats push to React.
    if (dt > 0) this.fpsSmooth += (1 / dt - this.fpsSmooth) * 0.1;
    this.statAccum += dt;
    if (this.statAccum >= 0.2) {
      this.statAccum = 0;
      this.cb.onStats?.({
        generation: this.life.generation,
        population: this.life.population,
        fps: Math.round(this.fpsSmooth),
      });
    }
  };

  // --- Imperative API for React --------------------------------------------

  setPlaying(v: boolean) {
    this.playing = v;
    this.stepAccumulator = 0;
  }

  stepOnce() {
    this.life.step();
    this.buildTransition();
    // Give a clearly visible animation even when paused.
    this.tDuration = Math.max(1 / this.stepsPerSecond, 0.45);
    this.tElapsed = 0;
    this.cb.onStats?.({
      generation: this.life.generation,
      population: this.life.population,
      fps: Math.round(this.fpsSmooth),
    });
  }

  setSpeed(stepsPerSecond: number) {
    this.stepsPerSecond = stepsPerSecond;
  }

  setDensity(d: number) {
    this.density = d;
  }

  setRule(ruleId: string) {
    const rule = RULES.find((r) => r.id === ruleId) ?? RULES[0];
    this.rule = rule;
    this.density = rule.density;
    this.life.setRule(rule);
    this.reseed();
    return rule;
  }

  reseed() {
    this.life.randomize(this.density);
    this.stepAccumulator = 0;
    this.tElapsed = 0;
    this.hasTransition = false; // show the seed statically until the first step
    this.updateInstances();
    this.cb.onStats?.({
      generation: this.life.generation,
      population: this.life.population,
      fps: Math.round(this.fpsSmooth),
    });
  }

  setBloom(strength: number) {
    this.bloom.strength = strength;
  }

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    window.removeEventListener("resize", this.onResize);
    window.removeEventListener("pointermove", this.onPointerMove);
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
    this.composer.dispose();
    this.renderer.dispose();
    if (this.renderer.domElement.parentElement === this.container) {
      this.container.removeChild(this.renderer.domElement);
    }
  }
}

export { RULES } from "./life";
export const CONFIG = { GRID_SIZE };
