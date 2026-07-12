// A faithful generalization of Conway's Game of Life to three dimensions.
//
// Conway's original rules (B3/S23) operate on a 2D grid where every cell has 8
// neighbours (the Moore neighbourhood). We keep every part of that machinery —
// a discrete grid, simultaneous updates, and birth/survival thresholds counted
// over the Moore neighbourhood — and change exactly one thing: the grid is now
// 3D, so each cell has 26 neighbours instead of 8.
//
// Because the neighbourhood grew, the specific counts "2, 3" no longer produce
// interesting behaviour, so we expose the birth/survival sets as data. The
// default preset (Bays' 5766) is the well-studied 3D analogue of Conway's Life:
// it sustains stable structures, oscillators, and gliders, just like B3/S23
// does in 2D. The literal B3/S23 rule is also included for the purist.

export interface Rule {
  id: string;
  name: string;
  /** Neighbour counts (0..26) at which a dead cell is born. */
  birth: number[];
  /** Neighbour counts (0..26) at which a live cell survives. */
  survival: number[];
  /** Recommended seed density for this rule (0..1). */
  density: number;
  description: string;
}

// The default is Conway's EXACT rule (B3/S23) applied to the 3D Moore
// neighbourhood: nothing about the rule changes except that "neighbourhood" now
// means the 26 surrounding cells instead of 8, which is inherent to being 3D.
// Empirically it grows outward from a small seed in all directions and settles
// into a churning ~16% cloud — no rule changes required. The remaining presets
// are retuned birth/survival counts found by an empirical sweep, offering
// different aesthetics (denser, slower, more crystalline). Counts run 0..26.
export const RULES: Rule[] = [
  {
    id: "conway",
    name: "Conway B3/S23 (3D)",
    survival: [2, 3],
    birth: [3],
    density: 0.5,
    description:
      "Conway's exact rule — born on 3 neighbours, survive on 2-3 — over the full 26-neighbour 3D Moore neighbourhood. Only the dimension changes. From a small seed it grows outward in every direction and sustains a stable ~16% cloud.",
  },
  {
    id: "b3s567",
    name: "Bloom — B3/S5-7",
    survival: [5, 6, 7],
    birth: [3],
    density: 0.5,
    description:
      "Same birth count as Conway (3) but survival retuned to 5-7. Grows into a rounder, more solid bloom.",
  },
  {
    id: "b4s567",
    name: "Slow Bloom — B4/S5-7",
    survival: [5, 6, 7],
    birth: [4],
    density: 0.5,
    description:
      "Grows more gradually, so you can watch the frontier advance through the volume before it settles.",
  },
  {
    id: "b23s45",
    name: "Coral — B2-3/S4-5",
    survival: [4, 5],
    birth: [2, 3],
    density: 0.5,
    description: "A busier, denser bloom with constant churn across the surface.",
  },
];

export class Life3D {
  readonly size: number;
  private readonly n: number;
  private cells: Uint8Array;
  private next: Uint8Array;
  /** Consecutive generations each cell has been alive; 0 when dead. */
  ages: Uint16Array;
  private birthLut: Uint8Array; // birthLut[count] === 1 -> birth
  private survivalLut: Uint8Array;
  generation = 0;
  population = 0;

  constructor(size: number, rule: Rule) {
    this.size = size;
    this.n = size * size * size;
    this.cells = new Uint8Array(this.n);
    this.next = new Uint8Array(this.n);
    this.ages = new Uint16Array(this.n);
    this.birthLut = new Uint8Array(27);
    this.survivalLut = new Uint8Array(27);
    this.setRule(rule);
  }

  setRule(rule: Rule) {
    this.birthLut.fill(0);
    this.survivalLut.fill(0);
    for (const b of rule.birth) this.birthLut[b] = 1;
    for (const s of rule.survival) this.survivalLut[s] = 1;
  }

  private idx(x: number, y: number, z: number) {
    return x + y * this.size + z * this.size * this.size;
  }

  isAlive(x: number, y: number, z: number) {
    return this.cells[this.idx(x, y, z)] === 1;
  }

  clear() {
    this.cells.fill(0);
    this.next.fill(0);
    this.ages.fill(0);
    this.generation = 0;
    this.population = 0;
  }

  /**
   * Randomly seed a small cube at the centre of the world, so that colonies
   * grow *outward* into empty space in all three dimensions — the whole point
   * of a 3D automaton. A large seed would just churn in place. `seedFraction`
   * is the side length of the seed region as a fraction of the world; `rand`
   * lets callers supply a seeded PRNG for reproducibility.
   */
  randomize(density: number, seedFraction = 0.28, rand: () => number = Math.random) {
    this.clear();
    const s = this.size;
    const seedSide = Math.max(3, Math.round(s * seedFraction));
    const margin = Math.floor((s - seedSide) / 2);
    const lo = margin;
    const hi = margin + seedSide;
    let pop = 0;
    for (let z = lo; z < hi; z++) {
      for (let y = lo; y < hi; y++) {
        for (let x = lo; x < hi; x++) {
          if (rand() < density) {
            const i = this.idx(x, y, z);
            this.cells[i] = 1;
            this.ages[i] = 1;
            pop++;
          }
        }
      }
    }
    this.population = pop;
  }

  /** Advance one generation. Returns the new population. */
  step(): number {
    const s = this.size;
    const cells = this.cells;
    const next = this.next;
    const ages = this.ages;
    const bLut = this.birthLut;
    const sLut = this.survivalLut;
    let pop = 0;

    for (let z = 0; z < s; z++) {
      const zLo = z > 0 ? z - 1 : 0;
      const zHi = z < s - 1 ? z + 1 : s - 1;
      for (let y = 0; y < s; y++) {
        const yLo = y > 0 ? y - 1 : 0;
        const yHi = y < s - 1 ? y + 1 : s - 1;
        for (let x = 0; x < s; x++) {
          const xLo = x > 0 ? x - 1 : 0;
          const xHi = x < s - 1 ? x + 1 : s - 1;

          // Count the 26 neighbours (bounded box, no wrap-around).
          let count = 0;
          for (let nz = zLo; nz <= zHi; nz++) {
            const zOff = nz * s * s;
            for (let ny = yLo; ny <= yHi; ny++) {
              const yzOff = zOff + ny * s;
              for (let nx = xLo; nx <= xHi; nx++) {
                count += cells[yzOff + nx];
              }
            }
          }

          const i = this.idx(x, y, z);
          const alive = cells[i];
          count -= alive; // don't count the cell itself

          let live: number;
          if (alive) {
            live = sLut[count];
          } else {
            live = bLut[count];
          }

          next[i] = live as number;
          if (live) {
            ages[i] = alive ? ages[i] + 1 : 1;
            pop++;
          } else {
            ages[i] = 0;
          }
        }
      }
    }

    // Swap buffers.
    this.cells = next;
    this.next = cells;
    this.generation++;
    this.population = pop;
    return pop;
  }

  /** Iterate live cells, invoking cb with coordinates, age and linear index. */
  forEachLive(cb: (x: number, y: number, z: number, age: number, i: number) => void) {
    const s = this.size;
    const cells = this.cells;
    const ages = this.ages;
    let i = 0;
    for (let z = 0; z < s; z++) {
      for (let y = 0; y < s; y++) {
        for (let x = 0; x < s; x++, i++) {
          if (cells[i]) cb(x, y, z, ages[i], i);
        }
      }
    }
  }

  /**
   * Describe the change from the previous generation to the current one, so the
   * renderer can animate it smoothly rather than snapping. Must be called after
   * step() and before the next step(): the previous generation still lives in
   * the swapped-out buffer (`this.next`).
   *
   * `kind`: 0 = survivor (held in place), 1 = newborn, 2 = dying.
   * For a newborn, (sx,sy,sz) is its ORIGIN — the centroid of the live
   * neighbours that caused the birth — and (ex,ey,ez) is its destination cell,
   * so it can slide/grow outward from the parent mass. Survivors and dying
   * cells report origin == destination.
   */
  emitTransition(
    cb: (
      kind: 0 | 1 | 2,
      sx: number, sy: number, sz: number,
      ex: number, ey: number, ez: number,
      age: number,
    ) => void,
  ) {
    const s = this.size;
    const now = this.cells;
    const prev = this.next; // previous generation, valid right after a step()
    const ages = this.ages;
    let i = 0;
    for (let z = 0; z < s; z++) {
      const zLo = z > 0 ? z - 1 : 0;
      const zHi = z < s - 1 ? z + 1 : s - 1;
      for (let y = 0; y < s; y++) {
        const yLo = y > 0 ? y - 1 : 0;
        const yHi = y < s - 1 ? y + 1 : s - 1;
        for (let x = 0; x < s; x++, i++) {
          const alive = now[i];
          const wasAlive = prev[i];
          if (alive) {
            if (wasAlive) {
              cb(0, x, y, z, x, y, z, ages[i]); // survivor
            } else {
              // Newborn: origin = the NEAREST live parent neighbour in the
              // previous generation, so the cell slides a clear full cell into
              // place (a face neighbour when one exists) rather than drifting a
              // fraction of a cell from the neighbourhood average.
              const xLo = x > 0 ? x - 1 : 0;
              const xHi = x < s - 1 ? x + 1 : s - 1;
              let ox = x, oy = y, oz = z, best = Infinity;
              for (let nz = zLo; nz <= zHi; nz++) {
                for (let ny = yLo; ny <= yHi; ny++) {
                  for (let nx = xLo; nx <= xHi; nx++) {
                    if ((nx !== x || ny !== y || nz !== z) && prev[nx + ny * s + nz * s * s]) {
                      const d = (nx - x) * (nx - x) + (ny - y) * (ny - y) + (nz - z) * (nz - z);
                      if (d < best) { best = d; ox = nx; oy = ny; oz = nz; }
                    }
                  }
                }
              }
              cb(1, ox, oy, oz, x, y, z, ages[i]); // newborn slides from parent
            }
          } else if (wasAlive) {
            cb(2, x, y, z, x, y, z, 0); // dying
          }
        }
      }
    }
  }
}
