import type { Step } from "./types";
export const stateKey = (edges: string[]) => JSON.stringify(edges);

/** Preserve the focused branch even when a large trace is windowed. */
export function treeStates(trace: Step[], index: number, limit = 120) {
  const visible = trace.slice(0, Math.max(0, index + 1));
  const states = new Map<string, Step>();
  for (const step of visible) states.set(stateKey(step.edge_ids), step);
  const current = visible.at(-1);
  const required = new Set<string>();
  if (current) {
    for (let depth = 0; depth <= current.depth; depth++) {
      const edges = current.edge_ids.slice(0, depth);
      const key = stateKey(edges);
      required.add(key);
      if (!states.has(key))
        states.set(key, {
          ...current,
          id: -depth - 1,
          node: current.path[depth],
          depth,
          path: current.path.slice(0, depth + 1),
          edge_ids: edges,
          parent_path: edges.slice(0, -1),
          status: "path",
          g: depth,
          h: null,
          f: null,
          p: null,
        });
    }
  }
  const selected = new Map<string, Step>();
  for (const key of required) selected.set(key, states.get(key)!);
  for (const [key, step] of states) {
    if (selected.size >= limit) break;
    selected.set(key, step);
  }
  // Insertion order of the original trace keeps siblings stable while playing.
  return {
    states: [...states]
      .filter(([key]) => selected.has(key))
      .map(([, step]) => step),
    total: states.size,
    current,
    hidden: Math.max(0, states.size - selected.size),
  };
}
