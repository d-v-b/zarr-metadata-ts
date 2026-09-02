/**
 * The chunk grids this package interprets: syntax (configuration types) and
 * semantics (the spec's prose rules) for `regular` and `rectilinear`, in
 * one place per grid. The structural document layer never reads this
 * module — extension points are uninterpreted there by spec design.
 */

import type { PathedIssue } from "./errors.js";
import { configurationMissing, fieldParts, isIntArray, isPlainObject } from "./guards.js";

/** Configuration of the core `regular` chunk grid. */
export interface RegularChunkGridConfiguration {
  chunk_shape: number[];
}

/**
 * One dimension's spec in a rectilinear grid: a bare integer (uniform
 * shorthand, no sum constraint) or a list of chunk sizes and
 * `[size, count]` run-length pairs that must sum to the dimension length.
 */
export type RectilinearDimSpec = number | Array<number | [number, number]>;

/** Configuration of the zarr-extensions `rectilinear` chunk grid. */
export interface RectilinearChunkGridConfiguration {
  kind: "inline";
  chunk_shapes: RectilinearDimSpec[];
}

/** What interpreting a chunk grid yields for the rest of the semantic layer. */
export interface ChunkGridVerdict {
  /** Issues, pathed relative to the `chunk_grid` field. */
  issues: PathedIssue[];
  /**
   * Distinct chunk lengths per dimension — the division context a
   * sharding codec's inner chunks must satisfy — when derivable.
   */
  chunkSizes: number[][] | undefined;
}

function regularIssues(rawGrid: unknown, shape: number[] | undefined): ChunkGridVerdict {
  const issues: PathedIssue[] = [];
  let chunkSizes: number[][] | undefined;
  const configuration = fieldParts(rawGrid)?.configuration;
  if (configurationMissing(rawGrid)) {
    issues.push({
      path: [],
      message: '"regular" requires a configuration with "chunk_shape"',
      kind: "missing_key",
    });
  } else if (configuration !== undefined && !Object.hasOwn(configuration, "chunk_shape")) {
    issues.push({
      path: ["configuration", "chunk_shape"],
      message: "missing required key",
      kind: "missing_key",
    });
  }
  const configured = configuration?.["chunk_shape"];
  if (isIntArray(configured)) {
    if (shape !== undefined && configured.length !== shape.length) {
      issues.push({
        path: ["configuration", "chunk_shape"],
        message: `expected one length per dimension of shape (${shape.length})`,
        kind: "invalid_value",
      });
      // wrong arity: unusable as division context
    } else {
      chunkSizes = configured.map((length) => [length]);
    }
  }
  return { issues, chunkSizes };
}

function rectilinearIssues(rawGrid: unknown, shape: number[] | undefined): ChunkGridVerdict {
  const issues: PathedIssue[] = [];
  let chunkSizes: number[][] | undefined;
  const configuration = fieldParts(rawGrid)?.configuration;
  if (configurationMissing(rawGrid)) {
    issues.push({
      path: [],
      message: '"rectilinear" requires a configuration with "kind" and "chunk_shapes"',
      kind: "missing_key",
    });
  } else if (configuration !== undefined) {
    for (const key of ["kind", "chunk_shapes"]) {
      if (!Object.hasOwn(configuration, key)) {
        issues.push({
          path: ["configuration", key],
          message: "missing required key",
          kind: "missing_key",
        });
      }
    }
  }
  const specs = configuration?.["chunk_shapes"];
  if (Array.isArray(specs)) {
    if (shape !== undefined && specs.length !== shape.length) {
      issues.push({
        path: ["configuration", "chunk_shapes"],
        message: `expected one entry per dimension of shape (${shape.length})`,
        kind: "invalid_value",
      });
    } else {
      const perDim: (number[] | undefined)[] = specs.map((spec, dim) => {
        // Bare integer: uniform shorthand, no sum constraint (edge chunks
        // are permitted, as with the regular grid).
        if (Number.isInteger(spec)) return (spec as number) > 0 ? [spec as number] : undefined;
        if (!Array.isArray(spec)) return undefined; // registry schema's problem
        const sizes = new Set<number>();
        let total = 0;
        for (const entry of spec) {
          if (Number.isInteger(entry)) {
            sizes.add(entry as number);
            total += entry as number;
          } else if (
            Array.isArray(entry) &&
            entry.length === 2 &&
            Number.isInteger(entry[0]) &&
            Number.isInteger(entry[1])
          ) {
            sizes.add(entry[0] as number);
            total += (entry[0] as number) * (entry[1] as number);
          } else {
            return undefined; // malformed entry: registry schema's problem
          }
        }
        const extent = shape?.[dim];
        if (extent !== undefined && total !== extent) {
          issues.push({
            path: ["configuration", "chunk_shapes", dim],
            message: `expected chunk sizes summing to ${extent} along dimension ${dim}, got ${total}`,
            kind: "invalid_value",
          });
        }
        return [...sizes];
      });
      if (perDim.every((sizes) => sizes !== undefined)) {
        chunkSizes = perDim as number[][];
      }
    }
  }
  return { issues, chunkSizes };
}

/**
 * Interpret a document's `chunk_grid` field: required members, cross-field
 * rules against `shape`, and the division context for codec pipelines.
 * Unrecognized grid names yield no issues and no context — the extension
 * name space is open.
 */
export function chunkGridVerdict(rawGrid: unknown, shape: number[] | undefined): ChunkGridVerdict {
  if (!isPlainObject(rawGrid) && typeof rawGrid !== "string") {
    return { issues: [], chunkSizes: undefined }; // structural layer's problem
  }
  const name = fieldParts(rawGrid)?.name;
  if (name === "regular") return regularIssues(rawGrid, shape);
  if (name === "rectilinear") return rectilinearIssues(rawGrid, shape);
  return { issues: [], chunkSizes: undefined };
}
