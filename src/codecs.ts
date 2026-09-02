/**
 * The codecs this package interprets — `transpose` and `sharding_indexed` —
 * syntax (configuration types) and semantics together, plus the pipeline
 * walker that threads dimensional context between codecs (mirroring
 * zarr-python's chunk-spec threading): a valid transpose permutes the
 * per-dimension chunk sizes for the codecs after it, and any codec this
 * module cannot reason about — `reshape` may change a chunk's rank, and
 * unknown codecs may do anything — invalidates the dimensional context for
 * the rest of the pipeline instead of letting stale array-level facts
 * produce false verdicts.
 *
 * Other codecs' required fields (blosc, gzip, zstd, ...) are deliberately
 * NOT known here: they are registry-schema facts enforced by schema-driven
 * tooling, and a second hardcoded source of truth would drift.
 */

import type { PathedIssue } from "./errors.js";
import { configurationMissing, fieldParts, isIntArray, type Path } from "./guards.js";

/** Configuration of the core `transpose` codec. */
export interface TransposeCodecConfiguration {
  order: number[];
}

/** Configuration of the core `sharding_indexed` codec. */
export interface ShardingIndexedCodecConfiguration {
  chunk_shape: number[];
  codecs: unknown[];
  index_codecs: unknown[];
  index_location?: "start" | "end";
}

function isPermutation(order: number[]): boolean {
  const seen = new Set(order);
  return seen.size === order.length && order.every((entry) => entry >= 0 && entry < order.length);
}

/**
 * Walk one codec pipeline. `initialDims` is the array dimensionality at
 * this point (undefined when unknowable); `initialChunkSizes` holds the
 * distinct chunk lengths per dimension this pipeline may encode — what a
 * sharding codec's inner chunks must divide. A regular grid contributes
 * one size per dimension; a rectilinear grid may contribute several.
 */
export function pipelineIssues(
  pipeline: unknown[],
  path: Path,
  initialDims: number | undefined,
  initialChunkSizes: number[][] | undefined,
): PathedIssue[] {
  const issues: PathedIssue[] = [];
  // The dimensional context THREADS through the pipeline: transpose permutes
  // it, and anything this module cannot reason about invalidates it for the
  // remaining codecs.
  let dims = initialDims;
  let chunkSizes = initialChunkSizes;
  const dropContext = (): void => {
    dims = undefined;
    chunkSizes = undefined;
  };
  pipeline.forEach((entry, index) => {
    const parts = fieldParts(entry);
    if (parts === undefined) {
      dropContext(); // structurally invalid entry: no further reasoning
      return;
    }
    const { name, configuration } = parts;
    const configMissing = configurationMissing(entry);
    if (name === "transpose") {
      if (configMissing) {
        issues.push({
          path: [...path, index],
          message: '"transpose" requires a configuration with "order"',
          kind: "missing_key",
        });
        dropContext();
        return;
      }
      if (configuration !== undefined && !Object.hasOwn(configuration, "order")) {
        issues.push({
          path: [...path, index, "configuration", "order"],
          message: "missing required key",
          kind: "missing_key",
        });
        dropContext();
        return;
      }
      const order = configuration?.["order"];
      if (!isIntArray(order)) {
        dropContext(); // shape errors are the schema layer's
        return;
      }
      const orderPath = [...path, index, "configuration", "order"];
      let sound = true;
      if (!isPermutation(order)) {
        sound = false;
        issues.push({
          path: orderPath,
          message: `expected a permutation of the integers 0..${order.length - 1}`,
          kind: "invalid_value",
        });
      }
      if (dims !== undefined && order.length !== dims) {
        sound = false;
        issues.push({
          path: orderPath,
          message: `expected one entry per array dimension (${dims})`,
          kind: "invalid_value",
        });
      }
      if (!sound) {
        dropContext();
      } else if (chunkSizes !== undefined) {
        const sizes = chunkSizes;
        chunkSizes =
          order.length === sizes.length
            ? order.map((axis) => sizes[axis] as number[])
            : undefined;
      }
    } else if (name === "sharding_indexed") {
      if (configMissing) {
        issues.push({
          path: [...path, index],
          message:
            '"sharding_indexed" requires a configuration with "chunk_shape", "codecs", and "index_codecs"',
          kind: "missing_key",
        });
        dropContext();
        return;
      }
      if (configuration !== undefined) {
        for (const key of ["chunk_shape", "codecs", "index_codecs"]) {
          if (!Object.hasOwn(configuration, key)) {
            issues.push({
              path: [...path, index, "configuration", key],
              message: "missing required key",
              kind: "missing_key",
            });
          }
        }
      }
      const chunkShape = configuration?.["chunk_shape"];
      if (!isIntArray(chunkShape)) {
        // Still walk a present inner pipeline (entries deserve their own
        // verdicts) before the context becomes unknowable.
        const inner = configuration?.["codecs"];
        if (Array.isArray(inner)) {
          issues.push(
            ...pipelineIssues(
              inner,
              [...path, index, "configuration", "codecs"],
              undefined,
              undefined,
            ),
          );
        }
        dropContext();
        return;
      }
      const chunkShapePath = [...path, index, "configuration", "chunk_shape"];
      if (dims !== undefined && chunkShape.length !== dims) {
        issues.push({
          path: chunkShapePath,
          message: `expected one length per array dimension (${dims})`,
          kind: "invalid_value",
        });
      } else if (
        chunkSizes !== undefined &&
        chunkShape.length === chunkSizes.length &&
        chunkShape.every((length) => length > 0)
      ) {
        let violation: { axis: number; size: number } | undefined;
        chunkSizes.forEach((sizes, axis) => {
          if (violation !== undefined) return;
          const bad = sizes.find((size) => size % (chunkShape[axis] as number) !== 0);
          if (bad !== undefined) violation = { axis, size: bad };
        });
        if (violation !== undefined) {
          const { axis, size } = violation;
          const sizesNow = chunkSizes;
          const uniform = sizesNow.every((sizes) => sizes.length === 1);
          issues.push({
            path: chunkShapePath,
            message: uniform
              ? `expected ${JSON.stringify(chunkShape)} to evenly divide the outer chunk shape ${JSON.stringify(sizesNow.map((sizes) => sizes[0]))}`
              : `expected ${JSON.stringify(chunkShape)} to evenly divide every chunk size of the grid (dimension ${axis} has chunk size ${size})`,
            kind: "invalid_value",
          });
        }
      }
      const inner = configuration?.["codecs"];
      if (Array.isArray(inner)) {
        issues.push(
          ...pipelineIssues(
            inner,
            [...path, index, "configuration", "codecs"],
            chunkShape.length,
            chunkShape.map((length) => [length]),
          ),
        );
      }
      // index_codecs encode the shard index, whose shape differs from the
      // array's — no dimensional context applies there. After the
      // array -> bytes boundary the dimensional context is spent.
      dropContext();
    } else {
      // A codec this module cannot reason about (reshape may change a
      // chunk's rank; unknown codecs may do anything): stale array-level
      // facts must not judge the codecs after it.
      dropContext();
    }
  });
  return issues;
}
