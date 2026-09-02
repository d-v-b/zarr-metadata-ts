/** The core `sharding_indexed` codec: syntax and semantics. */
import type { PathedIssue } from "../errors.js";
import { isIntArray, type Path } from "../guards.js";
import { DROPPED, type PipelineContext } from "./index.js";

/** Configuration of the core `sharding_indexed` codec. */
export interface ShardingIndexedCodecConfiguration {
  chunk_shape: number[];
  codecs: unknown[];
  index_codecs: unknown[];
  index_location?: "start" | "end";
}

type Walk = (
  pipeline: unknown[],
  path: Path,
  dims: number | undefined,
  chunkSizes: number[][] | undefined,
) => PathedIssue[];

/**
 * Judge one sharding entry: it requires `chunk_shape`, `codecs`, and
 * `index_codecs`; its chunk_shape must match the array's dimensionality
 * and evenly divide every chunk it shards. The inner pipeline is walked
 * with the shard's own chunk context (`index_codecs` encode the shard
 * index, whose shape differs — no dimensional context applies there), and
 * the array -> bytes boundary spends the context for whatever follows.
 */
export function shardingIndexedIssues(
  configuration: Record<string, unknown> | undefined,
  configMissing: boolean,
  path: Path,
  index: number,
  context: PipelineContext,
  walk: Walk,
): { issues: PathedIssue[]; context: PipelineContext } {
  const issues: PathedIssue[] = [];
  if (configMissing) {
    issues.push({
      path: [...path, index],
      message:
        '"sharding_indexed" requires a configuration with "chunk_shape", "codecs", and "index_codecs"',
      kind: "missing_key",
    });
    return { issues, context: DROPPED };
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
        ...walk(inner, [...path, index, "configuration", "codecs"], undefined, undefined),
      );
    }
    return { issues, context: DROPPED };
  }
  const chunkShapePath = [...path, index, "configuration", "chunk_shape"];
  if (context.dims !== undefined && chunkShape.length !== context.dims) {
    issues.push({
      path: chunkShapePath,
      message: `expected one length per array dimension (${context.dims})`,
      kind: "invalid_value",
    });
  } else if (
    context.chunkSizes !== undefined &&
    chunkShape.length === context.chunkSizes.length &&
    chunkShape.every((length) => length > 0)
  ) {
    let violation: { axis: number; size: number } | undefined;
    context.chunkSizes.forEach((sizes, axis) => {
      if (violation !== undefined) return;
      const bad = sizes.find((size) => size % (chunkShape[axis] as number) !== 0);
      if (bad !== undefined) violation = { axis, size: bad };
    });
    if (violation !== undefined) {
      const { axis, size } = violation;
      const sizesNow = context.chunkSizes;
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
      ...walk(
        inner,
        [...path, index, "configuration", "codecs"],
        chunkShape.length,
        chunkShape.map((length) => [length]),
      ),
    );
  }
  return { issues, context: DROPPED };
}
