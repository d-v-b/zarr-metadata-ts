/**
 * The codecs this package interprets, one module per codec, plus the
 * pipeline walker that threads dimensional context between codecs
 * (mirroring zarr-python's chunk-spec threading): a valid transpose
 * permutes the per-dimension chunk sizes for the codecs after it, and any
 * codec this package cannot reason about — `reshape` may change a chunk's
 * rank, and unknown codecs may do anything — invalidates the dimensional
 * context for the rest of the pipeline instead of letting stale
 * array-level facts produce false verdicts.
 *
 * Other codecs' required fields (blosc, gzip, zstd, ...) are deliberately
 * NOT known here: they are registry-schema facts enforced by schema-driven
 * tooling, and a second hardcoded source of truth would drift.
 */
import type { PathedIssue } from "../errors.js";
import { configurationMissing, fieldParts, type Path } from "../guards.js";
import { shardingIndexedIssues } from "./sharding-indexed.js";
import { transposeIssues } from "./transpose.js";

export type { ShardingIndexedCodecConfiguration } from "./sharding-indexed.js";
export type { TransposeCodecConfiguration } from "./transpose.js";

/** The dimensional facts threaded between the codecs of one pipeline. */
export interface PipelineContext {
  /** Array dimensionality at this point; undefined when unknowable. */
  dims: number | undefined;
  /**
   * Distinct chunk lengths per dimension this pipeline may encode — what a
   * sharding codec's inner chunks must divide. A regular grid contributes
   * one size per dimension; a rectilinear grid may contribute several.
   */
  chunkSizes: number[][] | undefined;
}

/** The spent/unknowable context. */
export const DROPPED: PipelineContext = { dims: undefined, chunkSizes: undefined };

/** Walk one codec pipeline, threading the dimensional context. */
export function pipelineIssues(
  pipeline: unknown[],
  path: Path,
  initialDims: number | undefined,
  initialChunkSizes: number[][] | undefined,
): PathedIssue[] {
  const issues: PathedIssue[] = [];
  let context: PipelineContext = { dims: initialDims, chunkSizes: initialChunkSizes };
  pipeline.forEach((entry, index) => {
    const parts = fieldParts(entry);
    if (parts === undefined) {
      context = DROPPED; // structurally invalid entry: no further reasoning
      return;
    }
    const configMissing = configurationMissing(entry);
    if (parts.name === "transpose") {
      const result = transposeIssues(parts.configuration, configMissing, path, index, context);
      issues.push(...result.issues);
      context = result.context;
    } else if (parts.name === "sharding_indexed") {
      const result = shardingIndexedIssues(
        parts.configuration,
        configMissing,
        path,
        index,
        context,
        pipelineIssues,
      );
      issues.push(...result.issues);
      context = result.context;
    } else {
      // A codec this package cannot reason about: stale array-level facts
      // must not judge the codecs after it.
      context = DROPPED;
    }
  });
  return issues;
}
