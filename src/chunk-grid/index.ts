/**
 * The chunk grids this package interprets, one module per grid, assembled
 * into the dispatch the semantic layer uses. The structural document layer
 * never reads these modules — extension points are uninterpreted there by
 * spec design.
 */
import type { PathedIssue } from "../errors.js";
import { fieldParts, isPlainObject } from "../guards.js";
import { regularIssues } from "./regular.js";
import { rectilinearIssues } from "./rectilinear.js";

export type { RegularChunkGridConfiguration } from "./regular.js";
export type { RectilinearChunkGridConfiguration, RectilinearDimSpec } from "./rectilinear.js";

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
