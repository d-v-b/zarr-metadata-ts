/**
 * Semantic (cross-field) validation for v3 array metadata: the
 * orchestrator. The rules themselves live with their content —
 * `chunk-grids.ts`, `codecs.ts`, and `data-types.ts` each hold the syntax
 * (types) and semantics (procedures) for the names this package
 * interprets, mirroring how the spec, the Python reference's type layout,
 * and the zarr-extensions registry are organized. This module only walks
 * documents: extract shape, ask each content module for its verdicts,
 * thread the chunk-size context into codec pipelines, and descend into a
 * group's inline consolidated entries.
 *
 * Unrecognized names are skipped everywhere — the extension name space is
 * open, and a rule that guessed would lie. This layer has no counterpart
 * in the Python reference implementation (which stops at structure), so it
 * is covered by this package's own tests rather than the conformance
 * corpus. The structural document layer (validation.ts) stays
 * content-agnostic and corpus-governed; it never reads the content
 * modules.
 */

import { chunkGridVerdict } from "./chunk-grids.js";
import { pipelineIssues } from "./codecs.js";
import { fillValueIssue } from "./data-types.js";
import { treeOf, type ErrorTree, type PathedIssue } from "./errors.js";
import { fieldParts, isIntArray, isPlainObject } from "./guards.js";

function arraySemanticsIssues(value: unknown): PathedIssue[] {
  if (!isPlainObject(value) || value["node_type"] !== "array") return [];
  const issues: PathedIssue[] = [];
  const shape = isIntArray(value["shape"]) ? value["shape"] : undefined;

  const grid = chunkGridVerdict(value["chunk_grid"], shape);
  issues.push(
    ...grid.issues.map((issue) => ({ ...issue, path: ["chunk_grid", ...issue.path] })),
  );

  const dataType = fieldParts(value["data_type"]);
  if (dataType !== undefined && "fill_value" in value) {
    const message = fillValueIssue(dataType.name, value["fill_value"]);
    if (message !== undefined) {
      issues.push({ path: ["fill_value"], message, kind: "invalid_value" });
    }
  }

  const codecs = value["codecs"];
  if (Array.isArray(codecs)) {
    issues.push(...pipelineIssues(codecs, ["codecs"], shape?.length, grid.chunkSizes));
  }
  return issues;
}

/**
 * Every semantic (cross-field) problem in a v3 array metadata document, as
 * an error tree; an empty tree means no rule found a violation. Values that
 * are not v3 array documents yield an empty tree; run the structural
 * validators for structure. See `validateSemanticsV3` for the form that
 * also descends into a group's inline consolidated metadata.
 */
export function validateArraySemanticsV3(value: unknown): ErrorTree {
  return treeOf(arraySemanticsIssues(value));
}

// Matches the structural validators' MAX_JSON_DEPTH; the structural layer
// reports the depth violation, this layer just stops descending.
const MAX_CONSOLIDATED_DEPTH = 64;

function semanticsIssuesAtDepth(value: unknown, depth: number): PathedIssue[] {
  if (depth >= MAX_CONSOLIDATED_DEPTH || !isPlainObject(value)) return [];
  if (value["node_type"] === "array") return arraySemanticsIssues(value);
  if (value["node_type"] !== "group") return [];
  const consolidated = value["consolidated_metadata"];
  if (!isPlainObject(consolidated)) return [];
  const entries = consolidated["metadata"];
  if (!isPlainObject(entries)) return [];
  const issues: PathedIssue[] = [];
  for (const [key, entry] of Object.entries(entries)) {
    issues.push(
      ...semanticsIssuesAtDepth(entry, depth + 1).map((issue) => ({
        ...issue,
        path: ["consolidated_metadata", "metadata", key, ...issue.path],
      })),
    );
  }
  return issues;
}

/**
 * Every semantic problem in a v3 metadata document of either node type: an
 * array document gets the array rules, and a group document's inline
 * consolidated entries are checked recursively (each entry is a complete
 * array or group document, and nested groups may carry consolidated
 * metadata of their own). Issues inside entries are pathed through
 * `consolidated_metadata.metadata.<key>`.
 */
export function validateSemanticsV3(value: unknown): ErrorTree {
  return treeOf(semanticsIssuesAtDepth(value, 0));
}
