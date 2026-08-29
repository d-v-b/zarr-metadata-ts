/**
 * zarr-metadata: spec-defined metadata types and validators for Zarr v2 and
 * v3, in TypeScript.
 *
 * A port of the Python `zarr-metadata` package (the reference
 * implementation), kept in lockstep by a shared conformance corpus.
 */

export type { JSONValue, ZarrV3MetadataFieldJSON, ZarrV3NamedConfigJSON } from "./common.js";
export {
  flattenTree,
  formatTree,
  isEmptyTree,
  MetadataValidationError,
  treeOf,
  type ErrorTree,
  type Issue,
  type IssueKind,
  type IssuePath,
  type ParseResult,
  type PathedIssue,
} from "./errors.js";
export * from "./schemas.js";
export type { StandardSchemaV1 } from "./standard-schema.js";
export * from "./v2.js";
export * from "./v3.js";
export * from "./validation.js";
