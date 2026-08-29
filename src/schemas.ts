/**
 * Standard Schema (https://standardschema.dev) objects for every document
 * kind, so the validators plug into anything that accepts standard schemas
 * (tRPC, form libraries, ...).
 *
 * Each schema wraps the corresponding `safeParse*` function: validation is
 * synchronous, success carries the narrowed document, and failure carries
 * the flattened error tree as standard issues. Every issue additionally
 * keeps its machine-readable `kind` (an extension the standard permits);
 * consumers wanting the full `ErrorTree` should use `validate*` directly.
 */

import { flattenTree, type IssueKind, type ParseResult, type PathedIssue } from "./errors.js";
import type { JSONValue, ZarrV3MetadataFieldJSON } from "./common.js";
import type { StandardSchemaV1 } from "./standard-schema.js";
import {
  safeParseArrayMetadataV2,
  safeParseArrayMetadataV3,
  safeParseConsolidatedMetadataV2,
  safeParseConsolidatedMetadataV3,
  safeParseGroupMetadataV2,
  safeParseGroupMetadataV3,
  safeParseJson,
  safeParseMetadataFieldV3,
  safeParseMetadataV3,
} from "./validation.js";
import type {
  ZarrV2ArrayMetadataJSON,
  ZarrV2ConsolidatedMetadataJSON,
  ZarrV2GroupMetadataJSON,
} from "./v2.js";
import type {
  ZarrV3ArrayMetadataJSON,
  ZarrV3ConsolidatedMetadataJSON,
  ZarrV3GroupMetadataJSON,
  ZarrV3MetadataJSON,
} from "./v3.js";

/** A standard issue that also keeps zarr-metadata's machine-readable kind. */
export interface StandardIssue extends StandardSchemaV1.Issue {
  readonly kind: IssueKind;
}

function toStandardIssue(issue: PathedIssue): StandardIssue {
  return issue.path.length === 0
    ? { message: issue.message, kind: issue.kind }
    : { message: issue.message, path: [...issue.path], kind: issue.kind };
}

function schemaOf<T>(
  safeParse: (value: unknown) => ParseResult<T>,
): StandardSchemaV1<unknown, T> {
  return {
    "~standard": {
      version: 1,
      vendor: "zarr-metadata",
      validate: (value) => {
        const result = safeParse(value);
        return result.success
          ? { value: result.value }
          : { issues: flattenTree(result.errors).map(toStandardIssue) };
      },
    },
  };
}

export const jsonValueSchema: StandardSchemaV1<unknown, JSONValue> = schemaOf(safeParseJson);

export const metadataFieldV3Schema: StandardSchemaV1<unknown, ZarrV3MetadataFieldJSON> =
  schemaOf(safeParseMetadataFieldV3);

export const arrayMetadataV3Schema: StandardSchemaV1<unknown, ZarrV3ArrayMetadataJSON> =
  schemaOf(safeParseArrayMetadataV3);

export const groupMetadataV3Schema: StandardSchemaV1<unknown, ZarrV3GroupMetadataJSON> =
  schemaOf(safeParseGroupMetadataV3);

export const consolidatedMetadataV3Schema: StandardSchemaV1<
  unknown,
  ZarrV3ConsolidatedMetadataJSON
> = schemaOf(safeParseConsolidatedMetadataV3);

/** The complete `zarr.json` grammar, dispatching on `node_type`. */
export const metadataV3Schema: StandardSchemaV1<unknown, ZarrV3MetadataJSON> =
  schemaOf(safeParseMetadataV3);

export const arrayMetadataV2Schema: StandardSchemaV1<unknown, ZarrV2ArrayMetadataJSON> =
  schemaOf(safeParseArrayMetadataV2);

export const groupMetadataV2Schema: StandardSchemaV1<unknown, ZarrV2GroupMetadataJSON> =
  schemaOf(safeParseGroupMetadataV2);

export const consolidatedMetadataV2Schema: StandardSchemaV1<
  unknown,
  ZarrV2ConsolidatedMetadataJSON
> = schemaOf(safeParseConsolidatedMetadataV2);
