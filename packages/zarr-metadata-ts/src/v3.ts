/**
 * Zarr v3 metadata document types.
 *
 * See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html
 */

import type { JSONValue, ZarrV3MetadataFieldJSON } from "./common.js";

/**
 * The JSON value of an unknown top-level v3 metadata field.
 *
 * An object carrying the literal member `must_understand: false` may be
 * ignored. Every other JSON shape implicitly requires understanding;
 * recognition itself belongs to the reader rather than this structural type.
 */
export type ZarrV3ExtensionField = JSONValue;

/**
 * Zarr v3 array metadata document (the `zarr.json` content for an array).
 *
 * Extra keys may contain arbitrary JSON values (extension fields).
 *
 * See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#array-metadata
 */
export type ZarrV3ArrayMetadataJSON = {
  zarr_format: 3;
  node_type: "array";
  data_type: ZarrV3MetadataFieldJSON;
  shape: number[];
  chunk_grid: ZarrV3MetadataFieldJSON;
  chunk_key_encoding: ZarrV3MetadataFieldJSON;
  fill_value: JSONValue;
  codecs: ZarrV3MetadataFieldJSON[];
  attributes?: { [key: string]: JSONValue };
  storage_transformers?: ZarrV3MetadataFieldJSON[];
  dimension_names?: (string | null)[];
} & { [key: string]: JSONValue | undefined };

/**
 * Zarr v3 group metadata document (the `zarr.json` content for a group).
 *
 * Extra keys may contain arbitrary JSON values (extension fields).
 *
 * See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#group-metadata
 */
export type ZarrV3GroupMetadataJSON = {
  zarr_format: 3;
  node_type: "group";
  attributes?: { [key: string]: JSONValue };
} & { [key: string]: JSONValue | undefined };

/**
 * Inline consolidated metadata embedded in a v3 group.
 *
 * There is no Zarr v3 specification for consolidated metadata; this models
 * the inline-on-group convention used by the reference Python implementation
 * (and zarrs), where consolidated metadata is embedded as an extension field
 * on a group's `zarr.json`.
 */
export type ZarrV3ConsolidatedMetadataJSON = {
  kind: "inline";
  must_understand: false;
  metadata: { [key: string]: ZarrV3ArrayMetadataJSON | ZarrV3GroupMetadataJSON };
};

/**
 * The store key both v3 node types persist their metadata document under.
 * The document's `node_type` field distinguishes an array from a group.
 */
export const ZARR_V3_METADATA_STORE_KEY = "zarr.json";

/** The key under which consolidated metadata is embedded in a v3 group document. */
export const ZARR_V3_CONSOLIDATED_METADATA_KEY = "consolidated_metadata";

/** The standard top-level keys of a v3 array metadata document. */
export const ARRAY_METADATA_REQUIRED_KEYS_V3: ReadonlyArray<string> = [
  "zarr_format",
  "node_type",
  "data_type",
  "shape",
  "chunk_grid",
  "chunk_key_encoding",
  "fill_value",
  "codecs",
];
export const ARRAY_METADATA_OPTIONAL_KEYS_V3: ReadonlyArray<string> = [
  "attributes",
  "storage_transformers",
  "dimension_names",
];
export const ARRAY_METADATA_STANDARD_KEYS_V3: ReadonlyArray<string> = [
  ...ARRAY_METADATA_REQUIRED_KEYS_V3,
  ...ARRAY_METADATA_OPTIONAL_KEYS_V3,
];

/** The standard top-level keys of a v3 group metadata document. */
export const GROUP_METADATA_REQUIRED_KEYS_V3: ReadonlyArray<string> = [
  "zarr_format",
  "node_type",
];
export const GROUP_METADATA_OPTIONAL_KEYS_V3: ReadonlyArray<string> = ["attributes"];
export const GROUP_METADATA_STANDARD_KEYS_V3: ReadonlyArray<string> = [
  ...GROUP_METADATA_REQUIRED_KEYS_V3,
  ...GROUP_METADATA_OPTIONAL_KEYS_V3,
];
