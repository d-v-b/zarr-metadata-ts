/**
 * Zarr v2 metadata document types.
 *
 * See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
 */

import type { JSONValue } from "./common.js";

/**
 * The v2 dtype representation.
 *
 * Either a numpy-style dtype string (e.g. `"<f8"`, `"|S10"`) or an array of
 * field records describing a structured dtype. Each field record is either a
 * 2-tuple `[name, datatype]` or a 3-tuple `[name, datatype, shape]` (the
 * 3-tuple form indicates a subarray field). A field datatype may itself be
 * another structured dtype.
 *
 * See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html#data-type-encoding
 */
export type ZarrV2DataTypeMetadata =
  | string
  | Array<[string, ZarrV2DataTypeMetadata] | [string, ZarrV2DataTypeMetadata, number[]]>;

/** `"C"` (row-major) or `"F"` (column-major) — the in-chunk byte layout. */
export type ZarrV2ArrayOrder = "C" | "F";
export const ZARR_V2_ARRAY_ORDER: ReadonlyArray<ZarrV2ArrayOrder> = ["C", "F"];

/** `"."` (legacy default) or `"/"` (nested directories). */
export type ZarrV2ArrayDimensionSeparator = "." | "/";
export const ZARR_V2_ARRAY_DIMENSION_SEPARATOR: ReadonlyArray<ZarrV2ArrayDimensionSeparator> =
  [".", "/"];

/**
 * A numcodecs configuration object, used as a v2 compressor or filter.
 *
 * The required `id` field names the codec; codec-specific parameters
 * (e.g. `cname`, `clevel` for blosc) appear as extra fields.
 */
export type ZarrV2CodecMetadata = { id: string } & { [key: string]: JSONValue | undefined };

/**
 * On-disk `.zarray` file content.
 *
 * User attributes live in a sibling `.zattrs` file and are NOT part of this
 * type; see `ZarrV2ZAttrsJSON`.
 */
export type ZarrV2ZArrayJSON = {
  zarr_format: 2;
  shape: number[];
  chunks: number[];
  dtype: ZarrV2DataTypeMetadata;
  compressor: ZarrV2CodecMetadata | null;
  fill_value: JSONValue;
  order: ZarrV2ArrayOrder;
  filters: ZarrV2CodecMetadata[] | null;
  dimension_separator?: ZarrV2ArrayDimensionSeparator;
};

/**
 * Zarr v2 array metadata document, in-memory merged form: the `.zarray`
 * fields plus the sibling `.zattrs` attributes folded in as `attributes`.
 */
export type ZarrV2ArrayMetadataJSON = ZarrV2ZArrayJSON & {
  attributes?: { [key: string]: JSONValue };
};

/** On-disk `.zgroup` file content. The spec defines exactly one field. */
export type ZarrV2ZGroupJSON = {
  zarr_format: 2;
};

/**
 * Zarr v2 group metadata document, in-memory merged form: the `.zgroup`
 * field plus the sibling `.zattrs` attributes folded in as `attributes`.
 */
export type ZarrV2GroupMetadataJSON = ZarrV2ZGroupJSON & {
  attributes?: { [key: string]: JSONValue };
};

/** On-disk `.zattrs` file content: a JSON object of user attributes. */
export type ZarrV2ZAttrsJSON = { [key: string]: JSONValue };

/**
 * `.zmetadata` file contents (v2 consolidated metadata).
 *
 * NOT a spec artifact: a reference-implementation convention. The `metadata`
 * map uses flat path keys (`"foo/bar/.zarray"`, `"foo/.zattrs"`, ...)
 * pointing to the JSON contents of the file at that path.
 */
export type ZarrV2ConsolidatedMetadataJSON = {
  zarr_consolidated_format: 1;
  metadata: { [key: string]: JSONValue };
};

export const ZARR_V2_ARRAY_METADATA_STORE_KEY = ".zarray";
export const ZARR_V2_GROUP_METADATA_STORE_KEY = ".zgroup";
export const ZARR_V2_ATTRIBUTES_STORE_KEY = ".zattrs";
export const ZARR_V2_CONSOLIDATED_METADATA_STORE_KEY = ".zmetadata";

/** The standard top-level keys of a merged v2 array metadata document. */
export const ARRAY_METADATA_REQUIRED_KEYS_V2: ReadonlyArray<string> = [
  "zarr_format",
  "shape",
  "chunks",
  "dtype",
  "compressor",
  "fill_value",
  "order",
  "filters",
];
export const ARRAY_METADATA_OPTIONAL_KEYS_V2: ReadonlyArray<string> = [
  "dimension_separator",
  "attributes",
];
export const ARRAY_METADATA_STANDARD_KEYS_V2: ReadonlyArray<string> = [
  ...ARRAY_METADATA_REQUIRED_KEYS_V2,
  ...ARRAY_METADATA_OPTIONAL_KEYS_V2,
];

/** The standard top-level keys of a merged v2 group metadata document. */
export const GROUP_METADATA_REQUIRED_KEYS_V2: ReadonlyArray<string> = ["zarr_format"];
export const GROUP_METADATA_OPTIONAL_KEYS_V2: ReadonlyArray<string> = ["attributes"];
export const GROUP_METADATA_STANDARD_KEYS_V2: ReadonlyArray<string> = [
  ...GROUP_METADATA_REQUIRED_KEYS_V2,
  ...GROUP_METADATA_OPTIONAL_KEYS_V2,
];
