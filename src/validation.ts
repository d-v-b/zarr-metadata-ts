/**
 * Structural validation for Zarr metadata documents.
 *
 * A faithful port of `zarr_metadata.model._validation` (the Python reference
 * implementation). Validators check JSON structure (key presence, value
 * shapes, and fixed literals like `zarr_format`), not domain validity. Each
 * concept gets a `validate*` function returning every problem found, an
 * `is*` type guard, and a `parse*` function that narrows or throws
 * `MetadataValidationError`.
 *
 * Two Python behaviors have no JS analog and are intentionally absent:
 *
 * - tuple-vs-list canonicalization (`arrays_to_tuples`): JSON arrays are
 *   plain arrays in JS, so there is nothing to normalize;
 * - int-vs-float literal spelling (`zarr_format: 3.0` vs `3`): `JSON.parse`
 *   collapses both to the number `3`, so JS cannot reject the float
 *   spelling. The shared conformance corpus avoids fixtures that hinge on
 *   this distinction.
 */

import type { JSONValue, ZarrV3MetadataFieldJSON } from "./common.js";
import {
  MetadataValidationError,
  prefix,
  problem,
  type ValidationProblem,
} from "./problems.js";
import {
  ARRAY_METADATA_REQUIRED_KEYS_V2,
  ZARR_V2_ARRAY_DIMENSION_SEPARATOR,
  ZARR_V2_ARRAY_ORDER,
  ARRAY_METADATA_STANDARD_KEYS_V2,
  GROUP_METADATA_REQUIRED_KEYS_V2,
  GROUP_METADATA_STANDARD_KEYS_V2,
  type ZarrV2ArrayMetadataJSON,
  type ZarrV2ConsolidatedMetadataJSON,
  type ZarrV2GroupMetadataJSON,
} from "./v2.js";
import {
  ARRAY_METADATA_REQUIRED_KEYS_V3,
  ARRAY_METADATA_STANDARD_KEYS_V3,
  GROUP_METADATA_REQUIRED_KEYS_V3,
  GROUP_METADATA_STANDARD_KEYS_V3,
  ZARR_V3_CONSOLIDATED_METADATA_KEY,
  type ZarrV3ArrayMetadataJSON,
  type ZarrV3ConsolidatedMetadataJSON,
  type ZarrV3GroupMetadataJSON,
} from "./v3.js";

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function show(value: unknown): string {
  if (value === undefined) return "undefined";
  const rendered = JSON.stringify(value);
  return rendered === undefined ? String(value) : rendered;
}

/** Return every reason `value` is not JSON-serializable (recursively). */
export function validateJson(value: unknown): ValidationProblem[] {
  if (typeof value === "number") {
    if (Number.isFinite(value)) return [];
    return [problem([], `non-finite number ${value} is not JSON`, "invalid_value")];
  }
  if (typeof value === "string" || typeof value === "boolean" || value === null) {
    return [];
  }
  const problems: ValidationProblem[] = [];
  if (Array.isArray(value)) {
    value.forEach((item, index) => {
      problems.push(...prefix(index, validateJson(item)));
    });
    return problems;
  }
  if (isPlainObject(value)) {
    for (const [key, item] of Object.entries(value)) {
      problems.push(...prefix(key, validateJson(item)));
    }
    return problems;
  }
  return [problem([], `not a JSON-serializable value: ${show(value)}`, "invalid_type")];
}

/** Whether `value` is a JSON structure (recursively). */
export function isJson(value: unknown): value is JSONValue {
  return validateJson(value).length === 0;
}

/** Return `value` narrowed to `JSONValue`, or throw `MetadataValidationError`. */
export function parseJson(value: unknown): JSONValue {
  const problems = validateJson(value);
  if (problems.length > 0) throw new MetadataValidationError(problems);
  return value as JSONValue;
}

/** One `missing_key` problem per required key absent from `doc`. */
function missingKeys(
  required: ReadonlyArray<string>,
  doc: Record<string, unknown>,
): ValidationProblem[] {
  return [...required]
    .filter((key) => !(key in doc))
    .sort()
    .map((key) => problem([key], "missing required key", "missing_key"));
}

/** One problem per member outside a closed document's declared shape. */
function unexpectedKeys(
  allowed: ReadonlyArray<string>,
  doc: Record<string, unknown>,
): ValidationProblem[] {
  return Object.keys(doc)
    .filter((key) => !allowed.includes(key))
    .map((key) => problem([key], "unexpected document member", "invalid_value"));
}

/** One `invalid_value` problem if `doc[key]` is present but not `expected`. */
function checkLiteral(
  doc: Record<string, unknown>,
  key: string,
  expected: string | number | boolean,
): ValidationProblem[] {
  if (key in doc && (typeof doc[key] !== typeof expected || doc[key] !== expected)) {
    return [
      problem([key], `expected ${show(expected)}, got ${show(doc[key])}`, "invalid_value"),
    ];
  }
  return [];
}

/** Validate v3 top-level unknown-field JSON payloads. */
function validateExtensionFieldsV3(
  doc: Record<string, unknown>,
  standardKeys: ReadonlyArray<string>,
  additionalReservedKeys: ReadonlyArray<string> = [],
): ValidationProblem[] {
  const reserved = new Set([...standardKeys, ...additionalReservedKeys]);
  const problems: ValidationProblem[] = [];
  for (const [key, value] of Object.entries(doc)) {
    if (reserved.has(key)) continue;
    problems.push(...prefix(key, validateJson(value)));
  }
  return problems;
}

/**
 * Return every reason `value` is not a v3 metadata field.
 *
 * A metadata field is a bare name string or an object containing `name` and
 * optional `configuration` and `must_understand` members.
 */
export function validateMetadataFieldV3(
  value: unknown,
  options: { allowMustUnderstandFalse?: boolean } = {},
): ValidationProblem[] {
  const { allowMustUnderstandFalse = true } = options;
  if (typeof value === "string") return [];
  if (!isPlainObject(value)) {
    return [
      problem([], "expected a metadata field (string or extension object)", "invalid_type"),
    ];
  }
  const problems: ValidationProblem[] = [];
  const allowedKeys = new Set(["name", "configuration", "must_understand"]);
  for (const key of Object.keys(value)) {
    if (!allowedKeys.has(key)) {
      problems.push(problem([key], "unexpected metadata field member", "invalid_value"));
    }
  }
  if (typeof value["name"] !== "string") {
    problems.push(problem(["name"], "expected a string name", "invalid_type"));
  }
  if ("configuration" in value) {
    const configuration = value["configuration"];
    if (!isPlainObject(configuration)) {
      problems.push(problem(["configuration"], "expected a mapping", "invalid_type"));
    } else {
      for (const [key, item] of Object.entries(configuration)) {
        problems.push(...prefix("configuration", prefix(key, validateJson(item))));
      }
    }
  }
  if ("must_understand" in value) {
    const mustUnderstand = value["must_understand"];
    if (typeof mustUnderstand !== "boolean") {
      problems.push(problem(["must_understand"], "expected a boolean", "invalid_type"));
    } else if (!allowMustUnderstandFalse && !mustUnderstand) {
      problems.push(
        problem(
          ["must_understand"],
          "false is not supported at this extension point",
          "invalid_value",
        ),
      );
    }
  }
  return problems;
}

/** Whether `value` is a v3 metadata field: a bare name or a named config. */
export function isMetadataFieldV3(value: unknown): value is ZarrV3MetadataFieldJSON {
  return validateMetadataFieldV3(value).length === 0;
}

/** Return `value` narrowed to `ZarrV3MetadataFieldJSON`, or throw. */
export function parseMetadataFieldV3(value: unknown): ZarrV3MetadataFieldJSON {
  const problems = validateMetadataFieldV3(value);
  if (problems.length > 0) throw new MetadataValidationError(problems);
  return value as ZarrV3MetadataFieldJSON;
}

/**
 * Whether `value` is an array of integers.
 *
 * `Number.isInteger` rejects booleans, non-finite numbers, and non-integral
 * floats, mirroring the Python bool-is-not-int rule.
 */
function isIntSequence(value: unknown): value is number[] {
  return Array.isArray(value) && value.every((item) => Number.isInteger(item));
}

/**
 * Validate a dimension sequence (`shape` / `chunks`) if present in `doc`.
 * Dimension lengths are non-negative integers.
 */
function validateDimSequence(doc: Record<string, unknown>, key: string): ValidationProblem[] {
  if (!(key in doc)) return [];
  const value = doc[key];
  if (!isIntSequence(value)) {
    return [problem([key], "expected a sequence of int", "invalid_type")];
  }
  if (value.some((item) => item < 0)) {
    return [problem([key], "expected non-negative integers", "invalid_value")];
  }
  return [];
}

/**
 * Whether `value` is shaped like a v2 dtype: a string or field records.
 *
 * A field record is a `[name, dtype]` or `[name, dtype, shape]` array, where
 * `dtype` is itself a string or nested field records and `shape` is an array
 * of int. The string content is NOT interpreted — whether the string names a
 * real dtype is domain validity, not structure.
 */
function isDtypeV2(value: unknown): boolean {
  if (typeof value === "string") return true;
  if (!Array.isArray(value)) return false;
  for (const record of value) {
    if (typeof record === "string" || !Array.isArray(record)) return false;
    if (record.length !== 2 && record.length !== 3) return false;
    if (typeof record[0] !== "string") return false;
    if (!isDtypeV2(record[1])) return false;
    if (record.length === 3 && !isIntSequence(record[2])) return false;
  }
  return true;
}

/** Whether `value` is shaped like a v2 codec config: an object with a string `id`. */
function isCodecV2(value: unknown): boolean {
  return isPlainObject(value) && typeof value["id"] === "string";
}

/** Validate a v2 codec's required shape and JSON-valued configuration. */
function validateCodecV2(value: unknown): ValidationProblem[] {
  if (!isCodecV2(value)) {
    return [problem([], "expected a codec configuration with a string 'id'", "invalid_type")];
  }
  return validateJson(value);
}

/**
 * Validate an `attributes` value: a JSON object.
 *
 * Unlike the other validators (which return value-relative locs for the
 * caller to prefix), this emits the already-parent-relative `["attributes"]`
 * loc, since it is only ever called with a document's `attributes` value.
 */
function validateAttributes(value: unknown): ValidationProblem[] {
  if (!isPlainObject(value)) {
    return [problem(["attributes"], "expected a mapping with string keys", "invalid_type")];
  }
  const problems: ValidationProblem[] = [];
  for (const [key, item] of Object.entries(value)) {
    problems.push(...prefix("attributes", prefix(key, validateJson(item))));
  }
  return problems;
}

/**
 * Return every reason `value` is not a structurally-valid v3 array doc.
 *
 * Checks structure, not domain validity. Unknown top-level keys are allowed
 * (they are extension fields).
 */
export function validateArrayMetadataV3(value: unknown): ValidationProblem[] {
  if (!isPlainObject(value)) {
    return [problem([], "expected a mapping", "invalid_type")];
  }
  const doc = value;
  const problems: ValidationProblem[] = missingKeys(ARRAY_METADATA_REQUIRED_KEYS_V3, doc);
  problems.push(...validateExtensionFieldsV3(doc, ARRAY_METADATA_STANDARD_KEYS_V3));
  problems.push(...checkLiteral(doc, "zarr_format", 3));
  problems.push(...checkLiteral(doc, "node_type", "array"));
  problems.push(...validateDimSequence(doc, "shape"));
  if ("fill_value" in doc) {
    problems.push(...prefix("fill_value", validateJson(doc["fill_value"])));
  }
  for (const key of ["data_type", "chunk_grid", "chunk_key_encoding"]) {
    if (key in doc) {
      problems.push(
        ...prefix(
          key,
          validateMetadataFieldV3(doc[key], { allowMustUnderstandFalse: false }),
        ),
      );
    }
  }
  for (const key of ["codecs", "storage_transformers"]) {
    if (key in doc) {
      const entries = doc[key];
      if (!Array.isArray(entries)) {
        problems.push(problem([key], "expected a sequence", "invalid_type"));
      } else {
        if (key === "codecs" && entries.length === 0) {
          problems.push(problem(["codecs"], "expected at least one codec", "invalid_value"));
        }
        entries.forEach((entry, index) => {
          problems.push(...prefix(key, prefix(index, validateMetadataFieldV3(entry))));
        });
      }
    }
  }
  if ("attributes" in doc) {
    problems.push(...validateAttributes(doc["attributes"]));
  }
  if ("dimension_names" in doc) {
    // Simple typed sequences (dimension_names, shape, chunks) report a single
    // field-level loc, not per-bad-item locs; per-index locs are reserved for
    // the metadata-field lists (codecs, storage_transformers).
    const names = doc["dimension_names"];
    if (!Array.isArray(names)) {
      problems.push(problem(["dimension_names"], "expected a sequence", "invalid_type"));
    } else if (!names.every((item) => item === null || typeof item === "string")) {
      problems.push(
        problem(["dimension_names"], "expected items of str or None", "invalid_type"),
      );
    } else if (isIntSequence(doc["shape"]) && names.length !== doc["shape"].length) {
      problems.push(
        problem(["dimension_names"], "expected one name per dimension of shape", "invalid_value"),
      );
    }
  }
  return problems;
}

/** Whether `value` is a structurally-valid v3 array metadata document. */
export function isArrayMetadataV3(value: unknown): value is ZarrV3ArrayMetadataJSON {
  return validateArrayMetadataV3(value).length === 0;
}

/** Return `value` as `ZarrV3ArrayMetadataJSON`, or throw `MetadataValidationError`. */
export function parseArrayMetadataV3(value: unknown): ZarrV3ArrayMetadataJSON {
  const problems = validateArrayMetadataV3(value);
  if (problems.length > 0) throw new MetadataValidationError(problems);
  return value as ZarrV3ArrayMetadataJSON;
}

/**
 * Return every reason `value` is not a valid inline consolidated envelope.
 *
 * Locs are value-relative (the caller prefixes with `consolidated_metadata`
 * where appropriate). Entries recurse into the array and group document
 * validators.
 */
export function validateConsolidatedMetadataV3(value: unknown): ValidationProblem[] {
  if (!isPlainObject(value)) {
    return [problem([], "expected a mapping", "invalid_type")];
  }
  const env = value;
  const problems: ValidationProblem[] = ["kind", "must_understand", "metadata"]
    .filter((key) => !(key in env))
    .map((key) => problem([key], "missing required key", "missing_key"));
  problems.push(...unexpectedKeys(["kind", "must_understand", "metadata"], env));
  problems.push(...checkLiteral(env, "kind", "inline"));
  if ("must_understand" in env && env["must_understand"] !== false) {
    problems.push(problem(["must_understand"], "expected False", "invalid_value"));
  }
  if ("metadata" in env) {
    const entries = env["metadata"];
    if (!isPlainObject(entries)) {
      problems.push(problem(["metadata"], "expected a mapping", "invalid_type"));
    } else {
      for (const [key, entry] of Object.entries(entries)) {
        const nodeType = isPlainObject(entry) ? entry["node_type"] : undefined;
        if (nodeType === "array") {
          problems.push(...prefix("metadata", prefix(key, validateArrayMetadataV3(entry))));
        } else if (nodeType === "group") {
          problems.push(...prefix("metadata", prefix(key, validateGroupMetadataV3(entry))));
        } else {
          problems.push(
            problem(["metadata", key, "node_type"], "expected 'array' or 'group'", "invalid_value"),
          );
        }
      }
    }
  }
  return problems;
}

/**
 * Return every reason `value` is not a structurally-valid v3 group doc.
 *
 * Checks structure, not domain validity. Unknown top-level keys are allowed
 * (extension fields); a `consolidated_metadata` key, if present, is
 * deep-validated (envelope and entries).
 */
export function validateGroupMetadataV3(value: unknown): ValidationProblem[] {
  if (!isPlainObject(value)) {
    return [problem([], "expected a mapping", "invalid_type")];
  }
  const doc = value;
  const problems: ValidationProblem[] = missingKeys(GROUP_METADATA_REQUIRED_KEYS_V3, doc);
  problems.push(
    ...validateExtensionFieldsV3(doc, GROUP_METADATA_STANDARD_KEYS_V3, [
      ZARR_V3_CONSOLIDATED_METADATA_KEY,
    ]),
  );
  problems.push(...checkLiteral(doc, "zarr_format", 3));
  problems.push(...checkLiteral(doc, "node_type", "group"));
  if ("attributes" in doc) {
    problems.push(...validateAttributes(doc["attributes"]));
  }
  if (
    ZARR_V3_CONSOLIDATED_METADATA_KEY in doc &&
    doc[ZARR_V3_CONSOLIDATED_METADATA_KEY] !== null
  ) {
    // consolidated_metadata: null (a historical zarr-python bug) is
    // structurally accepted so those stores remain readable.
    problems.push(
      ...prefix(
        ZARR_V3_CONSOLIDATED_METADATA_KEY,
        validateConsolidatedMetadataV3(doc[ZARR_V3_CONSOLIDATED_METADATA_KEY]),
      ),
    );
  }
  return problems;
}

/** Whether `value` is a structurally-valid v3 group metadata document. */
export function isGroupMetadataV3(value: unknown): value is ZarrV3GroupMetadataJSON {
  return validateGroupMetadataV3(value).length === 0;
}

/** Return `value` narrowed to `ZarrV3GroupMetadataJSON`, or throw. */
export function parseGroupMetadataV3(value: unknown): ZarrV3GroupMetadataJSON {
  const problems = validateGroupMetadataV3(value);
  if (problems.length > 0) throw new MetadataValidationError(problems);
  return value as ZarrV3GroupMetadataJSON;
}

/** Whether `value` is a valid inline consolidated envelope. */
export function isConsolidatedMetadataV3(value: unknown): value is ZarrV3ConsolidatedMetadataJSON {
  return validateConsolidatedMetadataV3(value).length === 0;
}

/**
 * Return every reason `value` is not a structurally-valid v3 metadata
 * document of either node type (the complete `zarr.json` grammar).
 *
 * Dispatches on `node_type`: `"array"` and `"group"` route to the
 * corresponding document validator; anything else is itself the problem.
 * This dispatcher has no direct Python analog (consumers there pick a
 * validator per node type); it exists for consumers handed an arbitrary
 * `zarr.json`, like editor tooling.
 */
export function validateMetadataV3(value: unknown): ValidationProblem[] {
  if (!isPlainObject(value)) {
    return [problem([], "expected a mapping", "invalid_type")];
  }
  const nodeType = value["node_type"];
  if (nodeType === "array") return validateArrayMetadataV3(value);
  if (nodeType === "group") return validateGroupMetadataV3(value);
  if (!("node_type" in value)) {
    return [problem(["node_type"], "missing required key", "missing_key")];
  }
  return [
    problem(["node_type"], `expected 'array' or 'group', got ${show(nodeType)}`, "invalid_value"),
  ];
}

/**
 * Return every reason `value` is not a structurally-valid v2 array doc.
 *
 * Checks structure, not domain validity: `dtype` must be a string or field
 * records, but the string content is not interpreted; `compressor` and
 * `filters` are required keys that may be `null`, and otherwise must be
 * codec configurations (objects with a string `id`).
 */
export function validateArrayMetadataV2(value: unknown): ValidationProblem[] {
  if (!isPlainObject(value)) {
    return [problem([], "expected a mapping", "invalid_type")];
  }
  const doc = value;
  const problems: ValidationProblem[] = missingKeys(ARRAY_METADATA_REQUIRED_KEYS_V2, doc);
  problems.push(...unexpectedKeys(ARRAY_METADATA_STANDARD_KEYS_V2, doc));
  problems.push(...checkLiteral(doc, "zarr_format", 2));
  const shapeProblems = validateDimSequence(doc, "shape");
  const chunksProblems = validateDimSequence(doc, "chunks");
  problems.push(...shapeProblems, ...chunksProblems);
  if (
    shapeProblems.length === 0 &&
    chunksProblems.length === 0 &&
    isIntSequence(doc["shape"]) &&
    isIntSequence(doc["chunks"]) &&
    doc["shape"].length !== doc["chunks"].length
  ) {
    problems.push(
      problem(["chunks"], "expected the same number of dimensions as shape", "invalid_value"),
    );
  }
  if ("dtype" in doc && !isDtypeV2(doc["dtype"])) {
    problems.push(
      problem(["dtype"], "expected a v2 dtype string or a sequence of field records", "invalid_type"),
    );
  }
  if ("order" in doc && !(ZARR_V2_ARRAY_ORDER as readonly unknown[]).includes(doc["order"])) {
    problems.push(
      problem(["order"], `expected 'C' or 'F', got ${show(doc["order"])}`, "invalid_value"),
    );
  }
  if ("compressor" in doc && doc["compressor"] !== null) {
    problems.push(...prefix("compressor", validateCodecV2(doc["compressor"])));
  }
  if ("filters" in doc) {
    const filters = doc["filters"];
    if (filters !== null && (!Array.isArray(filters) || !filters.every(isCodecV2))) {
      problems.push(
        problem(
          ["filters"],
          "expected null or a sequence of codec configurations with string 'id's",
          "invalid_type",
        ),
      );
    } else if (filters !== null && Array.isArray(filters)) {
      if (filters.length === 0) {
        problems.push(problem(["filters"], "expected at least one filter", "invalid_value"));
      }
      filters.forEach((item, index) => {
        problems.push(...prefix("filters", prefix(index, validateJson(item))));
      });
    }
  }
  if (
    "dimension_separator" in doc &&
    !(ZARR_V2_ARRAY_DIMENSION_SEPARATOR as readonly unknown[]).includes(doc["dimension_separator"])
  ) {
    problems.push(
      problem(
        ["dimension_separator"],
        `expected '.' or '/', got ${show(doc["dimension_separator"])}`,
        "invalid_value",
      ),
    );
  }
  if ("fill_value" in doc) {
    problems.push(...prefix("fill_value", validateJson(doc["fill_value"])));
  }
  if ("attributes" in doc) {
    problems.push(...validateAttributes(doc["attributes"]));
  }
  return problems;
}

/** Whether `value` is a structurally-valid v2 array metadata document. */
export function isArrayMetadataV2(value: unknown): value is ZarrV2ArrayMetadataJSON {
  return validateArrayMetadataV2(value).length === 0;
}

/** Return `value` as `ZarrV2ArrayMetadataJSON`, or throw `MetadataValidationError`. */
export function parseArrayMetadataV2(value: unknown): ZarrV2ArrayMetadataJSON {
  const problems = validateArrayMetadataV2(value);
  if (problems.length > 0) throw new MetadataValidationError(problems);
  return value as ZarrV2ArrayMetadataJSON;
}

/**
 * Return every reason `value` is not a structurally-valid v2 group doc.
 *
 * Validates the in-memory merged form: the `.zgroup` fields plus an optional
 * `attributes` mapping folded in from `.zattrs`.
 */
export function validateGroupMetadataV2(value: unknown): ValidationProblem[] {
  if (!isPlainObject(value)) {
    return [problem([], "expected a mapping", "invalid_type")];
  }
  const doc = value;
  const problems: ValidationProblem[] = missingKeys(GROUP_METADATA_REQUIRED_KEYS_V2, doc);
  problems.push(...unexpectedKeys(GROUP_METADATA_STANDARD_KEYS_V2, doc));
  problems.push(...checkLiteral(doc, "zarr_format", 2));
  if ("attributes" in doc) {
    problems.push(...validateAttributes(doc["attributes"]));
  }
  return problems;
}

/** Whether `value` is a structurally-valid v2 group metadata document. */
export function isGroupMetadataV2(value: unknown): value is ZarrV2GroupMetadataJSON {
  return validateGroupMetadataV2(value).length === 0;
}

/** Return `value` narrowed to `ZarrV2GroupMetadataJSON`, or throw. */
export function parseGroupMetadataV2(value: unknown): ZarrV2GroupMetadataJSON {
  const problems = validateGroupMetadataV2(value);
  if (problems.length > 0) throw new MetadataValidationError(problems);
  return value as ZarrV2GroupMetadataJSON;
}

/**
 * Return every reason `value` is not a structurally-valid `.zmetadata` doc
 * (v2 consolidated metadata).
 *
 * Ported from `ZarrV2ConsolidatedMetadata.from_json` in the Python reference
 * implementation. Entries are validated as JSON trees, not as per-node
 * documents: which nodes had a `.zattrs` file at all is information the
 * canonical representation must keep, and interpreting entries into node
 * documents is consumer work.
 */
export function validateConsolidatedMetadataV2(value: unknown): ValidationProblem[] {
  if (!isPlainObject(value)) {
    return [problem([], "expected a mapping", "invalid_type")];
  }
  const doc = value;
  const problems: ValidationProblem[] = ["zarr_consolidated_format", "metadata"]
    .filter((key) => !(key in doc))
    .map((key) => problem([key], "missing required key", "missing_key"));
  problems.push(...unexpectedKeys(["zarr_consolidated_format", "metadata"], doc));
  problems.push(...checkLiteral(doc, "zarr_consolidated_format", 1));
  if ("metadata" in doc) {
    const entries = doc["metadata"];
    if (!isPlainObject(entries)) {
      problems.push(problem(["metadata"], "expected a mapping with string keys", "invalid_type"));
    } else {
      for (const [key, item] of Object.entries(entries)) {
        problems.push(...prefix("metadata", prefix(key, validateJson(item))));
      }
    }
  }
  return problems;
}

/** Whether `value` is a structurally-valid v2 consolidated metadata document. */
export function isConsolidatedMetadataV2(
  value: unknown,
): value is ZarrV2ConsolidatedMetadataJSON {
  return validateConsolidatedMetadataV2(value).length === 0;
}

/** Return `value` as `ZarrV2ConsolidatedMetadataJSON`, or throw. */
export function parseConsolidatedMetadataV2(value: unknown): ZarrV2ConsolidatedMetadataJSON {
  const problems = validateConsolidatedMetadataV2(value);
  if (problems.length > 0) throw new MetadataValidationError(problems);
  return value as ZarrV2ConsolidatedMetadataJSON;
}
