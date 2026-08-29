/**
 * Semantic (cross-field) validation for v3 array metadata.
 *
 * The structural validators never interpret what an extension point names;
 * this module deliberately interprets a small set of WELL-KNOWN core
 * extension points to enforce the cross-field rules the specs state in
 * prose:
 *
 * - the `regular` chunk grid's `chunk_shape` has one length per dimension
 *   of `shape`;
 * - a `transpose` codec's `order` is a permutation of `0..n-1`, with one
 *   entry per array dimension;
 * - a `sharding_indexed` codec's inner `chunk_shape` matches the array's
 *   dimensionality and evenly divides the shape of the chunk it shards
 *   (the regular grid's chunk at the top level, the parent shard's inner
 *   chunk when sharding nests);
 * - `fill_value` has a JSON shape permitted for the named core data type
 *   (booleans for `bool`, ranged integers for the int types, numbers /
 *   the "NaN"-family sentinels / width-checked "0x…" strings for the
 *   float types, two-element arrays for the complex types).
 *
 * Unrecognized names are skipped everywhere — the extension name space is
 * open, and a rule that guessed would lie. This layer has no counterpart
 * in the Python reference implementation (which stops at structure), so it
 * is covered by this package's own tests rather than the conformance
 * corpus.
 */

import { treeOf, type ErrorTree, type PathedIssue } from "./errors.js";

type Path = ReadonlyArray<string | number>;

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const proto: unknown = Object.getPrototypeOf(value);
  return proto === null || proto === Object.prototype;
}

function isIntArray(value: unknown): value is number[] {
  return (
    Array.isArray(value) &&
    Object.keys(value).length === value.length &&
    value.every((item) => Number.isInteger(item))
  );
}

/** The `{name, configuration}` of a metadata field, when extractable. */
function fieldParts(
  field: unknown,
): { name: string; configuration: Record<string, unknown> | undefined } | undefined {
  if (typeof field === "string") return { name: field, configuration: undefined };
  if (isPlainObject(field) && typeof field["name"] === "string") {
    const configuration = field["configuration"];
    return {
      name: field["name"],
      configuration: isPlainObject(configuration) ? configuration : undefined,
    };
  }
  return undefined;
}

// --- fill_value vs data_type ----------------------------------------------

const INT_RANGES: Record<string, readonly [number, number]> = {
  int8: [-128, 127],
  int16: [-32768, 32767],
  int32: [-2147483648, 2147483647],
  int64: [-9223372036854775808, 9223372036854775807],
  uint8: [0, 255],
  uint16: [0, 65535],
  uint32: [0, 4294967295],
  uint64: [0, 18446744073709551615],
};

const FLOAT_HEX_DIGITS: Record<string, number> = {
  float16: 4,
  float32: 8,
  float64: 16,
};

/** Hex digit width of each complex type's component floats. */
const COMPLEX_COMPONENT_HEX_DIGITS: Record<string, number> = {
  complex64: 8,
  complex128: 16,
};

function isFloatFill(value: unknown, hexDigits: number): boolean {
  if (typeof value === "number") return true;
  if (typeof value !== "string") return false;
  if (value === "NaN" || value === "Infinity" || value === "-Infinity") return true;
  return new RegExp(`^0x[0-9a-fA-F]{${hexDigits}}$`).test(value);
}

function floatFillMessage(name: string, hexDigits: number): string {
  return (
    `expected a number, "NaN", "Infinity", "-Infinity", or a ` +
    `${hexDigits}-hex-digit "0x..." string for data type ${JSON.stringify(name)}`
  );
}

/** The fill_value issue for a KNOWN core data type, if any; undefined otherwise. */
function fillValueIssue(dataTypeName: string, fill: unknown): string | undefined {
  if (dataTypeName === "bool") {
    return typeof fill === "boolean"
      ? undefined
      : 'expected a boolean fill value for data type "bool"';
  }
  const intRange = INT_RANGES[dataTypeName];
  if (intRange !== undefined) {
    const [low, high] = intRange;
    return Number.isInteger(fill) && (fill as number) >= low && (fill as number) <= high
      ? undefined
      : `expected an integer in [${low}, ${high}] for data type ${JSON.stringify(dataTypeName)}`;
  }
  const floatDigits = FLOAT_HEX_DIGITS[dataTypeName];
  if (floatDigits !== undefined) {
    return isFloatFill(fill, floatDigits)
      ? undefined
      : floatFillMessage(dataTypeName, floatDigits);
  }
  const complexDigits = COMPLEX_COMPONENT_HEX_DIGITS[dataTypeName];
  if (complexDigits !== undefined) {
    const ok =
      Array.isArray(fill) &&
      fill.length === 2 &&
      fill.every((part) => isFloatFill(part, complexDigits));
    return ok
      ? undefined
      : `expected a two-element [real, imaginary] array for data type ${JSON.stringify(dataTypeName)}`;
  }
  return undefined; // unrecognized data type: no verdict
}

// --- codec pipelines (transpose / sharding) --------------------------------

function isPermutation(order: number[]): boolean {
  const seen = new Set(order);
  return seen.size === order.length && order.every((entry) => entry >= 0 && entry < order.length);
}

/**
 * Walk one codec pipeline. `dims` is the array dimensionality at this point
 * (undefined when unknowable); `outerChunkShape` is the shape of the chunk
 * this pipeline encodes — what a sharding codec's inner chunks must divide.
 */
function pipelineIssues(
  pipeline: unknown[],
  path: Path,
  dims: number | undefined,
  outerChunkShape: number[] | undefined,
): PathedIssue[] {
  const issues: PathedIssue[] = [];
  pipeline.forEach((entry, index) => {
    const parts = fieldParts(entry);
    if (parts === undefined) return;
    const { name, configuration } = parts;
    if (name === "transpose") {
      const order = configuration?.["order"];
      if (!isIntArray(order)) return; // shape errors are the schema layer's
      const orderPath = [...path, index, "configuration", "order"];
      if (!isPermutation(order)) {
        issues.push({
          path: orderPath,
          message: `expected a permutation of the integers 0..${order.length - 1}`,
          kind: "invalid_value",
        });
      }
      if (dims !== undefined && order.length !== dims) {
        issues.push({
          path: orderPath,
          message: `expected one entry per array dimension (${dims})`,
          kind: "invalid_value",
        });
      }
    } else if (name === "sharding_indexed") {
      const chunkShape = configuration?.["chunk_shape"];
      if (!isIntArray(chunkShape)) return;
      const chunkShapePath = [...path, index, "configuration", "chunk_shape"];
      if (dims !== undefined && chunkShape.length !== dims) {
        issues.push({
          path: chunkShapePath,
          message: `expected one length per array dimension (${dims})`,
          kind: "invalid_value",
        });
      } else if (
        outerChunkShape !== undefined &&
        chunkShape.length === outerChunkShape.length &&
        chunkShape.every((length) => length > 0) &&
        outerChunkShape.some((outer, axis) => outer % (chunkShape[axis] as number) !== 0)
      ) {
        issues.push({
          path: chunkShapePath,
          message: `expected ${JSON.stringify(chunkShape)} to evenly divide the outer chunk shape ${JSON.stringify(outerChunkShape)}`,
          kind: "invalid_value",
        });
      }
      const inner = configuration?.["codecs"];
      if (Array.isArray(inner)) {
        issues.push(
          ...pipelineIssues(inner, [...path, index, "configuration", "codecs"], dims, chunkShape),
        );
      }
      // index_codecs encode the shard index, whose shape differs from the
      // array's — no dimensional context applies there.
    }
  });
  return issues;
}

/**
 * Every semantic (cross-field) problem in a v3 array metadata document, as
 * an error tree; an empty tree means no rule found a violation. Values that
 * are not v3 array documents yield an empty tree; run the structural
 * validators for structure.
 */
export function validateArraySemanticsV3(value: unknown): ErrorTree {
  if (!isPlainObject(value) || value["node_type"] !== "array") return treeOf([]);
  const issues: PathedIssue[] = [];
  const shape = isIntArray(value["shape"]) ? value["shape"] : undefined;
  const dims = shape?.length;

  const grid = fieldParts(value["chunk_grid"]);
  let chunkShape: number[] | undefined;
  if (grid?.name === "regular") {
    const configured = grid.configuration?.["chunk_shape"];
    if (isIntArray(configured)) {
      chunkShape = configured;
      if (dims !== undefined && configured.length !== dims) {
        issues.push({
          path: ["chunk_grid", "configuration", "chunk_shape"],
          message: `expected one length per dimension of shape (${dims})`,
          kind: "invalid_value",
        });
        chunkShape = undefined; // wrong arity: unusable as division context
      }
    }
  }

  const dataType = fieldParts(value["data_type"]);
  if (dataType !== undefined && "fill_value" in value) {
    const message = fillValueIssue(dataType.name, value["fill_value"]);
    if (message !== undefined) {
      issues.push({ path: ["fill_value"], message, kind: "invalid_value" });
    }
  }

  const codecs = value["codecs"];
  if (Array.isArray(codecs)) {
    issues.push(...pipelineIssues(codecs, ["codecs"], dims, chunkShape));
  }
  return treeOf(issues);
}
