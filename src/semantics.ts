/**
 * Semantic (cross-field) validation for v3 array metadata.
 *
 * The structural validators never interpret what an extension point names;
 * this module deliberately interprets a small set of WELL-KNOWN core
 * extension points to enforce the cross-field rules the specs state in
 * prose:
 *
 * - the `regular` chunk grid REQUIRES a configuration with `chunk_shape`,
 *   which has one length per dimension of `shape`;
 * - the `rectilinear` chunk grid requires a configuration with `kind` and
 *   `chunk_shapes`; `chunk_shapes` has one entry per dimension, and each
 *   explicit chunk list (integers and `[size, count]` run-length pairs)
 *   sums exactly to that dimension's length (the bare-integer uniform
 *   shorthand carries no sum constraint, like the regular grid);
 * - a `transpose` codec requires a configuration with `order`, a
 *   permutation of `0..n-1` with one entry per array dimension;
 * - a `sharding_indexed` codec requires a configuration with
 *   `chunk_shape`, `codecs`, and `index_codecs`; its inner `chunk_shape`
 *   matches the array's
 *   dimensionality and evenly divides every chunk it shards — the grid's
 *   chunk at the top level (each distinct per-dimension size, for
 *   rectilinear grids), the parent shard's inner chunk when sharding
 *   nests;
 * - `fill_value` has a JSON shape permitted for the named core data type
 *   (booleans for `bool`, ranged integers for the int types, numbers /
 *   the "NaN"-family sentinels / width-checked "0x…" strings for the
 *   float types, two-element arrays for the complex types).
 *
 * Dimensional context is THREADED through codec pipelines rather than
 * assumed constant (mirroring zarr-python's chunk-spec threading): a valid
 * `transpose` permutes the per-dimension chunk sizes for the codecs after
 * it, and any codec this layer cannot reason about — `reshape` may change
 * a chunk's rank, and unknown codecs may do anything — invalidates the
 * dimensional context for the rest of the pipeline instead of letting
 * stale array-level facts produce false verdicts.
 *
 * Required-configuration rules cover ONLY the names this layer interprets
 * (the two grids, transpose, sharding). Other extensions' required fields
 * are registry-schema facts — the editor's registry layer enforces them
 * from the vendored schemas, and duplicating that knowledge here as
 * hardcoded rules would drift. Unrecognized names are skipped everywhere —
 * the extension name space is open, and a rule that guessed would lie. This layer has no counterpart
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
 * (undefined when unknowable); `outerChunkSizes` holds the distinct chunk
 * lengths per dimension this pipeline may encode — what a sharding codec's
 * inner chunks must divide. A regular grid contributes one size per
 * dimension; a rectilinear grid may contribute several.
 */
function pipelineIssues(
  pipeline: unknown[],
  path: Path,
  initialDims: number | undefined,
  initialChunkSizes: number[][] | undefined,
): PathedIssue[] {
  const issues: PathedIssue[] = [];
  // The dimensional context THREADS through the pipeline: transpose permutes
  // it, and anything this layer cannot reason about invalidates it for the
  // remaining codecs.
  let dims = initialDims;
  let chunkSizes = initialChunkSizes;
  const dropContext = (): void => {
    dims = undefined;
    chunkSizes = undefined;
  };
  pipeline.forEach((entry, index) => {
    const parts = fieldParts(entry);
    if (parts === undefined) {
      dropContext(); // structurally invalid entry: no further reasoning
      return;
    }
    const { name, configuration } = parts;
    // Whether the entry genuinely lacks a configuration member (a malformed
    // one is the structural layer's complaint, not ours).
    const configMissing =
      typeof entry === "string" ||
      (isPlainObject(entry) && !Object.hasOwn(entry, "configuration"));
    if (name === "transpose") {
      if (configMissing) {
        issues.push({
          path: [...path, index],
          message: '"transpose" requires a configuration with "order"',
          kind: "missing_key",
        });
        dropContext();
        return;
      }
      if (configuration !== undefined && !Object.hasOwn(configuration, "order")) {
        issues.push({
          path: [...path, index, "configuration", "order"],
          message: "missing required key",
          kind: "missing_key",
        });
        dropContext();
        return;
      }
      const order = configuration?.["order"];
      if (!isIntArray(order)) {
        dropContext(); // shape errors are the schema layer's
        return;
      }
      const orderPath = [...path, index, "configuration", "order"];
      let sound = true;
      if (!isPermutation(order)) {
        sound = false;
        issues.push({
          path: orderPath,
          message: `expected a permutation of the integers 0..${order.length - 1}`,
          kind: "invalid_value",
        });
      }
      if (dims !== undefined && order.length !== dims) {
        sound = false;
        issues.push({
          path: orderPath,
          message: `expected one entry per array dimension (${dims})`,
          kind: "invalid_value",
        });
      }
      if (!sound) {
        dropContext();
      } else if (chunkSizes !== undefined) {
        const sizes = chunkSizes;
        chunkSizes =
          order.length === sizes.length
            ? order.map((axis) => sizes[axis] as number[])
            : undefined;
      }
    } else if (name === "sharding_indexed") {
      if (configMissing) {
        issues.push({
          path: [...path, index],
          message:
            '"sharding_indexed" requires a configuration with "chunk_shape", "codecs", and "index_codecs"',
          kind: "missing_key",
        });
        dropContext();
        return;
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
            ...pipelineIssues(
              inner,
              [...path, index, "configuration", "codecs"],
              undefined,
              undefined,
            ),
          );
        }
        dropContext();
        return;
      }
      const chunkShapePath = [...path, index, "configuration", "chunk_shape"];
      if (dims !== undefined && chunkShape.length !== dims) {
        issues.push({
          path: chunkShapePath,
          message: `expected one length per array dimension (${dims})`,
          kind: "invalid_value",
        });
      } else if (
        chunkSizes !== undefined &&
        chunkShape.length === chunkSizes.length &&
        chunkShape.every((length) => length > 0)
      ) {
        let violation: { axis: number; size: number } | undefined;
        chunkSizes.forEach((sizes, axis) => {
          if (violation !== undefined) return;
          const bad = sizes.find((size) => size % (chunkShape[axis] as number) !== 0);
          if (bad !== undefined) violation = { axis, size: bad };
        });
        if (violation !== undefined) {
          const { axis, size } = violation;
          const sizesNow = chunkSizes;
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
          ...pipelineIssues(
            inner,
            [...path, index, "configuration", "codecs"],
            chunkShape.length,
            chunkShape.map((length) => [length]),
          ),
        );
      }
      // index_codecs encode the shard index, whose shape differs from the
      // array's — no dimensional context applies there. After the
      // array -> bytes boundary the dimensional context is spent.
      dropContext();
    } else {
      // A codec this layer cannot reason about (reshape may change a
      // chunk's rank; unknown codecs may do anything): stale array-level
      // facts must not judge the codecs after it.
      dropContext();
    }
  });
  return issues;
}

function arraySemanticsIssues(value: unknown): PathedIssue[] {
  if (!isPlainObject(value) || value["node_type"] !== "array") return [];
  const issues: PathedIssue[] = [];
  const shape = isIntArray(value["shape"]) ? value["shape"] : undefined;
  const dims = shape?.length;

  const rawGrid = value["chunk_grid"];
  const grid = fieldParts(rawGrid);
  // Whether the field genuinely lacks a configuration member (a malformed
  // one — configuration: 5 — is the structural layer's complaint, not ours).
  const configMissing =
    typeof rawGrid === "string" ||
    (isPlainObject(rawGrid) && !Object.hasOwn(rawGrid, "configuration"));
  let chunkSizes: number[][] | undefined;
  if (grid?.name === "regular") {
    if (configMissing) {
      issues.push({
        path: ["chunk_grid"],
        message: '"regular" requires a configuration with "chunk_shape"',
        kind: "missing_key",
      });
    } else if (grid.configuration !== undefined && !Object.hasOwn(grid.configuration, "chunk_shape")) {
      issues.push({
        path: ["chunk_grid", "configuration", "chunk_shape"],
        message: "missing required key",
        kind: "missing_key",
      });
    }
    const configured = grid.configuration?.["chunk_shape"];
    if (isIntArray(configured)) {
      if (dims !== undefined && configured.length !== dims) {
        issues.push({
          path: ["chunk_grid", "configuration", "chunk_shape"],
          message: `expected one length per dimension of shape (${dims})`,
          kind: "invalid_value",
        });
        // wrong arity: unusable as division context
      } else {
        chunkSizes = configured.map((length) => [length]);
      }
    }
  } else if (grid?.name === "rectilinear") {
    if (configMissing) {
      issues.push({
        path: ["chunk_grid"],
        message: '"rectilinear" requires a configuration with "kind" and "chunk_shapes"',
        kind: "missing_key",
      });
    } else if (grid.configuration !== undefined) {
      for (const key of ["kind", "chunk_shapes"]) {
        if (!Object.hasOwn(grid.configuration, key)) {
          issues.push({
            path: ["chunk_grid", "configuration", key],
            message: "missing required key",
            kind: "missing_key",
          });
        }
      }
    }
    const specs = grid.configuration?.["chunk_shapes"];
    if (Array.isArray(specs)) {
      if (dims !== undefined && specs.length !== dims) {
        issues.push({
          path: ["chunk_grid", "configuration", "chunk_shapes"],
          message: `expected one entry per dimension of shape (${dims})`,
          kind: "invalid_value",
        });
      } else {
        const perDim: (number[] | undefined)[] = specs.map((spec, dim) => {
          // Bare integer: uniform shorthand, no sum constraint (edge chunks
          // are permitted, as with the regular grid).
          if (Number.isInteger(spec)) return spec as number > 0 ? [spec as number] : undefined;
          if (!Array.isArray(spec)) return undefined; // registry schema's problem
          const sizes = new Set<number>();
          let total = 0;
          for (const entry of spec) {
            if (Number.isInteger(entry)) {
              sizes.add(entry as number);
              total += entry as number;
            } else if (
              Array.isArray(entry) &&
              entry.length === 2 &&
              Number.isInteger(entry[0]) &&
              Number.isInteger(entry[1])
            ) {
              sizes.add(entry[0] as number);
              total += (entry[0] as number) * (entry[1] as number);
            } else {
              return undefined; // malformed entry: registry schema's problem
            }
          }
          const extent = shape?.[dim];
          if (extent !== undefined && total !== extent) {
            issues.push({
              path: ["chunk_grid", "configuration", "chunk_shapes", dim],
              message: `expected chunk sizes summing to ${extent} along dimension ${dim}, got ${total}`,
              kind: "invalid_value",
            });
          }
          return [...sizes];
        });
        if (perDim.every((sizes) => sizes !== undefined)) {
          chunkSizes = perDim as number[][];
        }
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
    issues.push(...pipelineIssues(codecs, ["codecs"], dims, chunkSizes));
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
