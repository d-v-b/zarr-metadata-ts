/**
 * Semantic-layer tests. This layer has no Python counterpart, so it is
 * covered here rather than by the conformance corpus.
 */
import { describe, expect, it } from "vitest";

import {
  flattenTree,
  isEmptyTree,
  validateArraySemanticsV3,
  validateSemanticsV3,
} from "../src/index.js";

function array(overrides: Record<string, unknown>): Record<string, unknown> {
  return {
    zarr_format: 3,
    node_type: "array",
    shape: [12, 12],
    data_type: "float64",
    chunk_grid: { name: "regular", configuration: { chunk_shape: [6, 6] } },
    chunk_key_encoding: "default",
    fill_value: 0,
    codecs: ["bytes"],
    ...overrides,
  };
}

function messages(value: unknown): string[] {
  return flattenTree(validateArraySemanticsV3(value)).map((issue) => issue.message);
}

describe("validateArraySemanticsV3", () => {
  it("accepts semantically consistent documents of every checked flavor", () => {
    const valid = [
      array({}),
      array({ data_type: "bool", fill_value: true }),
      array({ data_type: "uint8", fill_value: 255 }),
      array({ data_type: "float32", fill_value: "NaN" }),
      array({ data_type: "float64", fill_value: "0x7ff8000000000000" }),
      array({ data_type: "complex64", fill_value: [1.5, "Infinity"] }),
      array({ data_type: "my.custom.dtype", fill_value: { anything: true } }),
      array({
        codecs: [
          { name: "transpose", configuration: { order: [1, 0] } },
          {
            name: "sharding_indexed",
            configuration: {
              chunk_shape: [3, 3],
              codecs: [
                { name: "transpose", configuration: { order: [0, 1] } },
                "bytes",
              ],
              index_codecs: ["bytes", "crc32c"],
            },
          },
        ],
      }),
      // scalar arrays: everything is zero-dimensional and empty
      array({
        shape: [],
        chunk_grid: { name: "regular", configuration: { chunk_shape: [] } },
      }),
      // rectilinear: bare-int shorthand (no sum rule), explicit lists and
      // RLE pairs summing exactly, and a sharding codec dividing every
      // distinct chunk size.
      array({
        chunk_grid: {
          name: "rectilinear",
          configuration: { kind: "inline", chunk_shapes: [6, [4, [2, 4]]] },
        },
        codecs: [
          {
            name: "sharding_indexed",
            configuration: { chunk_shape: [2, 2], codecs: ["bytes"], index_codecs: ["bytes"] },
          },
        ],
      }),
      // context threading: reshape may change the chunk's rank, so the
      // rank-4 transpose after it must not be judged by the array's rank
      // (the reshape spec explicitly endorses this combination)
      array({
        shape: [6, 6, 6],
        chunk_grid: { name: "regular", configuration: { chunk_shape: [6, 6, 6] } },
        codecs: [
          { name: "reshape", configuration: { shape: [[0], [1], 2, 3] } },
          { name: "transpose", configuration: { order: [3, 2, 1, 0] } },
          "bytes",
        ],
      }),
      // context threading: transpose permutes the chunk sizes the shard
      // must divide
      array({
        shape: [8, 12],
        chunk_grid: { name: "regular", configuration: { chunk_shape: [4, 6] } },
        codecs: [
          { name: "transpose", configuration: { order: [1, 0] } },
          {
            name: "sharding_indexed",
            configuration: { chunk_shape: [3, 4], codecs: ["bytes"], index_codecs: ["bytes"] },
          },
        ],
      }),
      // non-array documents and structural wrecks produce no semantic verdicts
      { zarr_format: 3, node_type: "group" },
      "not a document",
    ];
    for (const document of valid) {
      expect(isEmptyTree(validateArraySemanticsV3(document)), JSON.stringify(document)).toBe(true);
    }
  });

  it("rejects a chunk_shape arity mismatch with shape", () => {
    expect(
      messages(array({ chunk_grid: { name: "regular", configuration: { chunk_shape: [6] } } })),
    ).toEqual(["expected one length per dimension of shape (2)"]);
  });

  it("rejects a transpose order that is not a permutation", () => {
    expect(
      messages(array({ codecs: [{ name: "transpose", configuration: { order: [0, 2] } }, "bytes"] })),
    ).toEqual(["expected a permutation of the integers 0..1"]);
  });

  it("rejects a transpose order with the wrong dimensionality", () => {
    expect(
      messages(
        array({ codecs: [{ name: "transpose", configuration: { order: [0, 1, 2] } }, "bytes"] }),
      ),
    ).toEqual(["expected one entry per array dimension (2)"]);
  });

  it("rejects a sharding chunk_shape with the wrong dimensionality", () => {
    expect(
      messages(
        array({
          codecs: [
            {
              name: "sharding_indexed",
              configuration: { chunk_shape: [3], codecs: ["bytes"], index_codecs: ["bytes"] },
            },
          ],
        }),
      ),
    ).toEqual(["expected one length per array dimension (2)"]);
  });

  it("rejects a sharding chunk_shape that does not divide the outer chunk shape", () => {
    expect(
      messages(
        array({
          codecs: [
            {
              name: "sharding_indexed",
              configuration: { chunk_shape: [4, 3], codecs: ["bytes"], index_codecs: ["bytes"] },
            },
          ],
        }),
      ),
    ).toEqual(["expected [4,3] to evenly divide the outer chunk shape [6,6]"]);
  });

  it("rejects nested inner chunks that do not divide their parent shard's chunks", () => {
    expect(
      messages(
        array({
          codecs: [
            {
              name: "sharding_indexed",
              configuration: {
                chunk_shape: [6, 6],
                codecs: [
                  {
                    name: "sharding_indexed",
                    configuration: { chunk_shape: [4, 6], codecs: ["bytes"], index_codecs: ["bytes"] },
                  },
                ],
                index_codecs: ["bytes"],
              },
            },
          ],
        }),
      ),
    ).toEqual(["expected [4,6] to evenly divide the outer chunk shape [6,6]"]);
  });

  it("rejects a rectilinear chunk_shapes arity mismatch with shape", () => {
    expect(
      messages(
        array({
          chunk_grid: { name: "rectilinear", configuration: { kind: "inline", chunk_shapes: [12] } },
        }),
      ),
    ).toEqual(["expected one entry per dimension of shape (2)"]);
  });

  it("rejects explicit rectilinear chunk lists that do not sum to the dimension length", () => {
    expect(
      messages(
        array({
          chunk_grid: {
            name: "rectilinear",
            configuration: { kind: "inline", chunk_shapes: [[4, [3, 2]], [5, 5, 5]] },
          },
        }),
      ),
    ).toEqual([
      "expected chunk sizes summing to 12 along dimension 0, got 10",
      "expected chunk sizes summing to 12 along dimension 1, got 15",
    ]);
  });

  it("rejects a sharding chunk_shape that does not divide every rectilinear chunk size", () => {
    expect(
      messages(
        array({
          chunk_grid: {
            name: "rectilinear",
            configuration: { kind: "inline", chunk_shapes: [[4, [2, 4]], 6] },
          },
          codecs: [
            {
              name: "sharding_indexed",
              configuration: { chunk_shape: [4, 3], codecs: ["bytes"], index_codecs: ["bytes"] },
            },
          ],
        }),
      ),
    ).toEqual([
      "expected [4,3] to evenly divide every chunk size of the grid (dimension 0 has chunk size 2)",
    ]);
  });

  it("rejects a shard sized for the un-permuted chunk after a transpose", () => {
    expect(
      messages(
        array({
          shape: [8, 12],
          chunk_grid: { name: "regular", configuration: { chunk_shape: [4, 6] } },
          codecs: [
            { name: "transpose", configuration: { order: [1, 0] } },
            {
              name: "sharding_indexed",
              configuration: { chunk_shape: [4, 6], codecs: ["bytes"], index_codecs: ["bytes"] },
            },
          ],
        }),
      ),
    ).toEqual(["expected [4,6] to evenly divide the outer chunk shape [6,4]"]);
  });

  it("makes no dimensional claims after an invalid transpose", () => {
    // The bad order is reported, but the sharding checks downstream of it
    // are suppressed: the chunk's true shape is unknowable from here.
    expect(
      messages(
        array({
          codecs: [
            { name: "transpose", configuration: { order: [0, 0] } },
            {
              name: "sharding_indexed",
              configuration: { chunk_shape: [5, 5], codecs: ["bytes"], index_codecs: ["bytes"] },
            },
          ],
        }),
      ),
    ).toEqual(["expected a permutation of the integers 0..1"]);
  });

  it("descends into inline consolidated entries, nested groups included", () => {
    const doc = {
      zarr_format: 3,
      node_type: "group",
      consolidated_metadata: {
        kind: "inline",
        must_understand: false,
        metadata: {
          bad_array: array({ data_type: "int32", fill_value: "NaN" }),
          nested_group: {
            zarr_format: 3,
            node_type: "group",
            consolidated_metadata: {
              kind: "inline",
              must_understand: false,
              metadata: { deeper: array({ data_type: "bool", fill_value: 0 }) },
            },
          },
        },
      },
    };
    // validateArraySemanticsV3 stays array-only...
    expect(isEmptyTree(validateArraySemanticsV3(doc))).toBe(true);
    // ...while validateSemanticsV3 paths each entry's issues through the envelope.
    expect(
      flattenTree(validateSemanticsV3(doc)).map((issue) => [issue.path.join("."), issue.message]),
    ).toEqual([
      [
        "consolidated_metadata.metadata.bad_array.fill_value",
        'expected an integer in [-2147483648, 2147483647] for data type "int32"',
      ],
      [
        "consolidated_metadata.metadata.nested_group.consolidated_metadata.metadata.deeper.fill_value",
        'expected a boolean fill value for data type "bool"',
      ],
    ]);
    // A plain array document goes through the same entry point unchanged.
    expect(isEmptyTree(validateSemanticsV3(array({})))).toBe(true);
  });

  it("rejects fill values that do not fit the core data type", () => {
    const cases: Array<[Record<string, unknown>, string]> = [
      [{ data_type: "bool", fill_value: 0 }, 'expected a boolean fill value for data type "bool"'],
      [
        { data_type: "uint8", fill_value: 300 },
        'expected an integer in [0, 255] for data type "uint8"',
      ],
      [
        { data_type: "int32", fill_value: 1.5 },
        'expected an integer in [-2147483648, 2147483647] for data type "int32"',
      ],
      [
        { data_type: "float32", fill_value: "0x7ff8000000000000" },
        'expected a number, "NaN", "Infinity", "-Infinity", or a 8-hex-digit "0x..." string for data type "float32"',
      ],
      [
        { data_type: "float64", fill_value: [] },
        'expected a number, "NaN", "Infinity", "-Infinity", or a 16-hex-digit "0x..." string for data type "float64"',
      ],
      [
        { data_type: "complex128", fill_value: [1] },
        'expected a two-element [real, imaginary] array for data type "complex128"',
      ],
    ];
    for (const [overrides, message] of cases) {
      expect(messages(array(overrides)), JSON.stringify(overrides)).toEqual([message]);
    }
  });
});

describe("chunk grid configuration requirements", () => {
  it("requires a configuration for the regular grid", () => {
    expect(messages(array({ chunk_grid: "regular" }))).toEqual([
      '"regular" requires a configuration with "chunk_shape"',
    ]);
    expect(messages(array({ chunk_grid: { name: "regular" } }))).toEqual([
      '"regular" requires a configuration with "chunk_shape"',
    ]);
    expect(messages(array({ chunk_grid: { name: "regular", configuration: {} } }))).toEqual([
      "missing required key",
    ]);
  });

  it("requires kind and chunk_shapes for the rectilinear grid", () => {
    expect(messages(array({ chunk_grid: "rectilinear" }))).toEqual([
      '"rectilinear" requires a configuration with "kind" and "chunk_shapes"',
    ]);
    expect(
      messages(array({ chunk_grid: { name: "rectilinear", configuration: { kind: "inline" } } })),
    ).toEqual(["missing required key"]);
    expect(
      messages(
        array({ chunk_grid: { name: "rectilinear", configuration: { chunk_shapes: [12, 12] } } }),
      ),
    ).toEqual(["missing required key"]);
  });

  it("leaves a malformed configuration to the structural layer", () => {
    // configuration: 5 is present-but-wrong; the structural metadata-field
    // validator owns that complaint.
    expect(messages(array({ chunk_grid: { name: "regular", configuration: 5 } }))).toEqual([]);
  });
});

describe("codec configuration requirements", () => {
  it("requires order for transpose and the three sharding members", () => {
    expect(messages(array({ codecs: ["transpose", "bytes"] }))).toEqual([
      '"transpose" requires a configuration with "order"',
    ]);
    expect(messages(array({ codecs: [{ name: "transpose", configuration: {} }, "bytes"] }))).toEqual(
      ["missing required key"],
    );
    expect(messages(array({ codecs: ["sharding_indexed"] }))).toEqual([
      '"sharding_indexed" requires a configuration with "chunk_shape", "codecs", and "index_codecs"',
    ]);
    expect(
      messages(
        array({
          codecs: [{ name: "sharding_indexed", configuration: { chunk_shape: [6, 6] } }],
        }),
      ).sort(),
    ).toEqual(["missing required key", "missing required key"]);
  });

  it("still judges a present inner pipeline when sharding's chunk_shape is missing", () => {
    expect(
      messages(
        array({
          codecs: [
            {
              name: "sharding_indexed",
              configuration: {
                codecs: [{ name: "transpose", configuration: { order: [0, 0] } }, "bytes"],
                index_codecs: ["bytes"],
              },
            },
          ],
        }),
      ).sort(),
    ).toEqual(["expected a permutation of the integers 0..1", "missing required key"]);
  });

  it("leaves the uninterpreted codecs' required fields to the registry schemas", () => {
    // gzip/blosc/zstd requirements are registry-schema facts, not semantic
    // rules — a second hardcoded source of truth would drift.
    expect(messages(array({ codecs: ["bytes", "gzip"] }))).toEqual([]);
  });
});
