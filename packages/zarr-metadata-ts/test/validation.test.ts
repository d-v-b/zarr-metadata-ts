/**
 * TS-specific unit tests: behavior the conformance corpus cannot express
 * because its fixtures must be JSON text (non-JSON runtime values), plus the
 * parse/guard/error API surface.
 */
import { describe, expect, it } from "vitest";

import {
  formatProblem,
  isArrayMetadataV3,
  MetadataValidationError,
  parseArrayMetadataV3,
  parseJson,
  validateJson,
  validateMetadataV3,
} from "../src/index.js";

const VALID_ARRAY = {
  zarr_format: 3,
  node_type: "array",
  shape: [10],
  data_type: "float64",
  chunk_grid: { name: "regular", configuration: { chunk_shape: [5] } },
  chunk_key_encoding: "default",
  fill_value: 0,
  codecs: ["bytes"],
};

describe("validateJson", () => {
  it("rejects non-finite numbers", () => {
    const problems = validateJson({ a: [1, Infinity] });
    expect(problems).toEqual([
      { loc: ["a", 1], message: "non-finite number Infinity is not JSON", kind: "invalid_value" },
    ]);
  });

  it("rejects non-JSON runtime values with their path", () => {
    const problems = validateJson({ a: { b: undefined } });
    expect(problems).toHaveLength(1);
    expect(problems[0]!.loc).toEqual(["a", "b"]);
    expect(problems[0]!.kind).toBe("invalid_type");
  });
});

describe("parse functions", () => {
  it("throw MetadataValidationError carrying every problem", () => {
    expect.assertions(3);
    try {
      parseArrayMetadataV3({ zarr_format: 3, node_type: "array" });
    } catch (error) {
      expect(error).toBeInstanceOf(MetadataValidationError);
      const problems = (error as MetadataValidationError).problems;
      expect(problems).toHaveLength(6);
      expect(problems.every((p) => p.kind === "missing_key")).toBe(true);
    }
  });

  it("return the input narrowed on success", () => {
    expect(parseArrayMetadataV3(VALID_ARRAY)).toBe(VALID_ARRAY);
    expect(parseJson([1, "a", null])).toEqual([1, "a", null]);
  });
});

describe("type guards", () => {
  it("accept valid and reject invalid documents", () => {
    expect(isArrayMetadataV3(VALID_ARRAY)).toBe(true);
    expect(isArrayMetadataV3({ ...VALID_ARRAY, codecs: [] })).toBe(false);
    expect(isArrayMetadataV3("not a doc")).toBe(false);
  });
});

describe("validateMetadataV3 dispatcher", () => {
  it("routes arrays and groups to the right validator", () => {
    expect(validateMetadataV3(VALID_ARRAY)).toEqual([]);
    expect(validateMetadataV3({ zarr_format: 3, node_type: "group" })).toEqual([]);
  });

  it("reports a missing or unusable node_type", () => {
    expect(validateMetadataV3({ zarr_format: 3 })).toEqual([
      { loc: ["node_type"], message: "missing required key", kind: "missing_key" },
    ]);
    const problems = validateMetadataV3({ zarr_format: 3, node_type: "dataset" });
    expect(problems).toHaveLength(1);
    expect(problems[0]!.kind).toBe("invalid_value");
  });

  it("rejects non-mapping documents", () => {
    expect(validateMetadataV3([])[0]!.kind).toBe("invalid_type");
  });
});

describe("formatProblem", () => {
  it("joins the loc with dots and uses <root> for an empty loc", () => {
    expect(formatProblem({ loc: ["codecs", 0, "name"], message: "m", kind: "invalid_type" })).toBe(
      "codecs.0.name: m",
    );
    expect(formatProblem({ loc: [], message: "m", kind: "invalid_type" })).toBe("<root>: m");
  });
});
