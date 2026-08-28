/**
 * Runs the shared conformance corpus (../conformance) against the
 * TypeScript validators. The same corpus runs against the Python reference
 * implementation via scripts/check_conformance.py.
 */
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import {
  validateArrayMetadataV2,
  validateArrayMetadataV3,
  validateConsolidatedMetadataV2,
  validateGroupMetadataV2,
  validateGroupMetadataV3,
  type ValidationProblem,
} from "../src/index.js";

const CORPUS_DIR = fileURLToPath(new URL("../conformance/", import.meta.url));

const VALIDATORS: Record<string, (value: unknown) => ValidationProblem[]> = {
  v3_array: validateArrayMetadataV3,
  v3_group: validateGroupMetadataV3,
  v2_array: validateArrayMetadataV2,
  v2_group: validateGroupMetadataV2,
  v2_consolidated: validateConsolidatedMetadataV2,
};

interface ConformanceCase {
  description: string;
  document: unknown;
  problems: Array<{ loc: Array<string | number>; kind: string }>;
}

/** Order-insensitive canonical form of a problem set: sorted (loc, kind) pairs. */
function canonical(problems: Array<{ loc: ReadonlyArray<string | number>; kind: string }>): string[] {
  return problems.map((p) => JSON.stringify([p.loc, p.kind])).sort();
}

const kinds = readdirSync(CORPUS_DIR)
  .filter((name) => name.endsWith(".json"))
  .map((name) => name.replace(/\.json$/, ""));

it("corpus covers every registered document kind", () => {
  expect(kinds.sort()).toEqual(Object.keys(VALIDATORS).sort());
});

for (const kind of kinds) {
  const validator = VALIDATORS[kind];
  describe(kind, () => {
    if (validator === undefined) {
      it("has a registered validator", () => {
        throw new Error(`no validator registered for corpus kind ${kind}`);
      });
      return;
    }
    const cases: ConformanceCase[] = JSON.parse(
      readFileSync(join(CORPUS_DIR, `${kind}.json`), "utf-8"),
    );
    for (const testCase of cases) {
      it(testCase.description, () => {
        const actual = validator(testCase.document);
        expect(canonical(actual)).toEqual(canonical(testCase.problems));
      });
    }
  });
}
