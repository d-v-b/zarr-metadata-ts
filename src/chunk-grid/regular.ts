/** The core `regular` chunk grid: syntax and semantics. */
import type { PathedIssue } from "../errors.js";
import { configurationMissing, fieldParts, isIntArray } from "../guards.js";
import type { ChunkGridVerdict } from "./index.js";

/** Configuration of the core `regular` chunk grid. */
export interface RegularChunkGridConfiguration {
  chunk_shape: number[];
}

export function regularIssues(rawGrid: unknown, shape: number[] | undefined): ChunkGridVerdict {
  const issues: PathedIssue[] = [];
  let chunkSizes: number[][] | undefined;
  const configuration = fieldParts(rawGrid)?.configuration;
  if (configurationMissing(rawGrid)) {
    issues.push({
      path: [],
      message: '"regular" requires a configuration with "chunk_shape"',
      kind: "missing_key",
    });
  } else if (configuration !== undefined && !Object.hasOwn(configuration, "chunk_shape")) {
    issues.push({
      path: ["configuration", "chunk_shape"],
      message: "missing required key",
      kind: "missing_key",
    });
  }
  const configured = configuration?.["chunk_shape"];
  if (isIntArray(configured)) {
    if (shape !== undefined && configured.length !== shape.length) {
      issues.push({
        path: ["configuration", "chunk_shape"],
        message: `expected one length per dimension of shape (${shape.length})`,
        kind: "invalid_value",
      });
      // wrong arity: unusable as division context
    } else {
      chunkSizes = configured.map((length) => [length]);
    }
  }
  return { issues, chunkSizes };
}
