/** The core `transpose` codec: syntax and semantics. */
import type { PathedIssue } from "../errors.js";
import { isIntArray, type Path } from "../guards.js";
import { DROPPED, type PipelineContext } from "./index.js";

/** Configuration of the core `transpose` codec. */
export interface TransposeCodecConfiguration {
  order: number[];
}

function isPermutation(order: number[]): boolean {
  const seen = new Set(order);
  return seen.size === order.length && order.every((entry) => entry >= 0 && entry < order.length);
}

/**
 * Judge one transpose entry: it requires a configuration with `order`,
 * which must be a permutation with one entry per array dimension. A sound
 * transpose permutes the context's chunk sizes; an unsound one makes the
 * downstream context unknowable.
 */
export function transposeIssues(
  configuration: Record<string, unknown> | undefined,
  configMissing: boolean,
  path: Path,
  index: number,
  context: PipelineContext,
): { issues: PathedIssue[]; context: PipelineContext } {
  const issues: PathedIssue[] = [];
  if (configMissing) {
    issues.push({
      path: [...path, index],
      message: '"transpose" requires a configuration with "order"',
      kind: "missing_key",
    });
    return { issues, context: DROPPED };
  }
  if (configuration !== undefined && !Object.hasOwn(configuration, "order")) {
    issues.push({
      path: [...path, index, "configuration", "order"],
      message: "missing required key",
      kind: "missing_key",
    });
    return { issues, context: DROPPED };
  }
  const order = configuration?.["order"];
  if (!isIntArray(order)) {
    return { issues, context: DROPPED }; // shape errors are the schema layer's
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
  if (context.dims !== undefined && order.length !== context.dims) {
    sound = false;
    issues.push({
      path: orderPath,
      message: `expected one entry per array dimension (${context.dims})`,
      kind: "invalid_value",
    });
  }
  if (!sound) return { issues, context: DROPPED };
  const sizes = context.chunkSizes;
  return {
    issues,
    context: {
      dims: context.dims,
      chunkSizes:
        sizes !== undefined && order.length === sizes.length
          ? order.map((axis) => sizes[axis] as number[])
          : undefined,
    },
  };
}
