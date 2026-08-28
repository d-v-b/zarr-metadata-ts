/**
 * The problem/error vocabulary shared by every validator.
 *
 * Mirrors `zarr_metadata.model._validation` in the Python reference
 * implementation: every validator returns EVERY problem found (not just the
 * first), each problem carries a machine-readable `kind` and a `loc` path
 * from the document root to the offending value.
 */

/**
 * Machine-readable classification of a `ValidationProblem`.
 *
 * - `missing_key`: a required key (document key or store key) is absent.
 * - `invalid_type`: a value has the wrong structural type (e.g. a string
 *   where a mapping is required).
 * - `invalid_value`: a value has an acceptable type but invalid content
 *   (e.g. `zarr_format: 2` in a v3 document, `order: "Q"`).
 * - `invalid_json`: bytes/text that do not decode as JSON.
 */
export type ProblemKind = "missing_key" | "invalid_type" | "invalid_value" | "invalid_json";

/**
 * The path from the document root to an offending value, e.g.
 * `["codecs", 0, "name"]`. An empty loc refers to the document as a whole.
 */
export type Loc = ReadonlyArray<string | number>;

/** A single structural problem found while validating a metadata document. */
export interface ValidationProblem {
  readonly loc: Loc;
  readonly message: string;
  readonly kind: ProblemKind;
}

/** Render a problem as `"path.to.value: message"` (`"<root>: ..."` for an empty loc). */
export function formatProblem(problem: ValidationProblem): string {
  const location = problem.loc.length > 0 ? problem.loc.join(".") : "<root>";
  return `${location}: ${problem.message}`;
}

/**
 * Raised when a value fails structural metadata validation.
 *
 * Carries every problem found (not just the first) in `.problems`.
 */
export class MetadataValidationError extends Error {
  readonly problems: ReadonlyArray<ValidationProblem>;

  constructor(problems: ReadonlyArray<ValidationProblem>) {
    super(problems.map(formatProblem).join("\n"));
    this.name = "MetadataValidationError";
    this.problems = problems;
  }
}

/** Prepend `head` to the `loc` of every problem (for nested validators). */
export function prefix(
  head: string | number,
  problems: ValidationProblem[],
): ValidationProblem[] {
  return problems.map((p) => ({ loc: [head, ...p.loc], message: p.message, kind: p.kind }));
}

export function problem(loc: Loc, message: string, kind: ProblemKind): ValidationProblem {
  return { loc, message, kind };
}
