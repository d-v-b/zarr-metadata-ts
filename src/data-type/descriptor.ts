/**
 * The data type descriptor interface and the shared fill-value machinery
 * the per-type modules build on. One module per supported data type;
 * `index.ts` assembles them into the dispatch the semantic layer uses.
 */
import type { IssueKind, PathedIssue } from "../errors.js";

/** Context handed to a descriptor's fill check. */
export interface FillContext {
  /** The data_type field's configuration, when present and an object. */
  configuration: Record<string, unknown> | undefined;
  /** Recursion depth (struct fields may nest structs). */
  depth: number;
  /**
   * Recursive check for compound types: issues for `fill` judged against
   * the data type named by `dataTypeField`, pathed relative to that fill.
   */
  fillIssuesFor(dataTypeField: unknown, fill: unknown, depth: number): PathedIssue[];
}

/** Everything the semantic layer knows about one data type. */
export interface DataTypeDescriptor {
  /** Whether this module owns the given `data_type` name. */
  matches(name: string): boolean;
  /** Required configuration members, when the type needs a configuration. */
  requiredConfigKeys?: readonly string[];
  /** Name-level validity beyond ownership (e.g. r<N>'s multiple-of-8 rule). */
  nameIssue?(name: string): string | undefined;
  /** Fill-value issues, pathed relative to the fill value itself. */
  fillIssues(fill: unknown, name: string, context: FillContext): PathedIssue[];
}

export function issue(
  path: ReadonlyArray<string | number>,
  message: string,
  kind: IssueKind = "invalid_value",
): PathedIssue {
  return { path, message, kind };
}

/** A single unpathed issue — the common case for scalar types. */
export function simple(message: string): PathedIssue[] {
  return [issue([], message)];
}

export function named(matchName: string): (name: string) => boolean {
  return (name) => name === matchName;
}

// --- shared fill forms ------------------------------------------------------

/** A float fill value: a JSON number, a non-finite sentinel, or a hex string. */
export type FloatFillValue = number | "NaN" | "Infinity" | "-Infinity" | string;

/** A complex fill value: `[real, imaginary]`, each a float fill form. */
export type ComplexFillValue = [FloatFillValue, FloatFillValue];

export function isFloatFill(value: unknown, hexDigits: number): boolean {
  if (typeof value === "number") return true;
  if (typeof value !== "string") return false;
  if (value === "NaN" || value === "Infinity" || value === "-Infinity") return true;
  return new RegExp(`^0x[0-9a-fA-F]{${hexDigits}}$`).test(value);
}

export function floatFillMessage(name: string, hexDigits: number): string {
  return (
    `expected a number, "NaN", "Infinity", "-Infinity", or a ` +
    `${hexDigits}-hex-digit "0x..." string for data type ${JSON.stringify(name)}`
  );
}

export function isByteArray(value: unknown): value is number[] {
  return (
    Array.isArray(value) &&
    Object.keys(value).length === value.length &&
    value.every((item) => Number.isInteger(item) && (item as number) >= 0 && (item as number) <= 255)
  );
}

// --- per-family factories ---------------------------------------------------

export function intDataType(name: string, low: number, high: number): DataTypeDescriptor {
  return {
    matches: named(name),
    fillIssues: (fill) =>
      Number.isInteger(fill) && (fill as number) >= low && (fill as number) <= high
        ? []
        : simple(`expected an integer in [${low}, ${high}] for data type ${JSON.stringify(name)}`),
  };
}

export function floatDataType(name: string, hexDigits: number): DataTypeDescriptor {
  return {
    matches: named(name),
    fillIssues: (fill) =>
      isFloatFill(fill, hexDigits) ? [] : simple(floatFillMessage(name, hexDigits)),
  };
}

export function complexDataType(name: string, componentHexDigits: number): DataTypeDescriptor {
  return {
    matches: named(name),
    fillIssues: (fill) => {
      const ok =
        Array.isArray(fill) &&
        fill.length === 2 &&
        fill.every((part) => isFloatFill(part, componentHexDigits));
      return ok
        ? []
        : simple(
            `expected a two-element [real, imaginary] array for data type ${JSON.stringify(name)}`,
          );
    },
  };
}
