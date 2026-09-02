/**
 * Shared value guards for the content modules (chunk grids, codecs, data
 * types) and the semantics orchestrator. The structural document layer
 * (validation.ts) keeps its own copies deliberately: it is the
 * corpus-governed layer and stays self-contained.
 */

export type Path = ReadonlyArray<string | number>;

export function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const proto: unknown = Object.getPrototypeOf(value);
  return proto === null || proto === Object.prototype;
}

export function isIntArray(value: unknown): value is number[] {
  return (
    Array.isArray(value) &&
    Object.keys(value).length === value.length &&
    value.every((item) => Number.isInteger(item))
  );
}

/** The `{name, configuration}` of a metadata field, when extractable. */
export function fieldParts(
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

/**
 * Whether a metadata field genuinely lacks a `configuration` member (a
 * malformed one — `configuration: 5` — is the structural layer's
 * complaint, not the semantic layer's).
 */
export function configurationMissing(field: unknown): boolean {
  return (
    typeof field === "string" || (isPlainObject(field) && !Object.hasOwn(field, "configuration"))
  );
}
