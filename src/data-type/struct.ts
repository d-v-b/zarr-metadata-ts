import { isPlainObject } from "../guards.js";
import { issue, named, simple, type DataTypeDescriptor } from "./descriptor.js";

/** One field of a `struct` data type; `data_type` recurses (structs may nest). */
export interface StructField {
  name: string;
  data_type: unknown;
}

/** Configuration of the `struct` data type. */
export interface StructConfiguration {
  fields: StructField[];
}

/**
 * The zarr-extensions `struct` data type (heterogeneous record). Its fill
 * value is a JSON object mapping every field name to that field's fill
 * value, each judged recursively against the field's own data type.
 */
export const struct: DataTypeDescriptor = {
  matches: named("struct"),
  requiredConfigKeys: ["fields"],
  fillIssues: (fill, _name, context) => {
    if (!isPlainObject(fill)) {
      return simple('expected an object mapping field names to fill values for data type "struct"');
    }
    const fields = context.configuration?.["fields"];
    if (!Array.isArray(fields)) return []; // config shape is the schema layer's problem
    const declared = new Map<string, unknown>();
    for (const field of fields) {
      if (isPlainObject(field) && typeof field["name"] === "string") {
        declared.set(field["name"], field["data_type"]);
      }
    }
    if (declared.size !== fields.length) return []; // malformed fields: schema layer's problem
    const issues = [];
    for (const [fieldName, dataType] of declared) {
      if (!Object.hasOwn(fill, fieldName)) {
        issues.push(issue([fieldName], "missing required key", "missing_key"));
        continue;
      }
      issues.push(
        ...context
          .fillIssuesFor(dataType, fill[fieldName], context.depth + 1)
          .map((inner) => ({ ...inner, path: [fieldName, ...inner.path] })),
      );
    }
    for (const key of Object.keys(fill)) {
      if (!declared.has(key)) {
        issues.push(issue([key], "unexpected member (no such struct field)"));
      }
    }
    return issues;
  },
};
