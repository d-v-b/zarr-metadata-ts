import { named, simple, type DataTypeDescriptor } from "./descriptor.js";

/** The core `bool` data type. */
export const bool: DataTypeDescriptor = {
  matches: named("bool"),
  fillIssues: (fill) =>
    typeof fill === "boolean" ? [] : simple('expected a boolean fill value for data type "bool"'),
};
