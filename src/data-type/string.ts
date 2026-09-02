import { named, simple, type DataTypeDescriptor } from "./descriptor.js";

/** Fill value of the `string` data type: a JSON unicode string. */
export type StringFillValue = string;

/** The zarr-extensions `string` data type (variable-length UTF-8). */
export const string: DataTypeDescriptor = {
  matches: named("string"),
  fillIssues: (fill) =>
    typeof fill === "string" ? [] : simple('expected a string fill value for data type "string"'),
};
