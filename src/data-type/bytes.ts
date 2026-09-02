import { isByteArray, named, simple, type DataTypeDescriptor } from "./descriptor.js";

/**
 * Fill value of the `bytes` data type: an array of integers in `[0, 255]`
 * (one per byte) or a standard-alphabet base64 string.
 */
export type BytesFillValue = number[] | string;

const BASE64 = /^[A-Za-z0-9+/]*={0,2}$/;

/** Whether `value` is standard-alphabet base64 (padded length a multiple of 4). */
export function isBase64(value: string): boolean {
  return value.length % 4 === 0 && BASE64.test(value);
}

/** The zarr-extensions `bytes` data type (variable-length raw bytes). */
export const bytes: DataTypeDescriptor = {
  matches: named("bytes"),
  fillIssues: (fill) => {
    if (isByteArray(fill)) return [];
    if (typeof fill === "string" && isBase64(fill)) return [];
    return simple(
      'expected an array of integers in [0, 255] or a base64 string for data type "bytes"',
    );
  },
};
