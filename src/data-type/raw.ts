import { isByteArray, simple, type DataTypeDescriptor } from "./descriptor.js";

/** Fill value of the `r<N>` raw-bits data types: one integer per byte. */
export type RawBytesFillValue = number[];

const RAW = /^r([1-9][0-9]*)$/;

/**
 * The core `r<N>` raw-bits data type family. The name itself carries a
 * rule — N must be a positive multiple of 8 — and the fill value must
 * hold exactly N/8 bytes.
 */
export const raw: DataTypeDescriptor = {
  matches: (name) => RAW.test(name),
  nameIssue: (name) => {
    const bits = Number((RAW.exec(name) as RegExpExecArray)[1]);
    return bits % 8 === 0
      ? undefined
      : `expected "r<N>" with N a positive multiple of 8, got ${JSON.stringify(name)}`;
  },
  fillIssues: (fill, name) => {
    const bits = Number((RAW.exec(name) as RegExpExecArray)[1]);
    if (bits % 8 !== 0) return []; // the name issue already covers this document
    const expectedBytes = bits / 8;
    return isByteArray(fill) && fill.length === expectedBytes
      ? []
      : simple(
          `expected an array of ${expectedBytes} integers in [0, 255] for data type ${JSON.stringify(name)}`,
        );
  },
};
