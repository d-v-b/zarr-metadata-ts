import { numpyTemporalDataType } from "./numpy-datetime64.js";

export type {
  NumpyDatetime64Configuration as NumpyTimedelta64Configuration,
  NumpyDatetime64FillValue as NumpyTimedelta64FillValue,
} from "./numpy-datetime64.js";

/** The zarr-extensions `numpy.timedelta64` data type. */
export const numpyTimedelta64 = numpyTemporalDataType("numpy.timedelta64");
