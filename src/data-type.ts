/**
 * The core data types this package interprets: fill-value syntax per type,
 * in one place per family. Extension data types are skipped — their shapes
 * are registry-schema facts, and the name space is open.
 */

/** A float fill value: a JSON number, a non-finite sentinel, or a hex string. */
export type FloatFillValue = number | "NaN" | "Infinity" | "-Infinity" | string;

/** A complex fill value: `[real, imaginary]`, each a float fill form. */
export type ComplexFillValue = [FloatFillValue, FloatFillValue];

const INT_RANGES: Record<string, readonly [number, number]> = {
  int8: [-128, 127],
  int16: [-32768, 32767],
  int32: [-2147483648, 2147483647],
  int64: [-9223372036854775808, 9223372036854775807],
  uint8: [0, 255],
  uint16: [0, 65535],
  uint32: [0, 4294967295],
  uint64: [0, 18446744073709551615],
};

const FLOAT_HEX_DIGITS: Record<string, number> = {
  float16: 4,
  float32: 8,
  float64: 16,
};

/** Hex digit width of each complex type's component floats. */
const COMPLEX_COMPONENT_HEX_DIGITS: Record<string, number> = {
  complex64: 8,
  complex128: 16,
};

function isFloatFill(value: unknown, hexDigits: number): boolean {
  if (typeof value === "number") return true;
  if (typeof value !== "string") return false;
  if (value === "NaN" || value === "Infinity" || value === "-Infinity") return true;
  return new RegExp(`^0x[0-9a-fA-F]{${hexDigits}}$`).test(value);
}

function floatFillMessage(name: string, hexDigits: number): string {
  return (
    `expected a number, "NaN", "Infinity", "-Infinity", or a ` +
    `${hexDigits}-hex-digit "0x..." string for data type ${JSON.stringify(name)}`
  );
}

/** The fill_value issue for a KNOWN core data type, if any; undefined otherwise. */
export function fillValueIssue(dataTypeName: string, fill: unknown): string | undefined {
  if (dataTypeName === "bool") {
    return typeof fill === "boolean"
      ? undefined
      : 'expected a boolean fill value for data type "bool"';
  }
  const intRange = INT_RANGES[dataTypeName];
  if (intRange !== undefined) {
    const [low, high] = intRange;
    return Number.isInteger(fill) && (fill as number) >= low && (fill as number) <= high
      ? undefined
      : `expected an integer in [${low}, ${high}] for data type ${JSON.stringify(dataTypeName)}`;
  }
  const floatDigits = FLOAT_HEX_DIGITS[dataTypeName];
  if (floatDigits !== undefined) {
    return isFloatFill(fill, floatDigits)
      ? undefined
      : floatFillMessage(dataTypeName, floatDigits);
  }
  const complexDigits = COMPLEX_COMPONENT_HEX_DIGITS[dataTypeName];
  if (complexDigits !== undefined) {
    const ok =
      Array.isArray(fill) &&
      fill.length === 2 &&
      fill.every((part) => isFloatFill(part, complexDigits));
    return ok
      ? undefined
      : `expected a two-element [real, imaginary] array for data type ${JSON.stringify(dataTypeName)}`;
  }
  return undefined; // unrecognized data type: no verdict
}
