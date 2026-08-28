/**
 * Compile-time machinery tying the runtime key arrays to the document types.
 *
 * The validators need key names at runtime, and TypeScript types are erased,
 * so the arrays in v2.ts / v3.ts cannot be derived from the types the way
 * the Python package derives its key sets from `__required_keys__`. What the
 * type system CAN do is the reverse: prove each array is exactly the set of
 * required (or optional) keys declared by the document type, so the two can
 * never drift.
 */

/**
 * The declared properties of `T`, with index signatures stripped.
 *
 * The v3 document types carry a `[key: string]: ...` index signature for
 * extension fields; without stripping it, `keyof` collapses to `string` and
 * the per-key extraction below would be meaningless.
 */
type Known<T> = {
  [K in keyof T as string extends K ? never : number extends K ? never : K]: T[K];
};

/** The union of `T`'s required declared keys. */
export type RequiredKeysOf<T> = {
  [K in keyof Known<T>]-?: {} extends Pick<Known<T>, K> ? never : K;
}[keyof Known<T>];

/** The union of `T`'s optional declared keys. */
export type OptionalKeysOf<T> = {
  [K in keyof Known<T>]-?: {} extends Pick<Known<T>, K> ? K : never;
}[keyof Known<T>];

/**
 * Identity for a readonly tuple of keys, statically checked to be EXACTLY
 * the union `U`: the element constraint rejects keys outside `U`, and the
 * intersected conditional rejects the tuple when any member of `U` is
 * missing (surfacing as a "not assignable" error on the argument).
 *
 * Usage: `exactKeys<RequiredKeysOf<Doc>>()(["a", "b"])`.
 */
export function exactKeys<U extends PropertyKey>() {
  return <const A extends readonly U[]>(
    keys: A & ([U] extends [A[number]] ? unknown : never),
  ): A => keys;
}
