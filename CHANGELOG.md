# zarr-metadata

## 0.3.0

### Minor Changes

- Add `validateArraySemanticsV3`, a semantic (cross-field) layer over the
  structural validators: the regular chunk grid's arity against `shape`,
  `transpose` orders as dimension permutations, `sharding_indexed` inner
  chunk shapes matching dimensionality and evenly dividing the chunk they
  shard (recursively), and `fill_value` fitting the named core data type.
  Unrecognized extension names are skipped. This layer has no Python
  counterpart and is covered by package tests rather than the conformance
  corpus.

## 0.2.0

### Minor Changes

- Add `mustUnderstandExtensionFieldsV3`, reporting the extension-field keys of
  a v3 array or group document that a reader is obligated to understand — every
  unrecognized top-level key not waived with an explicit
  `"must_understand": false` object, per the v3 spec's rule that
  implementations must refuse to open nodes carrying obligated fields they do
  not recognize.
