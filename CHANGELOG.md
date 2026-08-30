# zarr-metadata

## 0.5.0

### Minor Changes

- Consolidated metadata gets full-depth coverage beyond structure. New
  `validateSemanticsV3` applies the semantic rules to a group's inline
  consolidated entries recursively (nested groups included), pathing issues
  through `consolidated_metadata.metadata.<key>`; `validateArraySemanticsV3`
  is unchanged. New `validateConsolidatedDocumentsV2` interprets `.zmetadata`
  entries by key suffix — `.zarray`/`.zgroup` as on-disk documents (also
  rejecting an `attributes` member, which belongs in the sibling `.zattrs`),
  `.zattrs` as a JSON object — the consumer-side layer the reference model
  deliberately leaves out; `validateConsolidatedMetadataV2` stays
  envelope-only for conformance-corpus parity.

## 0.4.1

### Patch Changes

- `validateArraySemanticsV3` now threads dimensional context through codec
  pipelines instead of assuming it constant (mirroring zarr-python's
  chunk-spec threading, d-v-b/zarr-python#303): a valid `transpose` permutes
  the per-dimension chunk sizes downstream shards must divide, and any codec
  the layer cannot reason about — `reshape` may change a chunk's rank,
  unknown codecs may do anything — invalidates the dimensional context for
  the rest of the pipeline. Fixes false rejections of spec-endorsed
  reshape→transpose chains and of shards sized for the transposed chunk, and
  a missed rejection of shards sized for the un-transposed chunk.

## 0.4.0

### Minor Changes

- `validateArraySemanticsV3` now interprets the `rectilinear` chunk grid:
  `chunk_shapes` must have one entry per dimension of `shape`, explicit chunk
  lists (integers and `[size, count]` run-length pairs) must sum exactly to
  their dimension's length (the bare-integer uniform shorthand carries no sum
  constraint), and a `sharding_indexed` codec's inner chunks must now evenly
  divide every distinct per-dimension chunk size of the grid — uniform or
  rectilinear — rather than only a regular grid's chunk shape.

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
