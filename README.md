# zarr-metadata (TypeScript)

Spec-defined metadata types and structural validators for
[Zarr](https://zarr.dev) v2 and v3, in TypeScript. A port of the Python
[`zarr-metadata`](https://pypi.org/project/zarr-metadata/) package (the
reference implementation), kept in lockstep by a shared conformance corpus.

```ts
import {
  validateMetadataV3,
  safeParseArrayMetadataV2,
  isEmptyTree,
  flattenTree,
} from "zarr-metadata";

// Validation produces a tree of errors mirroring the document's shape;
// an empty tree means the document is valid.
const errors = validateMetadataV3(JSON.parse(text));
if (!isEmptyTree(errors)) {
  errors.children.get("codecs"); // the subtree of codec problems
  flattenTree(errors); // the flat view, for diagnostics:
  // [{ path: ["codecs"], kind: "invalid_value", message: "expected at least one codec" }]
}

// Or the discriminated-union form:
const result = safeParseArrayMetadataV2(JSON.parse(zarrayText));
if (result.success) {
  result.value.shape; // typed as ZarrV2ArrayMetadataJSON
} else {
  result.errors; // the ErrorTree
}
```

Each document kind gets four entry points:

- `validate*(value)` → `ErrorTree` (empty tree means valid)
- `safeParse*(value)` → `{ success: true, value } | { success: false, errors }`
- `is*(value)` → type guard
- `parse*(value)` → narrowed document or throws `MetadataValidationError`
  (which carries the tree as `.errors` and its flat view as `.issues`)

Covered documents: v3 array/group (`zarr.json`, including inline
`consolidated_metadata`), v2 array/group merged forms (`.zarray` /
`.zgroup` + `.zattrs`), and v2 consolidated metadata (`.zmetadata`).
The `MetadataV3` family dispatches on `node_type` for consumers handed an
arbitrary `zarr.json`.

The error-tree shape follows TypeScript validation-library convention
(compare Zod's `treeifyError` and `safeParse`) rather than the Python
package's flat problem lists; `flattenTree`/`treeOf` convert between the
two, and the flat path+kind form remains the cross-language interchange
format the conformance corpus asserts on.

Validation is structural (key presence, value shapes, fixed literals), not
domain-level: extension points (codecs, chunk grids, data types) are never
interpreted, matching the Python package's layering.

Consumers include the
[Zarr Metadata VS Code extension](https://github.com/d-v-b/vscode-zarr).

## How correctness is maintained

The [conformance corpus](conformance/) is the contract with the Python
reference implementation: shared JSON fixtures asserting the exact problem
set (loc + kind) each document must produce. Both test suites run every
case; a spec change updates the corpus first, turning both implementations
red until each is fixed.

## Development

```bash
npm install
npm run build       # tsc → dist/
npm test            # vitest: conformance corpus + unit tests
```

Check the corpus against the Python reference implementation (needs `uv`;
`--with` takes the PyPI package or a path to a local zarr-python checkout's
`packages/zarr-metadata`):

```bash
uv run --with zarr-metadata --no-project python scripts/check_conformance.py
```
