# zarr-metadata (TypeScript)

Spec-defined metadata types and structural validators for
[Zarr](https://zarr.dev) v2 and v3, in TypeScript. A port of the Python
[`zarr-metadata`](https://pypi.org/project/zarr-metadata/) package (the
reference implementation), kept in lockstep by a shared conformance corpus.

```ts
import { validateMetadataV3, parseArrayMetadataV2 } from "zarr-metadata";

// Every problem, not just the first — each with a machine-readable kind
// and a loc path from the document root.
const problems = validateMetadataV3(JSON.parse(text));
// [{ loc: ["codecs"], kind: "invalid_value", message: "expected at least one codec" }]

// Or throw: parse* narrows to the document type or throws
// MetadataValidationError carrying `.problems`.
const zarray = parseArrayMetadataV2(JSON.parse(zarrayText));
```

Each document kind gets three entry points:

- `validate*(value)` → `ValidationProblem[]` (empty means valid)
- `is*(value)` → type guard
- `parse*(value)` → narrowed document or `MetadataValidationError`

Covered documents: v3 array/group (`zarr.json`, including inline
`consolidated_metadata`), v2 array/group merged forms (`.zarray` /
`.zgroup` + `.zattrs`), and v2 consolidated metadata (`.zmetadata`).
`validateMetadataV3` dispatches on `node_type` for consumers handed an
arbitrary `zarr.json`.

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
