# vscode-zarr

Editor tooling for [Zarr](https://zarr.dev) metadata documents, built on a
TypeScript port of the Python
[`zarr-metadata`](https://pypi.org/project/zarr-metadata/) package.

Two deliverables live here:

| Package | What it is |
| --- | --- |
| [`packages/zarr-metadata-ts`](packages/zarr-metadata-ts) | npm `zarr-metadata`: spec-defined types and loc-aware structural validators for Zarr v2 and v3 metadata documents. A port of the Python reference implementation. |
| [`packages/extension`](packages/extension) | The **Zarr Metadata** VS Code extension: schema-based validation, hover docs, and completions for `zarr.json`, `.zarray`, `.zgroup`, `.zattrs`, and `.zmetadata`, plus cross-field diagnostics powered by the library. |

## How correctness is maintained

- **Conformance corpus** ([`conformance/`](conformance/)): shared JSON
  fixtures asserting the exact problem set (loc + kind) each document must
  produce. The TypeScript test suite and the Python reference implementation
  both run every case; `scripts/check_conformance.py` runs the Python side.
- **Generated schemas** ([`scripts/generate_schemas.py`](scripts/generate_schemas.py)):
  the extension's JSON Schemas are emitted from the Python package's
  pydantic schema-input types, never hand-edited.

## Development

```bash
npm install
npm run build          # library then extension
npm test               # vitest: conformance corpus + unit tests
```

Check the corpus against the Python reference implementation (needs `uv`;
`--with` takes the PyPI package or a path to a local zarr-python checkout’s packages/zarr-metadata):

```bash
uv run --with zarr-metadata --no-project python scripts/check_conformance.py
```

Regenerate the extension's schemas after a zarr-metadata change:

```bash
uv run --with 'pydantic>=2.13' --with zarr-metadata --no-project python scripts/generate_schemas.py
```

## Trying the extension

Open this folder in VS Code and press **F5** ("Run Zarr Metadata
extension"). The development host opens [`example/`](example/), which
contains valid and deliberately broken metadata documents —
`example/broken_array/zarr.json` shows a missing required key, an empty
codec pipeline, a `must_understand: false` on `data_type`, an unexpected
metadata-field member, and a `dimension_names`/`shape` mismatch, each with
its own precisely-placed squiggle.
