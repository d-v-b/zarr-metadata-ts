---
"zarr-metadata": minor
---

The semantic layer now interprets the established convention data types
alongside the core scalars: `string` (string fills), `bytes` (byte arrays
or base64), `numpy.datetime64`/`numpy.timedelta64` (required
`unit`/`scale_factor` configuration; integer or `"NaT"` fills), the `r<N>`
raw-bits family (N must be a positive multiple of 8; fills carry exactly
N/8 bytes), and `struct` (required `fields` configuration; the fill is an
object judged per field against each field's own data type, recursively,
with missing and extraneous fields flagged). The content modules are also
reorganized one file per interpreted name — `data-type/`, `chunk-grid/`,
and `codec/` directories mirroring the Python reference's layout — and the
new configuration/fill types are exported.
