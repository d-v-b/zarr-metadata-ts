# zarr-metadata

## 0.2.0

### Minor Changes

- Add `mustUnderstandExtensionFieldsV3`, reporting the extension-field keys of
  a v3 array or group document that a reader is obligated to understand — every
  unrecognized top-level key not waived with an explicit
  `"must_understand": false` object, per the v3 spec's rule that
  implementations must refuse to open nodes carrying obligated fields they do
  not recognize.
