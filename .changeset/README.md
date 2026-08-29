# Changesets

Release-note fragments, one file per user-visible change — the npm-world
analog of the towncrier fragments used in zarr-python. Add one to a PR with:

```bash
npx changeset
```

pick the bump level (patch/minor/major) and write the changelog entry. The
Release workflow maintains a "Version Packages" PR that consumes pending
changesets; merging it bumps the version, updates CHANGELOG.md, and
publishes to npm. See RELEASING.md.
