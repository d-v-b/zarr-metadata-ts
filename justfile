# Development verbs for the zarr-metadata TypeScript package. Recipes run
# with this directory as the working directory regardless of where `just`
# is invoked.

# List available recipes
default:
    @just --list

# Install dependencies
install:
    npm install

# Compile src/ to dist/
build:
    npm run build

# Run the test suite; extra args are passed to vitest
test *args:
    npx vitest run {{ args }}

# Type-check the package sources and the test suite (including the
# expectTypeOf assertions, which only exist under the type checker)
typecheck:
    npm run typecheck

# Run the shared conformance corpus against the Python reference
# implementation. `ref` is what `uv run --with` installs: the PyPI package
# by default, or a path to a local zarr-python checkout's
# packages/zarr-metadata to test unreleased changes, e.g.
# `just conformance ../zarr-python/packages/zarr-metadata`.
conformance ref="zarr-metadata":
    uv run --with '{{ ref }}' --no-project python scripts/check_conformance.py

# Run everything CI runs for this package
check: typecheck test conformance

# Remove build output
clean:
    npm run clean
