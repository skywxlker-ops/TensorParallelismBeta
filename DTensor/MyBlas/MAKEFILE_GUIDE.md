# MyBlas Makefile - Quick Reference

## Basic Commands

### Build Everything
```bash
make              # Build library + all tests
make lib          # Build only the library
make tests        # Build library + all defined tests
```

### Build & Run Specific Test
```bash
make test TEST=test_hgemm_v2       # Build specific test
make run TEST=test_hgemm_v2        # Build and run specific test
```

### Cleanup
```bash
make clean        # Remove all build artifacts
make rebuild      # Clean + rebuild everything
```

### Get Help
```bash
make help         # Show all available commands
```

## Adding New Tests

To add a new test, simply edit the `TESTS` list in the Makefile:

```makefile
TESTS := \
	test_hgemm_v2 \
	test_batched_gemm \
	test_hgemm \
	your_new_test \     # Add here
	test_sgemm
```

That's it! No need to reconfigure or regenerate anything.

## Adding New Source Files

**Nothing to do!** The Makefile auto-discovers all `.cpp` and `.cu` files in the `src/` directory.

Just create your file:
- `src/level3/your_new_kernel.cu`

Run `make` and it will be automatically compiled and linked.

## Examples

```bash
# Quick development cycle for V2 kernel
make test TEST=test_hgemm_v2
make run TEST=test_hgemm_v2

# Rebuild after major changes
make rebuild

# Build just the library after kernel changes
make lib
```

## Build Output Location

- **Library**: `build/libmyblas.so`
- **Tests**: `build/test_*`
- **Objects**: `build/objects/`

## Key Differences from CMake

| Aspect | CMake | Makefile |
|--------|-------|----------|
| **Adding tests** | Auto-discovers (even incomplete) | Explicit list |
| **Reconfiguration** | Required after changes | Never needed |
| **Build command** | `cmake --build build -j8` | `make -j8` |
| **Clean** | `rm -rf build/*` | `make clean` |
| **Simplicity** | Complex, multi-step | Simple, one command |

## Notes

- The Makefile uses `-j8` flag implicitly for parallel builds
- RPATH is set automatically so tests can find the library
- All builds use `-g` for debugging symbols
- Warnings are shown but don't stop the build
