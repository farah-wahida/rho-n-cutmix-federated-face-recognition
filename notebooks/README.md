# Notebooks

The original research notebook contained repeated experiment cells, machine-specific paths, embedded figures, and large outputs. Its reusable logic has been moved into src/privacy_pipeline/ and scripts/.

If a tutorial notebook is added later, keep it as a thin caller of the package, clear all outputs before committing, and do not duplicate training or evaluation implementations.
