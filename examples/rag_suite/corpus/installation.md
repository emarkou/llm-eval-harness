# VectorDB Lite — Installation

## Requirements

VectorDB Lite requires Python 3.10 or higher. It has been tested on Linux, macOS,
and Windows (x86-64 and ARM64).

## Installing with pip

Install the core package:

```
pip install vectordb-lite
```

To enable GPU-accelerated index operations (requires CUDA 11.8+):

```
pip install vectordb-lite[gpu]
```

To install all optional extras including monitoring hooks and S3 persistence:

```
pip install vectordb-lite[all]
```

## Verifying the installation

After installation, confirm everything is working:

```
python -c "import vectordb_lite; print(vectordb_lite.__version__)"
```

A successful output looks like `0.9.2` (or the current release version).

## Default data directory

VectorDB Lite stores persistent index files in `~/.vectordb_lite/data` by default.
This directory is created automatically on first use. You can override this location
via the `VECTORDB_LITE_DATA_DIR` environment variable or in the configuration file.

## Upgrading

```
pip install --upgrade vectordb-lite
```

Upgrading between minor versions is generally safe. Upgrading across major versions
may require an index migration — see the migration guide for details.
