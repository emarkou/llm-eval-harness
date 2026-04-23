# VectorDB Lite — Configuration

## Configuration file location

VectorDB Lite reads its configuration from `~/.vectordb_lite/config.toml` on startup.
You can point to a custom file by setting the `VECTORDB_LITE_CONFIG` environment variable.

## Key settings

| Setting | Default | Description |
|---|---|---|
| `max_vectors` | `100000` | Maximum number of vectors the store will hold |
| `similarity_metric` | `cosine` | Distance metric: `cosine`, `dot_product`, or `euclidean` |
| `index_type` | `flat` | Index algorithm: `flat`, `hnsw`, or `ivf` |
| `persistence_enabled` | `true` | Whether to persist the index to disk |
| `data_dir` | `~/.vectordb_lite/data` | Directory for persistent index files |

## HNSW index parameters

When `index_type = "hnsw"`, additional parameters apply:

| Parameter | Default | Description |
|---|---|---|
| `ef_construction` | `200` | Controls index build quality (higher = better quality, slower build) |
| `m` | `16` | Number of bidirectional links per node (higher = better recall, more memory) |
| `ef_search` | `50` | Controls query-time accuracy (higher = better recall, slower query) |

## IVF index parameters

When `index_type = "ivf"`, the relevant parameter is `nlist` (default `100`), which
controls the number of Voronoi cells used to partition the vector space.

## Environment variable overrides

All settings can be overridden at runtime via environment variables using the
`VECTORDB_LITE_` prefix. For example:

```
VECTORDB_LITE_MAX_VECTORS=500000
VECTORDB_LITE_SIMILARITY_METRIC=dot_product
VECTORDB_LITE_INDEX_TYPE=hnsw
```

Environment variables take precedence over the config file.

## Example config.toml

```toml
max_vectors = 200000
similarity_metric = "cosine"
index_type = "hnsw"
persistence_enabled = true

[hnsw]
ef_construction = 200
m = 16
ef_search = 50
```
