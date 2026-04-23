# VectorDB Lite — Limitations

## Capacity limits

The maximum number of vectors depends on the chosen index type:

- **flat** index: up to **1 million** vectors (exact brute-force search)
- **hnsw** index: up to **10 million** vectors (approximate nearest-neighbour)
- **ivf** index: up to **50 million** vectors (approximate, requires training phase)

Exceeding the configured `max_vectors` limit raises a `CapacityError` at insert time.

## Vector dimensions

Supported vector dimensionality: **2 to 65,536**. Vectors in a single store must
all share the same dimension; mixing dimensions raises a `DimensionMismatchError`.

## Distributed and multi-node deployments

VectorDB Lite is a **single-node** library. There is no built-in support for
distributed deployments, sharding, or replication. If you need horizontal scaling,
consider using a dedicated vector database service.

## Memory constraints

In-memory operation is limited by the available RAM on the host machine. Each
32-bit float vector of dimension D consumes approximately `4 * D` bytes. A flat
index of 1 million 1536-dimensional vectors requires roughly **6 GB** of RAM.

## Concurrency

VectorDB Lite is **thread-safe for read operations** (searches). Write operations
(insert, update, delete) are not internally serialised; callers must apply
external locking (e.g., `threading.Lock`) if writes can occur concurrently.

## Persistence behaviour

Persistence is **synchronous and blocking**: each write call blocks until the
index snapshot is flushed to disk. This is simple and correct but can introduce
latency spikes under high write load.

## Security

VectorDB Lite has **no built-in authentication or access control**. All callers
with access to the process can read and write the index. Secure it at the
network or OS level.

## Backup and recovery

There is no built-in incremental backup mechanism. Snapshots are full copies of
the index. Point-in-time recovery is not supported natively.
