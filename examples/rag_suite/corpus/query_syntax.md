# VectorDB Lite — Query Syntax

## Initialising the client

```python
from vectordb_lite import VectorDB

db = VectorDB()           # uses config file / env vars
db = VectorDB(config_path="/path/to/config.toml")  # explicit config
```

## Inserting vectors

Insert a single vector with optional metadata:

```python
db.insert(vector=[0.1, 0.2, 0.3, ...], metadata={"id": "doc-001", "text": "Hello world"})
```

Batch insert for higher throughput:

```python
db.insert_batch(vectors=[[...], [...]], metadatas=[{"id": "a"}, {"id": "b"}])
```

Each inserted vector is assigned a unique integer `vector_id` returned by the call.

## Searching

Basic nearest-neighbour search (returns top 10 results by default):

```python
results = db.search(query_vector=[0.1, 0.2, ...], top_k=10)
```

Each element in `results` is a `SearchResult` object with three fields:
- `vector_id` — the integer ID assigned at insert time
- `score` — the similarity score (higher is more similar for cosine and dot-product)
- `metadata` — the dict you supplied at insert time

## Filtered search

Narrow results using metadata filters before the similarity search:

```python
results = db.search(
    query_vector=[...],
    top_k=5,
    filter={"category": {"$eq": "science"}, "year": {"$gt": 2020}},
)
```

Supported filter operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`.

## Updating and deleting

Update a vector's embedding and/or metadata by its ID:

```python
db.update(vector_id=42, new_vector=[...], new_metadata={"id": "doc-001", "text": "Updated"})
```

Delete a vector by ID:

```python
db.delete(vector_id=42)
```

## Persistence

Changes are written to disk synchronously after each mutating operation when
`persistence_enabled = true`. Call `db.flush()` explicitly to force a write
without making a change.
