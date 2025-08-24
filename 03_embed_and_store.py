# 03_embed_and_store.py
# Usage:
#   python 03_embed_and_store.py --chunks data\\chunks.jsonl --persist-dir vectorstore\\chroma --collection delhi_laws
#
# Creates a persistent Chroma DB with collection 'delhi_laws' using sentence-transformers
# model 'all-MiniLM-L6-v2'. Safe on re-runs (skips existing IDs).

import argparse
import json
import os
import time
from pathlib import Path
from typing import List

import chromadb
from chromadb.utils import embedding_functions

def log(msg: str):
    print(msg, flush=True)

def batched(iterable, n=256):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="Path to chunks.jsonl from step 2")
    ap.add_argument("--persist-dir", default="vectorstore/chroma", help="Chroma persistence directory")
    ap.add_argument("--collection", default="delhi_laws", help="Collection name")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Sentence-Transformers model name")
    ap.add_argument("--reset", action="store_true", help="If set, deletes existing collection first")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    # Disable Chroma telemetry noise
    os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"

    t0 = time.time()
    log(f"[INIT] chunks={args.chunks} persist_dir={args.persist_dir} "
        f"collection={args.collection} model={args.model} batch_size={args.batch_size}")

    persist_dir = Path(args.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    log("[STEP] Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=str(persist_dir))

    log("[STEP] Loading embedding model...")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=args.model)

    if args.reset:
        try:
            client.delete_collection(args.collection)
            log(f"[INFO] Deleted existing collection '{args.collection}'")
        except Exception as e:
            log(f"[WARN] Could not delete collection: {e}")

    log(f"[STEP] Getting/creating collection '{args.collection}'...")
    coll = client.get_or_create_collection(name=args.collection, embedding_function=embed_fn)

    # Track IDs if you want deduplication (currently no-op)
    existing_ids = set()

    ids: List[str] = []
    docs: List[str] = []
    metas: List[dict] = []

    log(f"[STEP] Reading chunks from {args.chunks}...")
    total = 0
    with open(args.chunks, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            rec = json.loads(line)
            rid = rec["id"]
            if rid in existing_ids:
                continue
            ids.append(rid)
            docs.append(rec["text"])
            metas.append(rec["metadata"])

            if len(ids) >= args.batch_size:
                coll.upsert(ids=ids, documents=docs, metadatas=metas)
                total += len(ids)
                log(f"[OK] Upserted batch of {len(ids)} (total={total})")
                ids, docs, metas = [], [], []

    if ids:
        coll.upsert(ids=ids, documents=docs, metadatas=metas)
        total += len(ids)
        log(f"[OK] Upserted final batch of {len(ids)} (total={total})")

    log(f"[DONE] Collection '{args.collection}' ready at {persist_dir} "
        f"with ~{total} records in {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
