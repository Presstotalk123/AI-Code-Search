"""
Solr setup script: apply schema and re-index from scratch.

Run this whenever the Docker volume is wiped or the schema needs to be re-applied:
    python config/setup_solr.py

What it does:
  1. Copies merged-managed-schema.xml into the container (includes all default Solr
     field types + our custom fields + knn_vector with dot_product/hnswBeamWidth=200)
  2. Reloads the Solr core to apply the schema
  3. Clears all existing documents (wipes the old HNSW graph)
  4. Re-indexes all documents with vector embeddings
  5. Optimizes the index to a single segment (one global HNSW graph, same as ChromaDB)
"""
import subprocess
import sys
import os
import requests
import yaml

CONTAINER = "ai_tools_solr"
CORE = "ai_tools_opinions"
SOLR_URL = f"http://localhost:8983/solr/{CORE}"
SCHEMA_SRC = "config/merged-managed-schema.xml"
SCHEMA_DEST = f"/var/solr/data/{CORE}/conf/managed-schema.xml"


def run(cmd, check=True):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"ERROR: {result.stderr or result.stdout}")
        sys.exit(1)
    return result.stdout.strip()


def step(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}")


def main():
    # Verify schema file exists
    if not os.path.exists(SCHEMA_SRC):
        print(f"ERROR: {SCHEMA_SRC} not found. Run the merge script first or check config/.")
        sys.exit(1)

    # Verify container is running
    result = run(f"docker ps --filter name={CONTAINER} --format {{{{.Names}}}}", check=False)
    if CONTAINER not in result:
        print(f"ERROR: Container '{CONTAINER}' is not running. Start it with: docker-compose up -d")
        sys.exit(1)

    step("Step 1: Copying schema into container")
    run(f'docker cp "{SCHEMA_SRC}" "{CONTAINER}:/tmp/managed-schema.xml"')
    run(f'docker exec -u root {CONTAINER} //bin/bash -c "cp //tmp/managed-schema.xml {SCHEMA_DEST}"')
    print("Schema copied.")

    step("Step 2: Reloading Solr core")
    r = requests.get(f"http://localhost:8983/solr/admin/cores?action=RELOAD&core={CORE}")
    data = r.json()
    if data["responseHeader"]["status"] != 0:
        print(f"ERROR: Core reload failed: {data.get('error', {}).get('msg')}")
        sys.exit(1)
    print("Core reloaded successfully.")

    # Verify knn_vector field type
    r = requests.get(f"{SOLR_URL}/schema/fieldtypes/knn_vector")
    ft = r.json().get("fieldType", {})
    print(f"knn_vector: similarityFunction={ft.get('similarityFunction')}, "
          f"hnswBeamWidth={ft.get('hnswBeamWidth')}, "
          f"hnswMaxConnections={ft.get('hnswMaxConnections')}")
    if ft.get("similarityFunction") != "dot_product":
        print("WARNING: knn_vector does not have dot_product similarity. Check the schema file.")

    step("Step 3: Clearing existing documents")
    import pysolr
    solr = pysolr.Solr(SOLR_URL, timeout=30)
    solr.delete(q="*:*")
    solr.commit()
    count = solr.search("*:*", rows=0).hits
    print(f"Documents remaining: {count}")

    step("Step 4: Re-indexing")
    print("Running indexing/run_indexing.py (answer 'yes' when prompted)...")
    # Run indexer as subprocess so it handles its own logging/prompts
    result = subprocess.run(
        [sys.executable, "indexing/run_indexing.py"],
        input="yes\n",
        text=True
    )
    if result.returncode != 0:
        print("ERROR: Indexing failed.")
        sys.exit(1)

    step("Step 5: Optimizing index (merging to single segment)")
    r = requests.get(f"{SOLR_URL}/update?optimize=true&maxSegments=1&waitFlush=true")
    if r.json()["responseHeader"]["status"] != 0:
        print("WARNING: Optimize failed. HNSW recall may be reduced.")
    else:
        r2 = requests.get(f"http://localhost:8983/solr/admin/cores?action=STATUS&core={CORE}")
        idx = r2.json()["status"][CORE].get("index", {})
        print(f"numDocs={idx.get('numDocs')} | segments={idx.get('segmentCount')}")

    print("\n" + "="*60)
    print("Setup complete. Start the API with: python api/app.py")
    print("="*60)


if __name__ == "__main__":
    main()
