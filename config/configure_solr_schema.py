"""Configure Solr schema via Schema API"""
import requests
import json

SOLR_URL = "http://localhost:8983/solr/ai_tools_opinions"

def add_field_type(name, config):
    """Add field type to Solr schema"""
    url = f"{SOLR_URL}/schema"
    response = requests.post(url, json={"add-field-type": config})
    if response.status_code == 200:
        print(f"✓ Added field type: {name}")
    else:
        print(f"✗ Failed to add field type {name}: {response.text}")

def add_field(config):
    """Add field to Solr schema"""
    url = f"{SOLR_URL}/schema"
    response = requests.post(url, json={"add-field": config})
    if response.status_code == 200:
        print(f"✓ Added field: {config['name']}")
    else:
        print(f"⚠ Field {config['name']}: {response.text}")

def add_copy_field(source, dest):
    """Add copy field to Solr schema"""
    url = f"{SOLR_URL}/schema"
    response = requests.post(url, json={"add-copy-field": {"source": source, "dest": dest}})
    if response.status_code == 200:
        print(f"✓ Added copy field: {source} → {dest}")
    else:
        print(f"⚠ Copy field {source}→{dest}: {response.text}")

def main():
    print("Configuring Solr schema for AI Tools Opinion Search...")
    print("=" * 60)

    # Add text_en field type (if not exists)
    print("\n1. Adding field types...")
    add_field_type("text_en", {
        "name": "text_en",
        "class": "solr.TextField",
        "positionIncrementGap": "100",
        "analyzer": {
            "tokenizer": {"class": "solr.StandardTokenizerFactory"},
            "filters": [
                {"class": "solr.StopFilterFactory", "ignoreCase": "true", "words": "lang/stopwords_en.txt"},
                {"class": "solr.LowerCaseFilterFactory"},
                {"class": "solr.EnglishPossessiveFilterFactory"},
                {"class": "solr.PorterStemFilterFactory"}
            ]
        }
    })

    # Add DenseVectorField type for Solr 10 HNSW vector search
    add_field_type("knn_vector", {
        "name": "knn_vector",
        "class": "solr.DenseVectorField",
        "vectorDimension": 384,
        "similarityFunction": "cosine"
    })

    # Add fields
    print("\n2. Adding fields...")
    fields = [
        {"name": "doc_id", "type": "string", "stored": True, "indexed": True, "required": True},
        {"name": "title", "type": "text_en", "stored": True, "indexed": True},
        {"name": "text", "type": "text_en", "stored": True, "indexed": True},
        {"name": "combined_content", "type": "text_en", "stored": False, "indexed": True},
        {"name": "source", "type": "string", "stored": True, "indexed": True, "docValues": True},
        {"name": "url", "type": "string", "stored": True, "indexed": False},
        {"name": "content_type", "type": "string", "stored": True, "indexed": True, "docValues": True},
        {"name": "tool_mentioned", "type": "string", "stored": True, "indexed": True, "multiValued": True, "docValues": True},
        {"name": "sentiment_label", "type": "string", "stored": True, "indexed": True, "docValues": True},
        {"name": "subjectivity", "type": "string", "stored": True, "indexed": True, "docValues": True},
        {"name": "aspects", "type": "string", "stored": True, "indexed": True, "multiValued": True, "docValues": True},
        {"name": "sarcasm", "type": "string", "stored": True, "indexed": True, "docValues": True},
        {"name": "subreddit", "type": "string", "stored": True, "indexed": True, "docValues": True},
        {"name": "author", "type": "string", "stored": True, "indexed": True},
        {"name": "date", "type": "pdate", "stored": True, "indexed": True, "docValues": True},
        {"name": "upvotes", "type": "pint", "stored": True, "indexed": True},
        {"name": "num_replies", "type": "pint", "stored": True, "indexed": False},
        # Vector field for Solr 10 HNSW semantic search
        {"name": "vector", "type": "knn_vector", "stored": False, "indexed": True}
    ]

    for field in fields:
        add_field(field)

    # Add copy fields
    print("\n3. Adding copy fields...")
    add_copy_field("title", "combined_content")
    add_copy_field("text", "combined_content")

    print("\n" + "=" * 60)
    print("✅ Schema configuration complete!")
    print("\nNext steps:")
    print("  1. Re-run indexing: python indexing/run_indexing.py")
    print("  2. Start API: python api/app.py")
    print("  3. Test search: http://localhost:5000")

if __name__ == '__main__':
    main()
