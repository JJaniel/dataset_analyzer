# test_embeddings.py
import os
import chromadb
from tools.pubmed_embedding import PubMedEmbedding
from tools.utils import read_dataset_sample
import json

def test_standalone_embedding():
    """Test script to create and verify embeddings independently"""
    
    # Initialize consistent persistent client
    client = chromadb.PersistentClient(path="./chromadb")
    collection = client.get_or_create_collection("features")
    
    # Test with a simple dataset
    test_data_path = input("Enter path to a test CSV file: ").strip()
    
    if not os.path.exists(test_data_path):
        print(f"File not found: {test_data_path}")
        return
    
    # Read sample
    df_sample = read_dataset_sample(test_data_path, sample_size=50)
    if df_sample is None:
        print("Failed to read dataset")
        return
    
    print(f"Dataset loaded: {df_sample.shape}")
    
    # Initialize embedding model
    model = PubMedEmbedding()
    
    # Process first 3 columns as test
    test_columns = df_sample.columns[:3]
    print(f"Testing with columns: {test_columns}")
    
    embeddings_created = 0
    
    for column in test_columns:
        try:
            # Simple test embedding without LLM
            embedding_text = f"Feature: {column}, Type: test_column"
            embedding = model.embed_documents([embedding_text])[0]
            
            # Store in ChromaDB
            metadata = {
                "dataset_id": test_data_path,
                "column_name": column,
                "semantic_meaning": f"Test embedding for {column}",
                "feature_type": "test"
            }
            
            collection.add(
                ids=[f"{test_data_path}::{column}"],
                embeddings=[embedding],
                metadatas=[metadata]
            )
            
            embeddings_created += 1
            print(f"✅ Created embedding for: {column}")
            
        except Exception as e:
            print(f"❌ Failed to embed {column}: {e}")
    
    print(f"\n📊 Test Results:")
    print(f"Embeddings created: {embeddings_created}")
    
    # Verify retrieval
    try:
        results = collection.get(include=['embeddings', 'metadatas'])
        print(f"Total embeddings in DB: {len(results['ids']) if results and 'ids' in results else 0}")
        
        if results and 'ids' in results:
            print("Sample IDs:", results['ids'][:3])
        
        return embeddings_created > 0
        
    except Exception as e:
        print(f"❌ Error retrieving embeddings: {e}")
        return False

def verify_existing_embeddings():
    """Check what's currently in the database"""
    try:
        client = chromadb.PersistentClient(path="./chromadb")
        collections = client.list_collections()
        
        print(f"Available collections: {[c.name for c in collections]}")
        
        if any(c.name == "features" for c in collections):
            collection = client.get_collection("features")
            results = collection.get(include=['embeddings', 'metadatas'])
            
            print(f"\n📊 Current Database Status:")
            print(f"Total embeddings: {len(results['ids']) if results and 'ids' in results else 0}")
            
            if results and 'metadatas' in results:
                datasets = set()
                for meta in results['metadatas']:
                    if meta and 'dataset_id' in meta:
                        dataset_name = os.path.basename(meta['dataset_id'])
                        datasets.add(dataset_name)
                
                print(f"Datasets: {list(datasets)}")
        else:
            print("No 'features' collection found")
            
    except Exception as e:
        print(f"Error checking database: {e}")

if __name__ == "__main__":
    print("=== Standalone Embedding Test ===")
    
    print("\n1. Checking existing embeddings...")
    verify_existing_embeddings()
    
    print("\n2. Creating test embeddings...")
    success = test_standalone_embedding()
    
    print("\n3. Final verification...")
    verify_existing_embeddings()
    
    if success:
        print("\n✅ Embedding test successful!")
    else:
        print("\n❌ Embedding test failed!")
