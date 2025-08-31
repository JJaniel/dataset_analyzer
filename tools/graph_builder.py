import networkx as nx
import chromadb
import itertools
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def debug_chromadb():
    """Debug function to see what's in ChromaDB"""
    # *** FIXED: Use PersistentClient consistently ***
    client = chromadb.PersistentClient(path="./chromadb")  # Changed from Client()
    try:
        collection = client.get_collection("features")
        # *** CRITICAL FIX: Include embeddings and metadatas ***
        results = collection.get(include=['embeddings', 'metadatas'])
        
        print(f"\n🔍 ChromaDB Debug:")
        print(f" - Collection exists: True")
        print(f" - Total items: {len(results['ids']) if results and 'ids' in results else 0}")
        print(f" - Has embeddings: {'embeddings' in results if results else False}")
        print(f" - Has metadatas: {'metadatas' in results if results else False}")
        
        if results and 'ids' in results and len(results['ids']) > 0:
            print(f" - Sample IDs: {results['ids'][:3]}")
            
            # Show sample datasets
            datasets = set()
            for metadata in results.get('metadatas', [])[:10]:
                if metadata and 'dataset_id' in metadata:
                    datasets.add(metadata['dataset_id'].split('\\')[-1] if '\\' in metadata['dataset_id'] else metadata['dataset_id'].split('/')[-1])
            print(f" - Datasets found: {list(datasets)[:5]}")
            
        return results
        
    except Exception as e:
        print(f"❌ ChromaDB Error: {e}")
        return None

def build_graph(similarity_threshold=0.85):
    """Build graph using ChromaDB with consistent PersistentClient"""
    # *** FIXED: Use PersistentClient consistently ***
    client = chromadb.PersistentClient(path="./chromadb")  # Changed from Client()
    
    try:
        collection = client.get_collection("features")
        print("✅ Successfully connected to ChromaDB collection 'features'")
    except Exception as e:
        print(f"❌ Error accessing ChromaDB collection 'features': {e}")
        return nx.Graph()

    try:
        # *** CRITICAL FIX: Include embeddings explicitly ***
        results = collection.get(include=['embeddings', 'metadatas'])
        
        # Check if we got the data we need
        embeddings = results.get('embeddings') if results else None
        metadatas = results.get('metadatas') if results else None
        
        if embeddings is None or len(embeddings) == 0:  # ✅ Fixed
            print("⚠️ No embeddings found in ChromaDB collection.")
            print("💡 Make sure Phase 1 (feature embedding) completed successfully.")
            return nx.Graph()
        
        if not metadatas or len(metadatas) != len(embeddings):
            print("⚠️ Missing or mismatched metadata for embeddings.")
            print(f"📊 Embeddings: {len(embeddings)}, Metadata: {len(metadatas) if metadatas else 0}")
            return nx.Graph()
        
        print(f"🔍 Found {len(embeddings)} embeddings for graph building")
        
        G = nx.Graph()
        
        # Add all datasets as nodes
        all_datasets = set()
        for meta in metadatas:
            if meta and 'dataset_id' in meta:
                # Extract filename from full path
                dataset_name = meta['dataset_id'].split('\\')[-1] if '\\' in meta['dataset_id'] else meta['dataset_id'].split('/')[-1]
                all_datasets.add(dataset_name)
        
        for dataset in all_datasets:
            G.add_node(dataset)
        
        print(f"📊 Added {len(all_datasets)} dataset nodes: {sorted(list(all_datasets))}")
        
        # Compare all pairs of features
        connections_found = 0
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                meta1 = metadatas[i]
                meta2 = metadatas[j]
                
                # Extract dataset names
                dataset1 = meta1['dataset_id'].split('\\')[-1] if '\\' in meta1['dataset_id'] else meta1['dataset_id'].split('/')[-1]
                dataset2 = meta2['dataset_id'].split('\\')[-1] if '\\' in meta2['dataset_id'] else meta2['dataset_id'].split('/')[-1]
                
                # Only compare features from different datasets
                if dataset1 != dataset2:
                    # *** FIXED SIMILARITY CALCULATION ***
                    # Proper cosine similarity computation
                    emb1 = np.array(embeddings[i])
                    emb2 = np.array(embeddings[j])
                    
                    # Cosine similarity
                    dot_product = np.dot(emb1, emb2)
                    norm1 = np.linalg.norm(emb1)
                    norm2 = np.linalg.norm(emb2)
                    
                    if norm1 > 0 and norm2 > 0:
                        similarity = dot_product / (norm1 * norm2)
                    else:
                        similarity = 0
                    
                    if similarity > similarity_threshold:
                        connections_found += 1
                        
                        # Create edge label with feature similarity info
                        feature1 = meta1.get('column_name', 'unknown')
                        feature2 = meta2.get('column_name', 'unknown')
                        edge_label = f"{feature1} ↔ {feature2} ({similarity:.3f})"
                        
                        # Add or update edge between datasets
                        if G.has_edge(dataset1, dataset2):
                            # Strengthen existing connection
                            existing_weight = G[dataset1][dataset2].get('weight', 0)
                            existing_connections = G[dataset1][dataset2].get('connections', [])
                            G[dataset1][dataset2]['weight'] = max(existing_weight, similarity)
                            G[dataset1][dataset2]['connections'] = existing_connections + [edge_label]
                        else:
                            # Add new edge
                            G.add_edge(dataset1, dataset2,
                                     weight=similarity,
                                     label=edge_label,
                                     connections=[edge_label])
                        
                        print(f"🔗 Found connection: {dataset1} ↔ {dataset2} via {feature1} ↔ {feature2} (similarity: {similarity:.3f})")
        
        print(f"\n📈 Graph Building Results:")
        print(f" - Dataset nodes: {G.number_of_nodes()}")
        print(f" - Dataset connections: {G.number_of_edges()}")
        print(f" - Feature-level connections found: {connections_found}")
        print(f" - Similarity threshold used: {similarity_threshold}")
        
        if G.number_of_edges() == 0:
            print(f"⚠️ No connections found above threshold {similarity_threshold}")
            print(f"💡 Try lowering the similarity threshold (e.g., 0.7 or 0.6)")
        else:
            print(f"✅ Successfully built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            
            # Show strongest connections
            print(f"\n🏆 Strongest dataset connections:")
            edges_by_weight = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
            for dataset1, dataset2, data in edges_by_weight[:5]:
                weight = data['weight']
                connection_count = len(data.get('connections', []))
                print(f" • {dataset1} ↔ {dataset2}: {weight:.3f} ({connection_count} feature connections)")
        
        return G
        
    except Exception as e:
        print(f"❌ Unexpected error in build_graph(): {e}")
        print(f"📊 Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return nx.Graph()

def analyze_graph_relationships(G):
    """
    Analyzes the built graph to find interesting relationships between datasets.
    """
    if G.number_of_nodes() == 0:
        return {"message": "Empty graph - no relationships to analyze"}
    
    analysis = {}
    
    # Basic graph metrics
    analysis['total_datasets'] = G.number_of_nodes()
    analysis['total_connections'] = G.number_of_edges()
    analysis['connected_components'] = nx.number_connected_components(G)
    analysis['graph_density'] = nx.density(G)
    
    # Find most connected datasets
    degree_centrality = nx.degree_centrality(G)
    analysis['most_connected_datasets'] = sorted(
        degree_centrality.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    # Find strongest connections
    if G.number_of_edges() > 0:
        strongest_connections = []
        for dataset1, dataset2, data in G.edges(data=True):
            strongest_connections.append({
                'dataset1': dataset1,
                'dataset2': dataset2,
                'similarity': data.get('weight', 0),
                'feature_connections': len(data.get('connections', [])),
                'connection_details': data.get('connections', [])
            })
        
        analysis['strongest_connections'] = sorted(
            strongest_connections,
            key=lambda x: x['similarity'],
            reverse=True
        )[:10]
    
    return analysis
