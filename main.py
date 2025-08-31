import os
import argparse
import json
import time

# New approach imports
from tools.feature_extractor import analyze_and_embed_features
from tools.graph_builder import build_graph, debug_chromadb, analyze_graph_relationships
from tools.report_generator import generate_report

# Old approach imports
from tools.data_analyzer import analyze_individual_dataset
from tools.data_synthesizer import synthesize_analyses
from tools.utils import read_dataset_sample


# In main.py or wherever you check status
def print_embedding_status():
    try:
        import chromadb
        # *** FIXED: Use PersistentClient ***
        client = chromadb.PersistentClient(path="./chromadb")  # ✅ Persistent, saved to disk

        collections = client.list_collections()
        
        if not any(c.name == "features" for c in collections):
            print("📝 No embeddings yet - 'features' collection does not exist")
            return {'total_embeddings': 0, 'datasets': {}}
        
        collection = client.get_collection("features")
        results = collection.get(include=['embeddings', 'metadatas'])
        
        if not results or 'ids' not in results or len(results['ids']) == 0:
            print("📝 'features' collection exists but is empty")
            return {'total_embeddings': 0, 'datasets': {}}
        
        total = len(results['ids'])
        
        # Count embeddings per dataset
        dataset_counts = {}
        for meta in results.get('metadatas', []):
            if meta and 'dataset_id' in meta:
                dataset_id = meta['dataset_id']
                dataset_name = dataset_id.split('\\')[-1] if '\\' in dataset_id else dataset_id.split('/')[-1]
                dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1
        
        print(f"📊 Vector DB Status:")
        print(f"  🔢 Total embeddings: {total}")
        print(f"  📁 Datasets with embeddings: {len(dataset_counts)}")
        print(f"  💾 Vector DB size: ~{total * 0.001:.1f}K vectors")
        
        if dataset_counts:
            print(f"  📋 Breakdown by dataset:")
            for dataset, count in sorted(dataset_counts.items()):
                print(f"     • {dataset}: {count} features")
        
        return {
            'total_embeddings': total,
            'datasets': dataset_counts,
            'collection_exists': True
        }
        
    except Exception as e:
        print(f"❌ Error checking vector DB status: {e}")
        return {'error': str(e)}


def get_embedding_progress_summary():
    """Get a compact summary for progress tracking."""
    status = print_embedding_status()
    if 'error' in status:
        return "Vector DB: Error"
    elif status['total_embeddings'] == 0:
        return "Vector DB: Empty"
    else:
        return f"Vector DB: {status['total_embeddings']} features from {len(status['datasets'])} datasets"


def check_chromadb_collection_exists():
    """Check if ChromaDB collection 'features' exists."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chromadb")  # Changed from Client()
        collections = [collection.name for collection in client.list_collections()]
        return "features" in collections
    except Exception as e:
        print(f"❌ Error checking ChromaDB: {e}")
        return False


def check_existing_embeddings():
    """Check if there are existing embeddings in ChromaDB and show user what's available."""
    if not check_chromadb_collection_exists():
        print("📝 No existing embeddings found - ChromaDB collection 'features' doesn't exist")
        return False, {}

    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chromadb")  # Changed from Client()
        collection = client.get_collection("features")
        # *** CRITICAL FIX: Include embeddings and metadatas ***
        results = collection.get(include=['embeddings', 'metadatas'])
        
        print(f"\n🔍 ChromaDB Status:")
        print(f"  - Collection exists: ✅")
        print(f"  - Total items: {len(results['ids']) if results and 'ids' in results else 0}")
        
        if results and 'ids' in results and len(results['ids']) > 0:
            # Count embeddings per dataset
            dataset_counts = {}
            for metadata in results.get('metadatas', []):
                if metadata and 'dataset_id' in metadata:
                    dataset_name = metadata['dataset_id'].split('\\')[-1] if '\\' in metadata['dataset_id'] else metadata['dataset_id'].split('/')[-1]
                    dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1
            
            print(f"📊 Found existing embeddings for {len(dataset_counts)} datasets:")
            for dataset, count in sorted(dataset_counts.items()):
                print(f"   • {dataset}: {count} features")
            
            return True, dataset_counts
        else:
            print("❌ Collection exists but contains no embeddings")
            return False, {}
            
    except Exception as e:
        print(f"❌ Error reading ChromaDB embeddings: {e}")
        return False, {}

def clear_chromadb():
    """Clear all existing embeddings from ChromaDB."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chromadb")  # Changed from Client()
        try:
            client.delete_collection("features")
            print("🗑️ Cleared existing embeddings")
        except ValueError:
            pass  # Collection doesn't exist
        return True
    except Exception as e:
        print(f"❌ Error clearing ChromaDB: {e}")
        return False

def ask_user_phase_choice(has_embeddings, existing_datasets):
    """Ask user what they want to do with interactive options."""
    print(f"\n{'='*60}")
    print(f"ANALYSIS WORKFLOW SELECTION")
    print(f"{'='*60}")
    
    if has_embeddings:
        print("✅ Found existing feature embeddings!")
        print(f"📊 You have embeddings for {len(existing_datasets)} datasets with {sum(existing_datasets.values())} total features")
        print("\nWhat would you like to do?")
        print("  1. 🔄 Add More Features (Run feature extraction + graph analysis)")
        print("  2. 🗑️  Start Fresh (Clear existing + extract features + graph analysis)")
        print("  3. ⚡ Skip to Graph Analysis (Use existing embeddings only)")
        print("  4. 🚪 Exit")
        
        while True:
            choice = input(f"\nChoose option (1/2/3/4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return choice
            print("Please enter 1, 2, 3, or 4")
    else:
        print("📝 No existing embeddings found.")
        print("\nWhat would you like to do?")
        print("  1. 🚀 Start Feature Extraction (Extract features + graph analysis)")
        print("  2. 🚪 Exit")
        
        while True:
            choice = input(f"\nChoose option (1/2): ").strip()
            if choice in ['1', '2']:
                return choice
            print("Please enter 1 or 2")


def clear_chromadb():
    """Clear all existing embeddings from ChromaDB."""
    try:
        import chromadb
        client = chromadb.Client()
        try:
            client.delete_collection("features")
            print("🗑️ Cleared existing embeddings")
        except ValueError:
            pass  # Collection doesn't exist
        return True
    except Exception as e:
        print(f"❌ Error clearing ChromaDB: {e}")
        return False


def run_phase1(folder_path):
    """Run Phase 1: Feature Analysis & Embedding with real-time status tracking"""
    print("\n--- Phase 1: Feature Analysis & Embedding ---")
    
    # Show initial status
    print(f"\n📊 Initial Vector DB Status:")
    initial_status = print_embedding_status()
    processed_files = 0
    total_embeddings = 0
    total_files = 0
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            total_files += 1
            print(f"\n🔄 Processing {filename}...")
            try:
                # This now returns the number of successfully embedded features
                embedded_count = analyze_and_embed_features(file_path)
                if embedded_count is not None and embedded_count > 0:
                    processed_files += 1
                    total_embeddings += embedded_count
                    
                    # Show running total after each dataset
                    current_status = get_embedding_progress_summary()
                    print(f"🏃‍♂️ Running Progress: {current_status}")
                    
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                continue

    print(f"\n📊 Phase 1 Final Summary:")
    final_status = print_embedding_status()
    print(f"  - Total files found: {total_files}")
    print(f"  - Successfully processed: {processed_files}")
    print(f"  - Total features embedded this session: {total_embeddings}")
    print(f"  - Skipped/Failed: {total_files - processed_files}")
    
    # Show net change
    initial_count = initial_status.get('total_embeddings', 0) if isinstance(initial_status, dict) else 0
    final_count = final_status.get('total_embeddings', 0) if isinstance(final_status, dict) else 0
    net_change = final_count - initial_count
    print(f"  - Net embeddings added: {net_change}")
    
    return processed_files > 0


def run_phase2(non_interactive=False):
    """Run Phase 2: Building Dataset Relationship Map"""
    print("\n--- Phase 2: Building Dataset Relationship Map ---")
    
    # Check vector DB status before graph building
    print(f"\n📊 Vector DB Status for Graph Building:")
    db_status = print_embedding_status()
    
    if db_status.get('total_embeddings', 0) == 0:
        print("❌ Cannot build graph: No embeddings found in vector DB")
        return None
    
    # Debug ChromaDB contents first
    debug_chromadb()
    
    # Ask user for similarity threshold
    similarity_threshold = 0.75
    if not non_interactive:
        print(f"\n🎛️ Graph Building Configuration:")
        threshold_input = input("Enter similarity threshold (0.0-1.0, default 0.75): ").strip()
        
        try:
            if threshold_input:
                similarity_threshold = float(threshold_input)
                similarity_threshold = max(0.0, min(1.0, similarity_threshold))  # Clamp between 0 and 1
        except ValueError:
            similarity_threshold = 0.75
            print(f"Invalid input, using default threshold: {similarity_threshold}")
    
    print(f"🔧 Using similarity threshold: {similarity_threshold}")
    
    try:
        dataset_graph = build_graph(similarity_threshold=similarity_threshold)
        
        if dataset_graph.number_of_nodes() == 0:
            print("⚠️  No graph was built. This could mean:")
            print("   1. No embeddings available in ChromaDB")
            print("   2. Similarity threshold is too high")
            print("   3. Features are too dissimilar to connect")
            
            if not non_interactive:
                # Offer to try lower threshold
                retry = input(f"\n🔄 Try with lower threshold (0.5)? (y/n): ").strip().lower()
                if retry == 'y':
                    print("🔄 Retrying with threshold 0.5...")
                    dataset_graph = build_graph(similarity_threshold=0.5)
            
            if dataset_graph.number_of_nodes() == 0:
                print("❌ Still no graph could be built. Check your embeddings.")
                return None
        
        return dataset_graph
        
    except Exception as e:
        print(f"❌ Error in Phase 2: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_phase3(dataset_graph):
    """Run Phase 3: Analysis & Report Generation"""
    print("\n--- Phase 3: Dataset Analysis & Report Generation ---")
    
    try:
        # Analyze relationships
        if dataset_graph and dataset_graph.number_of_nodes() > 0:
            relationship_analysis = analyze_graph_relationships(dataset_graph)
            
            print(f"🔗 Analysis Results:")
            print(f"  - Total datasets: {relationship_analysis.get('total_datasets', 0)}")
            print(f"  - Dataset connections: {relationship_analysis.get('total_connections', 0)}")
            print(f"  - Connected components: {relationship_analysis.get('connected_components', 0)}")
            print(f"  - Graph density: {relationship_analysis.get('graph_density', 0):.4f}")
            
            if relationship_analysis.get('strongest_connections'):
                print("\n🏆 Strongest dataset connections:")
                for conn in relationship_analysis['strongest_connections'][:5]:
                    print(f"   • {conn['dataset1']} ↔ {conn['dataset2']}: {conn['similarity']:.3f} ({conn['feature_connections']} features)")
            
            if relationship_analysis.get('most_connected_datasets'):
                print("\n📊 Most connected datasets:")
                for dataset, centrality in relationship_analysis['most_connected_datasets'][:5]:
                    print(f"   • {dataset}: {centrality:.3f} connectivity")
        
        # Generate report
        report = generate_report(dataset_graph)
        print("\n--- Data Potential Report ---")
        print(report)

        # Save reports
        report_filename = "data_potential_report.json"
        with open(report_filename, "w") as f:
            f.write(report)
        print(f"\n💾 Main report saved to: {report_filename}")
        
        # Save relationship analysis if we have a valid graph
        if dataset_graph and dataset_graph.number_of_nodes() > 0:
            relationships_filename = "dataset_relationships.json"
            with open(relationships_filename, "w") as f:
                json.dump(relationship_analysis, f, indent=2, default=str)
            print(f"💾 Relationship analysis saved to: {relationships_filename}")
            
    except Exception as e:
        print(f"❌ Error generating reports: {e}")


def run_discover_mode(folder_path, skip_phase1=False, non_interactive=False):
    """
    Runs the discover mode with optional phase skipping and real-time status tracking.
    FIXED: Always proceed to Phase 2 if embeddings exist.
    """
    print("--- Running Discover Mode: Vector-based Analysis ---")
    
    if non_interactive:
        # CLI mode - use skip_phase1 flag directly
        if skip_phase1:
            print("\n⚡ Skipping Phase 1 - Using existing embeddings (CLI mode)")
        else:
            print("\n🚀 Running full pipeline (CLI mode)")
            success = run_phase1(folder_path)
        
        # ALWAYS proceed to Phase 2 regardless of Phase 1 result
        dataset_graph = run_phase2(non_interactive=True)
        if dataset_graph:
            run_phase3(dataset_graph)
    else:
        # Interactive mode
        has_embeddings, existing_datasets = check_existing_embeddings()
        user_choice = ask_user_phase_choice(has_embeddings, existing_datasets)
        
        if has_embeddings:
            if user_choice == '1':  # Add more embeddings
                print("\n🔄 Adding more features to existing embeddings...")
                success = run_phase1(folder_path)
                dataset_graph = run_phase2()
                if dataset_graph:
                    run_phase3(dataset_graph)
            elif user_choice == '2':  # Replace all embeddings
                print("\n🗑️ Clearing existing embeddings and starting fresh...")
                if clear_chromadb():
                    success = run_phase1(folder_path)
                    dataset_graph = run_phase2()
                    if dataset_graph:
                        run_phase3(dataset_graph)
            elif user_choice == '3':  # Skip to analysis
                print("\n⚡ Skipping to graph analysis using existing embeddings...")
                dataset_graph = run_phase2()
                if dataset_graph:
                    run_phase3(dataset_graph)
            elif user_choice == '4':  # Exit
                print("👋 Goodbye!")
        else:
            if user_choice == '1':  # Start fresh
                print("\n🚀 Starting feature extraction and analysis...")
                success = run_phase1(folder_path)
                dataset_graph = run_phase2()
                if dataset_graph:
                    run_phase3(dataset_graph)
            elif user_choice == '2':  # Exit
                print("👋 Goodbye!")


def run_harmonize_mode(folder_path, output_json_path):
    """Runs the original analysis pipeline using LLM-based synthesis."""
    print("--- Running Harmonize Mode: LLM-based Synthesis ---")
    all_analyses = {}
    print("\n--- Phase 1: Individual Dataset Analysis ---")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            print(f"Analyzing {filename}...")
            df_sample = read_dataset_sample(file_path)
            if df_sample is not None:
                analysis, _ = analyze_individual_dataset(file_path, df_sample, ["google"], None)
                if analysis:
                    all_analyses[filename] = analysis

    if not all_analyses:
        print("No datasets were successfully analyzed.")
        return

    print("\n--- Phase 2: Cross-Dataset Synthesis ---")
    harmonization_map = synthesize_analyses(all_analyses, "", ["google"])

    if isinstance(harmonization_map, str):
        print(f"Error during synthesis: {harmonization_map}")
        return

    print("\n--- Harmonization Map (JSON) ---")
    print(json.dumps(harmonization_map, indent=2))

    if output_json_path:
        with open(output_json_path, 'w') as f:
            json.dump(harmonization_map, f, indent=2)
        print(f"\nHarmonization map saved to: {output_json_path}")


def main():
    """Main function to run the dataset analysis pipeline."""
    parser = argparse.ArgumentParser(description="Analyze datasets to find relationships.")
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='command', help='Analysis mode')
    
    # Discover mode subparser
    discover_parser = subparsers.add_parser('discover', help='Vector-based analysis with graph relationships')
    discover_parser.add_argument('folder_path', type=str, help='Path to folder containing datasets')
    discover_parser.add_argument('--skip-phase1', action='store_true', 
                                help='Skip feature extraction and use existing embeddings')
    discover_parser.add_argument('--non-interactive', action='store_true',
                                help='Run in non-interactive mode using CLI flags only')
    
    # Harmonize mode subparser  
    harmonize_parser = subparsers.add_parser('harmonize', help='LLM-based synthesis analysis')
    harmonize_parser.add_argument('folder_path', type=str, help='Path to folder containing datasets')
    harmonize_parser.add_argument('--output-json', type=str, default="harmonization_map.json",
                                 help='Path to save the harmonization map')
    
    # Status check subparser
    status_parser = subparsers.add_parser('status', help='Check current vector DB status')

    # Handle case where no subcommand is provided
    args = parser.parse_args()
    
    if not hasattr(args, 'command') or args.command is None:
        print("No command specified. Use 'discover', 'harmonize', or 'status'. Run with --help for more info.")
        parser.print_help()
        return

    if args.command == 'status':
        print("🔍 Checking Vector DB Status...")
        status = print_embedding_status()
        if status.get('total_embeddings', 0) > 0:
            print(f"\n✅ Vector DB is ready for graph analysis!")
        else:
            print(f"\n📝 Vector DB is empty. Run feature extraction first.")
        return

    if not os.path.isdir(args.folder_path):
        print(f"Error: The path '{args.folder_path}' is not a valid directory.")
        return

    if args.command == 'discover':
        run_discover_mode(args.folder_path, 
                         skip_phase1=args.skip_phase1, 
                         non_interactive=args.non_interactive)
    elif args.command == 'harmonize':
        run_harmonize_mode(args.folder_path, args.output_json)


if __name__ == "__main__":
    main()
