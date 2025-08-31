import chromadb
import polars as pl
from tools.pubmed_embedding import PubMedEmbedding
from tools.llm_manager import get_llm_response
from tools.utils import read_dataset_sample
import json
import time


def get_dataset_info(file_path, df_sample):
    """
    Get basic dataset information for user decision making.
    """
    info = {
        "file_name": file_path.split("\\")[-1] if "\\" in file_path else file_path.split("/")[-1],
        "dimensions": f"{df_sample.shape[0]} rows × {df_sample.shape[1]} columns",
        "total_columns": df_sample.shape[1],
        "memory_usage": f"{df_sample.estimated_size('mb'):.2f} MB" if hasattr(df_sample, 'estimated_size') else "Unknown",
    }
    
    # Calculate NaN percentages per column
    nan_stats = []
    for col in df_sample.columns[:5]:  # Show first 5 columns as sample
        if col.strip():  # Skip empty column names
            nan_pct = (df_sample[col].null_count() / len(df_sample)) * 100
            nan_stats.append(f"{col}: {nan_pct:.1f}%")
    
    info["sample_nan_percentages"] = nan_stats
    return info


def ask_user_to_process_dataset(file_path, df_sample):
    """
    Ask user whether to process this dataset with LLM analysis.
    """
    info = get_dataset_info(file_path, df_sample)
    
    print(f"\n{'='*60}")
    print(f"DATASET ANALYSIS DECISION")
    print(f"{'='*60}")
    print(f"📁 File: {info['file_name']}")
    print(f"📊 Dimensions: {info['dimensions']}")
    print(f"💾 Estimated Size: {info['memory_usage']}")
    print(f"🔍 Sample NaN Values:")
    for nan_stat in info['sample_nan_percentages']:
        print(f"   • {nan_stat}")
    
    if info['total_columns'] > 50:
        print(f"⚠️  Large dataset with {info['total_columns']} columns - LLM processing may take time and hit rate limits")
        print(f"💡 Consider using Cerebras or Groq first to avoid rate limits")
    
    print(f"\nOptions:")
    print(f"  1. 🚀 Process with LLM Analysis (semantic descriptions)")
    print(f"  2. ⏭️  Skip this dataset")
    
    while True:
        choice = input(f"\nChoose option (1/2): ").strip()
        if choice in ['1', '2']:
            return choice
        print("Please enter 1 or 2")


def get_feature_metadata(series):
    """
    Analyzes a polars Series to extract a rich set of metadata and determine its type.
    """
    dtype = series.dtype
    non_null_series = series.drop_nulls()
    num_unique = non_null_series.n_unique()
    total_values = len(series)
    nan_percentage = series.null_count() / total_values * 100
    uniqueness_ratio = num_unique / total_values if total_values > 0 else 0

    metadata = {
        "dtype": str(dtype),
        "nan_percentage": round(nan_percentage, 2),
        "num_unique": num_unique,
        "uniqueness_ratio": round(uniqueness_ratio, 2),
        "sample_values": non_null_series.head(5).to_list()
    }

    # 1. Datetime Type
    if dtype.is_temporal():
        metadata["feature_type"] = "datetime"
        stats = {
            "min": str(non_null_series.min()),
            "max": str(non_null_series.max()),
        }
        metadata["descriptive_statistics"] = stats
        return metadata

    # 2. Boolean Type
    if dtype == pl.Boolean:
        metadata["feature_type"] = "boolean"
        # Convert value_counts result to serializable dict
        value_counts = non_null_series.value_counts()
        stats_dict = {}
        for i in range(len(value_counts)):
            key = str(value_counts[i, 0])  # First column (value)
            val = int(value_counts[i, 1])  # Second column (count)
            stats_dict[key] = val
        metadata["descriptive_statistics"] = stats_dict
        return metadata

    # 3. Numeric Types (Int, Float)
    if dtype.is_numeric():
        # High Uniqueness ID
        if uniqueness_ratio > 0.95:
            metadata["feature_type"] = "id"
        # Categorical Integer
        elif num_unique < 20 and dtype.is_integer():
            metadata["feature_type"] = "categorical_int"
            # Convert value_counts result to serializable dict
            value_counts = non_null_series.value_counts()
            stats_dict = {}
            for i in range(len(value_counts)):
                key = str(value_counts[i, 0])  # First column (value)
                val = int(value_counts[i, 1])  # Second column (count)
                stats_dict[key] = val
            metadata["descriptive_statistics"] = stats_dict
        # Continuous Float
        elif dtype.is_float():
            metadata["feature_type"] = "continuous"
            # Convert describe result to serializable dict
            describe_result = non_null_series.describe()
            stats_dict = {}
            for i in range(len(describe_result)):
                key = str(describe_result[i, 0])  # Statistic name
                val = float(describe_result[i, 1]) if describe_result[i, 1] is not None else None  # Value
                stats_dict[key] = val
            metadata["descriptive_statistics"] = stats_dict
        # Discrete Integer
        else:
            metadata["feature_type"] = "discrete"
            # Convert describe result to serializable dict
            describe_result = non_null_series.describe()
            stats_dict = {}
            for i in range(len(describe_result)):
                key = str(describe_result[i, 0])  # Statistic name
                val = float(describe_result[i, 1]) if describe_result[i, 1] is not None else None  # Value
                stats_dict[key] = val
            metadata["descriptive_statistics"] = stats_dict
        return metadata

    # 4. String Type
    if dtype == pl.Utf8 or dtype == pl.String:
        # Attempt to convert to datetime
        try:
            datetime_series = non_null_series.str.to_datetime(errors='ignore')
            if not datetime_series.is_null().all():
                metadata["feature_type"] = "datetime_str"
                stats = {
                    "min": str(datetime_series.min()),
                    "max": str(datetime_series.max()),
                }
                metadata["descriptive_statistics"] = stats
                return metadata
        except (ValueError, TypeError):
            pass

        # High Uniqueness ID or Text
        if uniqueness_ratio > 0.9:
            # FIXED: Use len_chars() instead of lengths()
            avg_len = non_null_series.str.len_chars().mean()
            if avg_len > 50:  # Arbitrary threshold for long text
                metadata["feature_type"] = "text"
            else:
                metadata["feature_type"] = "id_str"
        # Categorical String
        else:
            metadata["feature_type"] = "categorical_str"
            # Convert value_counts result to serializable dict
            value_counts = non_null_series.value_counts().head(10)
            stats_dict = {}
            for i in range(len(value_counts)):
                key = str(value_counts[i, 0])  # First column (value)
                val = int(value_counts[i, 1])  # Second column (count)
                stats_dict[key] = val
            metadata["descriptive_statistics"] = stats_dict
        return metadata

    metadata["feature_type"] = "unknown"
    return metadata


def analyze_and_embed_features(file_path, sample_size=100):
    """
    Analyzes the features of a dataset, gets their semantic meaning, and stores them as embeddings.
    """
    df_sample = read_dataset_sample(file_path, sample_size=sample_size)
    if df_sample is None:
        return None

    # Ask user whether to process this dataset
    user_choice = ask_user_to_process_dataset(file_path, df_sample)
    
    if user_choice == '2':  # Skip dataset
        print(f"⏭️  Skipping {file_path}")
        return None
    
    client = chromadb.PersistentClient(path="./chromadb")
    collection = client.get_or_create_collection("features")
    model = PubMedEmbedding()

    print(f"\n🔄 Processing {df_sample.shape[1]} columns with LLM semantic analysis...")
    print(f"🤖 Using multi-provider LLM fallback: Cerebras → Groq → NVIDIA → Google")

    processed_count = 0
    successful_embeddings = 0
    failed_features = []

    for column in df_sample.columns:
        try:
            # Handle empty column names
            if not column or column.strip() == "":
                print(f"  - Skipping empty column name")
                continue
            
            processed_count += 1
            metadata = get_feature_metadata(df_sample[column])
            
            # Rate limiting: small delay between LLM requests
            if processed_count > 1:  # Skip delay for first request
                time.sleep(0.5)  # 500ms delay to avoid rate limits
            
            # Create detailed prompt for LLM
            template = f"""You are an expert data analyst. Based on the following metadata, provide a concise, one-sentence semantic meaning for the feature.

Feature Name: {column}
Inferred Type: {metadata.get('feature_type', 'unknown')}
Data Type: {metadata.get('dtype')}
Missing Values: {metadata.get('nan_percentage')}%
Uniqueness Ratio: {metadata.get('uniqueness_ratio')}
Sample Values: {str(metadata.get('sample_values', [])[:3])}

Provide a single sentence describing what this feature represents:"""
            
            # Try to get LLM response with fallback providers
            semantic_meaning, provider = get_llm_response(template, {}, ["cerebras", "groq", "nvidia", "nvidia_nemotron", "google"])
            
            if semantic_meaning:
                # Successfully got LLM response
                embedding_text = f"Feature: {column}, Type: {metadata.get('feature_type')}, Meaning: {semantic_meaning.strip()}"
                embedding = model.embed_documents([embedding_text])[0]

                # Add all metadata to ChromaDB, ensuring it's serializable
                chroma_metadata = metadata.copy()
                chroma_metadata["dataset_id"] = file_path
                chroma_metadata["column_name"] = column
                chroma_metadata["semantic_meaning"] = semantic_meaning.strip()
                chroma_metadata["llm_provider"] = provider
                
                # Ensure all values are JSON serializable
                for key, value in chroma_metadata.items():
                    if isinstance(value, (dict, list)):
                        chroma_metadata[key] = json.dumps(value)
                    elif isinstance(value, pl.Series):  # Handle any remaining Series objects
                        chroma_metadata[key] = json.dumps(value.to_list())
                    else:
                        chroma_metadata[key] = str(value)

                collection.add(
                    ids=[f"{file_path}::{column}"],
                    embeddings=[embedding],
                    metadatas=[chroma_metadata]
                )
                
                successful_embeddings += 1
                print(f"  ✅ Embedded feature: {column} (Type: {metadata.get('feature_type')}) via {provider}")
            else:
                # Failed to get LLM response from all providers
                failed_features.append(column)
                print(f"  ❌ Failed to get semantic meaning for feature: {column} (all LLM providers failed)")
                
        except Exception as e:
            failed_features.append(column)
            print(f"  ❌ Error processing feature {column}: {str(e)}")
            continue  # Continue with next column even if this one fails

    # Summary
    print(f"\n{'='*60}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully embedded: {successful_embeddings} features")
    print(f"❌ Failed features: {len(failed_features)}")
    if failed_features:
        print(f"🔍 Failed feature names: {', '.join(failed_features[:5])}" + ("..." if len(failed_features) > 5 else ""))
    print(f"📈 Success rate: {(successful_embeddings / processed_count * 100):.1f}%" if processed_count > 0 else "📈 Success rate: 0%")
    print(f"✅ Finished processing {file_path}")
