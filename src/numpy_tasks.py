import numpy as np

def q1_load_and_clean_data(filepath):
    """Q1: Load numerical features (age, avg_glucose_level, bmi) & Handle NaN."""
    print("--- Q1: Data Loading & Cleaning ---")

    data = np.genfromtxt(filepath, delimiter=',', skip_header=1, usecols=(2, 8, 9), missing_values='N/A', filling_values=np.nan)
    
    col_mean = np.nanmean(data[:, 2])
    data[:, 2] = np.where(np.isnan(data[:, 2]), col_mean, data[:, 2])
    
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}\n")
    return data

def q2_compute_statistics(data):
    """Q2: Mean, Median, and Standard Deviation."""
    print("--- Q2: Basic Statistics ---")
    features = ['Age', 'Avg_Glucose_Level', 'BMI']
    for i, feature in enumerate(features):
        print(f"{feature} -> Mean: {np.mean(data[:, i]):.2f}, Median: {np.median(data[:, i]):.2f}, Std: {np.std(data[:, i]):.2f}")
    print()

def q3_min_max_normalization(data):
    """Q3: Min-Max normalization using vectorized operations."""
    print("--- Q3: Min-Max Normalization ---")
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized = (data - min_vals) / (max_vals - min_vals)
    print("Normalized Data (first 2 rows):\n", normalized[:2], "\n")
    return normalized

def q4_boolean_filtering(data):
    """Q4: Filter patients age > 50 and avg_glucose_level > average."""
    print("--- Q4: Boolean Indexing ---")
    avg_glucose = np.mean(data[:, 1])
    mask = (data[:, 0] > 50) & (data[:, 1] > avg_glucose)
    filtered = data[mask]
    print(f"Total patients matched: {filtered.shape[0]}\n")

def q5_correlation_matrix(data):
    """Q5: Correlation matrix between numerical features."""
    print("--- Q5: Correlation Matrix ---")
    corr_matrix = np.corrcoef(data, rowvar=False)
    print(corr_matrix, "\n")

def q6_pairwise_euclidean(data):
    """Q6: Pairwise Euclidean distance (Vectorized)."""
    print("--- Q6: Pairwise Euclidean Distance ---")
    subset = data[:5]
    diff = subset[:, np.newaxis, :] - subset[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))
    print("Distances for first 5 patients:\n", distances, "\n")

def q7_normalization_from_scratch(data):
    """Q7: Min-Max and Z-score normalization from scratch."""
    print("--- Q7: Min-Max & Z-score from scratch ---")
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    z_score_data = (data - means) / stds
    print("Z-score Normalized (first 2 rows):\n", z_score_data[:2], "\n")

def q8_cosine_similarity(data):
    """Q8: Cosine similarity between two patient vectors."""
    print("--- Q8: Cosine Similarity ---")
    vec1, vec2 = data[0], data[1]
    dot_product = np.dot(vec1, vec2)
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    similarity = dot_product / (norm1 * norm2)
    print(f"Cosine Similarity (Patient 1 & 2): {similarity:.4f}\n")

def q9_manual_pca(data):
    """Q9: Manual PCA onto top 2 principal components."""
    print("--- Q9: Manual PCA ---")
    z_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    cov_matrix = np.cov(z_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_2_eigenvectors = eigenvectors[:, sorted_indices[:2]]
    projected_data = np.dot(z_data, top_2_eigenvectors)
    print("Projected Data Shape:", projected_data.shape)
    print("First 2 projected records:\n", projected_data[:2], "\n")
    return projected_data

def q10_batch_processing(data):
    """Q10: Split array into equal-sized batches and compute mean."""
    print("--- Q10: Batch Processing ---")
    batch_size = 1000
    num_batches = len(data) // batch_size
    truncated_data = data[:num_batches * batch_size]
    
    batches = np.split(truncated_data, num_batches)
    for i, batch in enumerate(batches[:3]): 
        print(f"Batch {i+1} Mean Vector: {np.mean(batch, axis=0)}")
    print("...\n")

if __name__ == "__main__":
    filepath = 'data/raw/healthcare-dataset-stroke-data.csv'
    dataset = q1_load_and_clean_data(filepath)
    q2_compute_statistics(dataset)
    q3_min_max_normalization(dataset)
    q4_boolean_filtering(dataset)
    q5_correlation_matrix(dataset)
    q6_pairwise_euclidean(dataset)
    q7_normalization_from_scratch(dataset)
    q8_cosine_similarity(dataset)
    q9_manual_pca(dataset)
    q10_batch_processing(dataset)