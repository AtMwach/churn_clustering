import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Handle missing values
    # 'TotalCharges' may have empty strings or missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Drop customerID as it's not useful for clustering
    df = df.drop('customerID', axis=1)
    
    # Separate numerical and categorical columns
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = [col for col in df.columns if col not in numerical_cols]
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    # Scale numerical features
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df, numerical_cols, categorical_cols

# 2. Apply k-Means clustering
def apply_kmeans(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans

# 3. Visualize clusters using PCA
def visualize_clusters(data, clusters, numerical_cols):
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data)
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(principal_components[:, 0], 
                        principal_components[:, 1], 
                        c=clusters, 
                        cmap='viridis')
    
    plt.title('Customer Segments (PCA Visualization)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(scatter, label='Cluster')
    plt.savefig('customer_clusters.png')
    plt.close()
    
    # Explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance ratio: {explained_variance}")
    
    return principal_components

# 4. Analyze cluster characteristics
def analyze_clusters(data, clusters, numerical_cols, categorical_cols):
    data = data.copy()  # Avoid modifying the original DataFrame
    data['Cluster'] = clusters
    
    # Numerical features analysis
    numerical_summary = data.groupby('Cluster')[numerical_cols].mean()
    
    # Categorical features analysis
    # Compute mode for each categorical column per cluster
    categorical_summary = {}
    for col in categorical_cols:
        # Group by cluster and find the most frequent value (mode) for each column
        mode_series = data.groupby('Cluster')[col].agg(lambda x: x.mode().iloc[0] if not x.empty else np.nan)
        categorical_summary[col] = mode_series
    
    # Convert to DataFrame
    categorical_summary = pd.DataFrame(categorical_summary)
    
    return numerical_summary, categorical_summary

# Main execution
def main(file_path='WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    # Load and preprocess
    df_processed, numerical_cols, categorical_cols = load_and_preprocess_data(file_path)
    
    # Apply k-Means
    clusters, kmeans = apply_kmeans(df_processed)
    
    # Visualize
    principal_components = visualize_clusters(df_processed, clusters, numerical_cols)
    
    # Analyze clusters
    numerical_summary, categorical_summary = analyze_clusters(df_processed, clusters, numerical_cols, categorical_cols)
    
    # Save results
    numerical_summary.to_csv('cluster_numerical_summary.csv')
    categorical_summary.to_csv('cluster_categorical_summary.csv')
    
    # Print cluster characteristics
    print("\nNumerical Features by Cluster (Standardized Values):")
    print(numerical_summary)
    print("\nMost Common Categorical Values by Cluster:")
    print(categorical_summary)
    
    # Save preprocessed dataset
    df_processed.to_csv('preprocessed_churn_data.csv', index=False)
    
    print("\nPreprocessing Steps Completed:")
    print("- Handled missing values in TotalCharges")
    print("- Dropped customerID column")
    print("- Encoded categorical variables using LabelEncoder")
    print("- Standardized numerical features")
    print("- Saved preprocessed dataset to 'preprocessed_churn_data.csv'")
    print("- Generated cluster visualization as 'customer_clusters.png'")
    print("- Saved cluster summaries to CSV files")

if __name__ == "__main__":
    main()