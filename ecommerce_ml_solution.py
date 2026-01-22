import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st

class EcommerceMLSolution:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.processed_df = None
        self.rfm_data = None
        self.customer_segments = None
        self.product_recommendation_model = None
        
    def load_and_clean_data(self):
        """Load and clean the dataset"""
        print("Loading and cleaning data...")
        
        # Make a copy of the dataframe
        df = self.df.copy()
        
        # Initial shape
        initial_shape = df.shape
        print(f"Initial dataset shape: {initial_shape}")
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        print(f"After removing duplicates: {df.shape}")
        
        # Convert InvoiceDate to datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Remove rows with missing CustomerID
        df = df[df['CustomerID'].notna()]
        print(f"After removing rows with missing CustomerID: {df.shape}")
        
        # Remove cancelled invoices (those that start with 'C')
        df = df[~df['InvoiceNo'].str.startswith('C', na=False)]
        print(f"After removing cancelled invoices: {df.shape}")
        
        # Remove rows with negative or zero Quantity and UnitPrice
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
        print(f"After removing invalid quantities/prices: {df.shape}")
        
        # Calculate total amount spent per transaction
        df['Amount'] = df['Quantity'] * df['UnitPrice']
        
        print(f"Final cleaned dataset shape: {df.shape}")
        print(f"Data cleaning removed {(initial_shape[0] - df.shape[0])} rows ({((initial_shape[0] - df.shape[0])/initial_shape[0]*100):.2f}%)")
        
        self.processed_df = df
        return df
    
    def perform_eda(self):
        """Perform exploratory data analysis"""
        print("\nPerforming Exploratory Data Analysis...")
        
        df = self.processed_df
        
        # Basic statistics
        print("\nDataset Info:")
        print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
        print(f"Number of unique customers: {df['CustomerID'].nunique()}")
        print(f"Number of unique products: {df['StockCode'].nunique()}")
        print(f"Number of unique countries: {df['Country'].nunique()}")
        print(f"Total revenue: £{df['Amount'].sum():,.2f}")
        
        # Top selling products
        top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
        print(f"\nTop 10 selling products:")
        print(top_products)
        
        # Sales by country
        sales_by_country = df.groupby('Country')['Amount'].sum().sort_values(ascending=False).head(10)
        print(f"\nTop 10 countries by sales:")
        print(sales_by_country)
        
        return {
            'date_range': (df['InvoiceDate'].min(), df['InvoiceDate'].max()),
            'num_customers': df['CustomerID'].nunique(),
            'num_products': df['StockCode'].nunique(),
            'num_countries': df['Country'].nunique(),
            'total_revenue': df['Amount'].sum(),
            'top_products': top_products,
            'sales_by_country': sales_by_country
        }
    
    def calculate_rfm_metrics(self):
        """Calculate Recency, Frequency, Monetary metrics for each customer"""
        print("\nCalculating RFM metrics...")
        
        df = self.processed_df
        
        # Define the reference date for recency calculation
        reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
        
        # Calculate RFM metrics
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
            'InvoiceNo': 'count',  # Frequency
            'Amount': 'sum'  # Monetary
        }).round(2)
        
        # Rename columns
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        # Handle any missing values (though unlikely after cleaning)
        rfm = rfm.fillna(0)
        
        print(f"RFM metrics calculated for {len(rfm)} customers")
        print(f"RFM metrics summary:")
        print(rfm.describe())
        
        self.rfm_data = rfm
        return rfm
    
    def determine_optimal_clusters(self, max_k=10):
        """Determine optimal number of clusters using elbow method and silhouette score"""
        print(f"\nDetermining optimal number of clusters (k={max_k})...")
        
        rfm_scaled = StandardScaler().fit_transform(self.rfm_data)
        
        # Calculate inertia (within-cluster sum of squares) and silhouette scores for different k values
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(rfm_scaled)
            
            inertias.append(kmeans.inertia_)
            silhouette_avg = silhouette_score(rfm_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Plot elbow curve and silhouette scores
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        
        ax[0].plot(k_range, inertias, marker='o')
        ax[0].set_title('Elbow Method for Optimal k')
        ax[0].set_xlabel('Number of Clusters (k)')
        ax[0].set_ylabel('Inertia')
        
        ax[1].plot(k_range, silhouette_scores, marker='o', color='red')
        ax[1].set_title('Silhouette Score vs Number of Clusters')
        ax[1].set_xlabel('Number of Clusters (k)')
        ax[1].set_ylabel('Silhouette Score')
        
        plt.tight_layout()
        plt.show()
        
        # Determine optimal k based on highest silhouette score
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters based on silhouette score: {optimal_k}")
        
        return optimal_k, k_range, inertias, silhouette_scores
    
    def perform_customer_segmentation(self, n_clusters=4):
        """Perform customer segmentation using K-Means clustering"""
        print(f"\nPerforming customer segmentation with {n_clusters} clusters...")
        
        # Scale the RFM data
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(self.rfm_data)
        
        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(rfm_scaled)
        
        # Add cluster labels to RFM data
        rfm_clustered = self.rfm_data.copy()
        rfm_clustered['Cluster'] = cluster_labels
        
        # Calculate cluster statistics
        cluster_stats = rfm_clustered.groupby('Cluster').agg({
            'Recency': ['mean', 'std'],
            'Frequency': ['mean', 'std'], 
            'Monetary': ['mean', 'std']
        }).round(2)
        
        print("Cluster Statistics:")
        print(cluster_stats)
        
        # Assign meaningful segment names based on RFM characteristics
        segment_names = {}
        cluster_profiles = rfm_clustered.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
        
        for cluster_id in cluster_profiles.index:
            r, f, m = cluster_profiles.loc[cluster_id]
            
            # Determine segment based on RFM values
            if r < cluster_profiles['Recency'].median() and f > cluster_profiles['Frequency'].median() and m > cluster_profiles['Monetary'].median():
                segment_names[cluster_id] = 'High-Value'
            elif f > cluster_profiles['Frequency'].median() and m > cluster_profiles['Monetary'].median():
                segment_names[cluster_id] = 'Regular'
            elif f < cluster_profiles['Frequency'].median() and m < cluster_profiles['Monetary'].median():
                segment_names[cluster_id] = 'Occasional'
            elif r > cluster_profiles['Recency'].median() and f < cluster_profiles['Frequency'].median() and m < cluster_profiles['Monetary'].median():
                segment_names[cluster_id] = 'At-Risk'
            else:
                segment_names[cluster_id] = f'Cluster_{cluster_id}'
        
        # Map cluster IDs to segment names
        rfm_clustered['Segment'] = rfm_clustered['Cluster'].map(segment_names)
        
        print("\nSegment Distribution:")
        print(rfm_clustered['Segment'].value_counts())
        
        # Store the segmentation results
        self.customer_segments = rfm_clustered
        
        return rfm_clustered, scaler, kmeans
    
    def visualize_customer_segments(self):
        """Visualize customer segments"""
        print("\nCreating visualizations for customer segments...")
        
        rfm_segmented = self.customer_segments
        
        # Plot 1: 3D scatter plot of RFM
        fig = plt.figure(figsize=(15, 5))
        
        ax1 = fig.add_subplot(131, projection='3d')
        scatter = ax1.scatter(
            rfm_segmented['Recency'], 
            rfm_segmented['Frequency'], 
            rfm_segmented['Monetary'],
            c=rfm_segmented['Cluster'], 
            cmap='viridis'
        )
        ax1.set_xlabel('Recency')
        ax1.set_ylabel('Frequency')
        ax1.set_zlabel('Monetary')
        ax1.set_title('RFM Clusters (3D)')
        
        # Plot 2: Recency vs Frequency colored by segment
        ax2 = fig.add_subplot(132)
        sns.scatterplot(data=rfm_segmented, x='Recency', y='Frequency', hue='Segment', ax=ax2)
        ax2.set_title('Recency vs Frequency by Segment')
        
        # Plot 3: Recency vs Monetary colored by segment
        ax3 = fig.add_subplot(133)
        sns.scatterplot(data=rfm_segmented, x='Recency', y='Monetary', hue='Segment', ax=ax3)
        ax3.set_title('Recency vs Monetary by Segment')
        
        plt.tight_layout()
        plt.show()
        
        # Bar chart of segment distribution
        plt.figure(figsize=(10, 6))
        segment_counts = rfm_segmented['Segment'].value_counts()
        sns.barplot(x=segment_counts.index, y=segment_counts.values)
        plt.title('Customer Segment Distribution')
        plt.xlabel('Segment')
        plt.ylabel('Number of Customers')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def create_product_recommendation_system(self):
        """Create product recommendation system using collaborative filtering"""
        print("\nCreating product recommendation system...")
        
        df = self.processed_df
        
        # Create a pivot table: customers x products (quantity)
        customer_product_matrix = df.pivot_table(
            index='CustomerID',
            columns='StockCode',
            values='Quantity',
            aggfunc='sum',
            fill_value=0
        )
        
        print(f"Customer-product matrix shape: {customer_product_matrix.shape}")
        
        # Calculate product similarities using cosine similarity
        product_similarity = cosine_similarity(customer_product_matrix.T)
        product_similarity_df = pd.DataFrame(
            product_similarity,
            index=customer_product_matrix.columns,
            columns=customer_product_matrix.columns
        )
        
        # Store the recommendation model
        self.product_recommendation_model = {
            'customer_product_matrix': customer_product_matrix,
            'product_similarity': product_similarity_df
        }
        
        return product_similarity_df
    
    def recommend_products(self, product_code, top_n=5):
        """Recommend products similar to a given product"""
        if self.product_recommendation_model is None:
            raise ValueError("Recommendation model not created yet. Call create_product_recommendation_system() first.")
        
        similarity_df = self.product_recommendation_model['product_similarity']
        
        # Check if product exists in the model
        if product_code not in similarity_df.index:
            return f"Product '{product_code}' not found in the dataset."
        
        # Get similarity scores for the product
        product_similarities = similarity_df[product_code].sort_values(ascending=False)
        
        # Exclude the product itself and get top N similar products
        similar_products = product_similarities[1:top_n+1]  # Skip the first one (itself)
        
        # Get product descriptions
        df = self.processed_df
        product_descriptions = df[['StockCode', 'Description']].drop_duplicates().set_index('StockCode')
        
        recommendations = []
        for prod_code, similarity in similar_products.items():
            desc = product_descriptions.loc[prod_code, 'Description'] if prod_code in product_descriptions.index else 'Unknown'
            recommendations.append({
                'StockCode': prod_code,
                'Description': desc,
                'Similarity': round(similarity, 4)
            })
        
        return recommendations
    
    def predict_customer_segment(self, recency, frequency, monetary, scaler, kmeans, segment_mapping):
        """Predict segment for a new customer based on RFM values"""
        # Scale the input values
        scaled_input = scaler.transform([[recency, frequency, monetary]])
        
        # Predict cluster
        cluster_id = kmeans.predict(scaled_input)[0]
        
        # Map to segment name
        segment_name = segment_mapping.get(cluster_id, f'Cluster_{cluster_id}')
        
        return segment_name

def run_streamlit_app():
    """Run the Streamlit app"""
    st.title("🛍️ E-commerce ML Solution")
    st.markdown("""
    This application demonstrates two machine learning systems:
    - **Customer Segmentation**: Groups customers using RFM analysis and K-Means clustering
    - **Product Recommendations**: Recommends similar products using collaborative filtering
    """)
    
    # Initialize session state
    if 'solution' not in st.session_state:
        st.session_state.solution = None
        st.session_state.scaler = None
        st.session_state.kmeans = None
        st.session_state.segment_mapping = None
    
    # File upload
    uploaded_file = st.file_uploader("Upload your Online Retail CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Initialize the solution
        if st.session_state.solution is None:
            solution = EcommerceMLSolution(uploaded_file)
            st.session_state.solution = solution
            
            # Process data
            with st.spinner("Processing data..."):
                solution.load_and_clean_data()
                solution.calculate_rfm_metrics()
                
                # Perform clustering
                optimal_k, _, _, _ = solution.determine_optimal_clusters(max_k=6)
                rfm_clustered, scaler, kmeans = solution.perform_customer_segmentation(n_clusters=optimal_k)
                
                # Store models in session state
                st.session_state.scaler = scaler
                st.session_state.kmeans = kmeans
                
                # Create segment mapping
                segment_mapping = dict(zip(rfm_clustered['Cluster'], rfm_clustered['Segment']))
                st.session_state.segment_mapping = segment_mapping
                
                # Create recommendation system
                solution.create_product_recommendation_system()
            
            st.success("Data processing complete!")
        
        # Tabs for different functionalities
        tab1, tab2 = st.tabs(["🔍 Product Recommendations", "📊 Customer Segmentation"])
        
        with tab1:
            st.header("Product Recommendation System")
            st.markdown("Enter a product code to get similar product recommendations")
            
            # Get unique product codes for dropdown
            df = st.session_state.solution.processed_df
            product_options = df['StockCode'].unique()
            
            product_code = st.selectbox("Select Product Code", options=sorted(product_options))
            
            if st.button("Get Recommendations"):
                with st.spinner("Finding similar products..."):
                    try:
                        recommendations = st.session_state.solution.recommend_products(product_code, top_n=5)
                        
                        if isinstance(recommendations, str):
                            st.error(recommendations)
                        else:
                            st.subheader(f"Products similar to '{product_code}':")
                            
                            for i, rec in enumerate(recommendations, 1):
                                st.write(f"{i}. **{rec['StockCode']}** - {rec['Description']}")
                                st.write(f"   Similarity: {rec['Similarity']:.4f}")
                                st.write("---")
                    
                    except Exception as e:
                        st.error(f"Error getting recommendations: {str(e)}")
        
        with tab2:
            st.header("Customer Segmentation")
            st.markdown("Enter RFM values to predict customer segment")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                recency = st.number_input("Recency (Days since last purchase)", min_value=0, value=30)
            
            with col2:
                frequency = st.number_input("Frequency (Number of transactions)", min_value=0, value=5)
            
            with col3:
                monetary = st.number_input("Monetary (Total amount spent)", min_value=0.0, value=100.0)
            
            if st.button("Predict Segment"):
                if st.session_state.scaler is not None and st.session_state.kmeans is not None:
                    with st.spinner("Predicting customer segment..."):
                        segment = st.session_state.solution.predict_customer_segment(
                            recency, frequency, monetary,
                            st.session_state.scaler,
                            st.session_state.kmeans,
                            st.session_state.segment_mapping
                        )
                        
                        st.success(f"The predicted customer segment is: **{segment}**")
                        
                        # Show segment explanation
                        if 'High-Value' in segment:
                            st.info("💡 High-Value: Recently active customers with high purchase frequency and monetary value")
                        elif 'Regular' in segment:
                            st.info("💡 Regular: Consistent customers with moderate activity and spending")
                        elif 'Occasional' in segment:
                            st.info("💡 Occasional: Low activity customers with lower spending")
                        elif 'At-Risk' in segment:
                            st.info("⚠️ At-Risk: Inactive customers with low purchase history")
                        else:
                            st.info(f"Segment {segment} characteristics")
                else:
                    st.error("Models not initialized properly")
    
    else:
        st.info("Please upload an Online Retail CSV file to get started")

if __name__ == "__main__":
    # Check if running in Streamlit context
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--streamlit":
        run_streamlit_app()
    else:
        # Run the full analysis
        print("Running complete e-commerce ML solution...")
        
        # Initialize the solution
        solution = EcommerceMLSolution("online_retail.csv")
        
        # Step 1: Load and clean data
        cleaned_df = solution.load_and_clean_data()
        
        # Step 2: Perform EDA
        eda_results = solution.perform_eda()
        
        # Step 3: Calculate RFM metrics
        rfm_data = solution.calculate_rfm_metrics()
        
        # Step 4: Determine optimal clusters
        optimal_k, k_range, inertias, silhouette_scores = solution.determine_optimal_clusters()
        
        # Step 5: Perform customer segmentation
        rfm_clustered, scaler, kmeans = solution.perform_customer_segmentation(n_clusters=optimal_k)
        
        # Step 6: Visualize segments
        solution.visualize_customer_segments()
        
        # Step 7: Create product recommendation system
        similarity_matrix = solution.create_product_recommendation_system()
        
        # Step 8: Test product recommendations
        sample_product = solution.processed_df['StockCode'].iloc[0]
        recommendations = solution.recommend_products(sample_product)
        print(f"\nSample product recommendations for '{sample_product}':")
        for rec in recommendations:
            print(f"- {rec['Description']} (similarity: {rec['Similarity']:.4f})")
        
        print("\nAnalysis complete! The solution is ready for deployment in Streamlit.")
        print("\nTo run the Streamlit app, use: python ecommerce_ml_solution.py --streamlit")