import streamlit as st
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

# Enhanced page config with theme
st.set_page_config(
    page_title="E-commerce ML Solution By Piyush",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/piyush-eon/ecommerce-ml-solution',
        'Report a bug': "https://github.com/piyush-eon/ecommerce-ml-solution/issues",
        'About': "# E-commerce ML Solution v1.0"
    }
)

# Custom CSS for futuristic look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&family=Exo+2:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Exo 2', sans-serif;
    }
    
    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 2.5rem;
        background: linear-gradient(45deg, #00dbde, #fc00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-family: 'Orbitron', monospace;
        font-size: 1.8rem;
        color: #00dbde;
        border-bottom: 2px solid #fc00ff;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a2a6c, #b21f1f, #1a2a6c);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .segment-card {
        background: linear-gradient(135deg, #6a11cb, #2575fc);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin: 0.5rem 0;
    }
    
    .recommendation-item {
        background: linear-gradient(135deg, #141E30, #243B55);
        border-radius: 8px;
        padding: 0.8rem;
        color: white;
        margin: 0.5rem 0;
        border-left: 4px solid #00dbde;
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #00dbde, #fc00ff);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0, 219, 222, 0.4);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #2C3E50, #1A237E);
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% {transform: scale(1);}
        50% {transform: scale(1.05);}
        100% {transform: scale(1);}
    }
</style>
""", unsafe_allow_html=True)

# Header with futuristic styling
st.markdown('<div class="main-title">🛍️ E-commerce Machine Learning Solution By Piyush</div>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
This application demonstrates two advanced machine learning systems:

- **Customer Segmentation**: Groups customers using RFM analysis and K-Means clustering
- **Product Recommendations**: Recommends similar products using collaborative filtering
</div>
""", unsafe_allow_html=True)

# Add sidebar with additional controls
with st.sidebar:
    st.markdown('<h2 style="color: #00dbde;">⚙️ Controls</h2>', unsafe_allow_html=True)
    
    # Add futuristic-themed inputs
    st.markdown('<div style="background: rgba(0, 219, 222, 0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;">', unsafe_allow_html=True)
    show_visualizations = st.checkbox("Show Detailed Visualizations", value=True)
    cluster_count = st.slider("Number of Clusters", min_value=2, max_value=8, value=4)
    recommendations_count = st.slider("Recommendations Count", min_value=3, max_value=10, value=5)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Stats panel
    st.markdown('<h3 style="color: #fc00ff;">📊 Live Stats</h3>', unsafe_allow_html=True)
    if 'solution' in st.session_state and st.session_state.processing_complete:
        df = st.session_state.solution.processed_df
        st.metric("Customers Analyzed", df['CustomerID'].nunique())
        st.metric("Products Available", df['StockCode'].nunique())
        st.metric("Total Transactions", len(df))
    else:
        st.info("Upload data to see stats")

class EcommerceMLSolution:
    def __init__(self, df):
        self.df = df
        self.processed_df = None
        self.rfm_data = None
        self.customer_segments = None
        self.product_recommendation_model = None
        
    def load_and_clean_data(self):
        """Load and clean the dataset"""
        st.info("Loading and cleaning data...")
        
        # Make a copy of the dataframe
        df = self.df.copy()
        
        # Initial shape
        initial_shape = df.shape
        st.write(f"Initial dataset shape: {initial_shape}")
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        st.write(f"After removing duplicates: {df.shape}")
        
        # Convert InvoiceDate to datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Remove rows with missing CustomerID
        df = df[df['CustomerID'].notna()]
        st.write(f"After removing rows with missing CustomerID: {df.shape}")
        
        # Remove cancelled invoices (those that start with 'C')
        df = df[~df['InvoiceNo'].str.startswith('C', na=False)]
        st.write(f"After removing cancelled invoices: {df.shape}")
        
        # Remove rows with negative or zero Quantity and UnitPrice
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
        st.write(f"After removing invalid quantities/prices: {df.shape}")
        
        # Calculate total amount spent per transaction
        df['Amount'] = df['Quantity'] * df['UnitPrice']
        
        st.write(f"Final cleaned dataset shape: {df.shape}")
        st.write(f"Data cleaning removed {(initial_shape[0] - df.shape[0])} rows ({((initial_shape[0] - df.shape[0])/initial_shape[0]*100):.2f}%)")
        
        self.processed_df = df
        return df
    
    def calculate_rfm_metrics(self):
        """Calculate Recency, Frequency, Monetary metrics for each customer"""
        st.info("Calculating RFM metrics...")
        
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
        
        st.write(f"RFM metrics calculated for {len(rfm)} customers")
        st.write("RFM metrics summary:")
        st.write(rfm.describe())
        
        self.rfm_data = rfm
        return rfm
    
    def determine_optimal_clusters(self, max_k=10):
        """Determine optimal number of clusters using elbow method and silhouette score"""
        st.info(f"Determining optimal number of clusters (k={max_k})...")
        
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

        st.pyplot(fig)
        
        # Determine optimal k based on highest silhouette score
        optimal_k = k_range[np.argmax(silhouette_scores)]
        st.success(f"Optimal number of clusters based on silhouette score: {optimal_k}")
        
        return optimal_k, k_range, inertias, silhouette_scores
    
    def perform_customer_segmentation(self, n_clusters=4):
        """Perform customer segmentation using K-Means clustering"""
        st.info(f"Performing customer segmentation with {n_clusters} clusters...")
        
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
        
        st.write("Cluster Statistics:")
        st.write(cluster_stats)
        
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
        
        st.write("Segment Distribution:")
        st.write(rfm_clustered['Segment'].value_counts())
        
        # Store the segmentation results
        self.customer_segments = rfm_clustered
        
        return rfm_clustered, scaler, kmeans
    
    def visualize_customer_segments(self):
        """Visualize customer segments"""
        st.info("Creating visualizations for customer segments...")
        
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
        
        st.pyplot(fig)
        
        # Bar chart of segment distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        segment_counts = rfm_segmented['Segment'].value_counts()
        sns.barplot(x=segment_counts.index, y=segment_counts.values, ax=ax)
        ax.set_title('Customer Segment Distribution')
        ax.set_xlabel('Segment')
        ax.set_ylabel('Number of Customers')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    def create_product_recommendation_system(self):
        """Create product recommendation system using collaborative filtering"""
        st.info("Creating product recommendation system...")
        
        df = self.processed_df
        
        # Create a pivot table: customers x products (quantity)
        customer_product_matrix = df.pivot_table(
            index='CustomerID',
            columns='StockCode',
            values='Quantity',
            aggfunc='sum',
            fill_value=0
        )
        
        st.write(f"Customer-product matrix shape: {customer_product_matrix.shape}")
        
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

# File upload
uploaded_file = st.file_uploader("Upload your Online Retail CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)
    
    # Initialize session state
    if 'solution' not in st.session_state:
        st.session_state.solution = EcommerceMLSolution(df)
        st.session_state.scaler = None
        st.session_state.kmeans = None
        st.session_state.segment_mapping = None
        st.session_state.processing_complete = False
    
    # Process data if not already done
    if not st.session_state.processing_complete:
        with st.spinner("Processing data..."):
            # Load and clean data
            solution = st.session_state.solution
            solution.load_and_clean_data()
            
            # Calculate RFM metrics
            solution.calculate_rfm_metrics()
            
            # Determine optimal clusters
            optimal_k, _, _, _ = solution.determine_optimal_clusters(max_k=6)
            
            # Perform customer segmentation
            rfm_clustered, scaler, kmeans = solution.perform_customer_segmentation(n_clusters=optimal_k)
            
            # Store models in session state
            st.session_state.scaler = scaler
            st.session_state.kmeans = kmeans
            
            # Create segment mapping
            segment_mapping = dict(zip(rfm_clustered['Cluster'], rfm_clustered['Segment']))
            st.session_state.segment_mapping = segment_mapping
            
            # Create recommendation system
            solution.create_product_recommendation_system()
            
            # Mark as complete
            st.session_state.processing_complete = True
        
        st.success("Data processing complete!")
    
    # Futuristic tabs with enhanced styling
    st.markdown('<div class="section-header">AI-Powered Analytics Modules</div>', unsafe_allow_html=True)
    
    # Create animated tabs
    tab1, tab2 = st.columns(2)
    with tab1:
        st.markdown('<div class="metric-card" style="text-align: center; padding: 1.5rem;"><h3>🔍 Product Recommendations</h3><p>Discover similar products based on customer behavior</p></div>', unsafe_allow_html=True)
    with tab2:
        st.markdown('<div class="metric-card" style="text-align: center; padding: 1.5rem;"><h3>📊 Customer Segmentation</h3><p>Classify customers using RFM analysis</p></div>', unsafe_allow_html=True)
    
    # Actual tabs
    tab1, tab2 = st.tabs(["🔍 Product Recommendations", "📊 Customer Segmentation"])
    
    with tab1:
        st.markdown('<div class="section-header">Product Recommendation Engine</div>', unsafe_allow_html=True)
        st.markdown("Enter a product code to get AI-powered product recommendations")
        
        # Enhanced product selection with search
        df = st.session_state.solution.processed_df
        
        # Create a mapping of product codes to descriptions for search
        product_desc_map = df[['StockCode', 'Description']].drop_duplicates().set_index('StockCode')['Description'].to_dict()
        
        product_options = df['StockCode'].unique()
        
        # Add search functionality
        search_term = st.text_input("Search for a product by name or code", "")
        
        if search_term:
            # Filter options based on search in both code and description
            filtered_products = []
            search_lower = search_term.lower()
            for code in product_options:
                # Check if search term is in the code or description
                desc = product_desc_map.get(code, "")
                if search_lower in str(code).lower() or search_lower in desc.lower():
                    filtered_products.append(code)
            if len(filtered_products) == 0:
                st.warning("No products found. Showing all products.")
                filtered_products = product_options
        else:
            filtered_products = product_options
        
        # Create display options with both code and description
        display_options = {}
        for code in sorted([str(item) for item in filtered_products]):
            desc = product_desc_map.get(code, "No description")
            display_options[f"{code} - {desc[:50]}..." if len(desc) > 50 else f"{code} - {desc}"] = code
        
        # Create reverse mapping to get code from selection
        reverse_mapping = {v: k for k, v in display_options.items()}
        
        selected_display = st.selectbox("Select Product (Code - Description)", options=sorted(display_options.keys()))
        product_code = display_options[selected_display]
        
        # Enhanced recommendation display
        if st.button("🚀 Get Smart Recommendations", type="primary"):
            with st.spinner("🤖 AI is analyzing customer patterns..."):
                try:
                    recommendations = st.session_state.solution.recommend_products(str(product_code), top_n=recommendations_count)
                    
                    if isinstance(recommendations, str):
                        st.error(recommendations)
                    else:
                        st.markdown(f'<div class="metric-card" style="padding: 1.5rem;"><h4>🎯 Products similar to {product_code}</h4></div>', unsafe_allow_html=True)
                        
                        # Display recommendations in a more visually appealing way
                        for i, rec in enumerate(recommendations, 1):
                            similarity_percent = rec['Similarity'] * 100
                            st.markdown(f"<div class='recommendation-item'><strong>{i}. {rec['StockCode']}</strong><br><em>{rec['Description']}</em><br><small>Similarity: {'█' * int(similarity_percent/10)}{'░' * (10-int(similarity_percent/10))} {similarity_percent:.1f}%</small></div>", unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"❌ Error getting recommendations: {str(e)}")
        
        # Add trending products section
        if st.checkbox("Show Trending Products"):
            st.markdown('<div class="metric-card" style="padding: 1.5rem;"><h4>🔥 Trending Products</h4></div>', unsafe_allow_html=True)
            
            # Get top selling products
            top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(5)
            for i, (desc, qty) in enumerate(top_products.items(), 1):
                st.markdown(f"<div class='recommendation-item'><strong>{i}. {desc[:50]}...</strong><br><small>Sold: {qty:,} units</small></div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="section-header">Customer Segmentation Intelligence</div>', unsafe_allow_html=True)
        st.markdown("Enter RFM values to predict customer segment using AI")
        
        # Create interactive RFM input cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card" style="text-align: center; padding: 1rem;"><h4>📅 Recency</h4><p>Days since last purchase</p></div>', unsafe_allow_html=True)
            recency = st.slider("Days since last purchase", min_value=0, max_value=365, value=30)
            
        with col2:
            st.markdown('<div class="metric-card" style="text-align: center; padding: 1rem;"><h4>🔄 Frequency</h4><p>Number of transactions</p></div>', unsafe_allow_html=True)
            frequency = st.slider("Number of transactions", min_value=0, max_value=100, value=5)
        
        with col3:
            st.markdown('<div class="metric-card" style="text-align: center; padding: 1rem;"><h4>💰 Monetary</h4><p>Total amount spent</p></div>', unsafe_allow_html=True)
            monetary = st.slider("Total amount spent (£)", min_value=0.0, max_value=5000.0, value=100.0, step=10.0)
        
        # Enhanced prediction button with animation
        if st.button("🔮 Predict Customer Segment", type="primary"):
            if st.session_state.scaler is not None and st.session_state.kmeans is not None:
                with st.spinner("🧠 AI is analyzing customer profile..."):
                    segment = st.session_state.solution.predict_customer_segment(
                        recency, frequency, monetary,
                        st.session_state.scaler,
                        st.session_state.kmeans,
                        st.session_state.segment_mapping
                    )
                    
                    # Animated success message
                    st.balloons()
                    
                    # Enhanced segment display
                    st.markdown(f'<div class="segment-card" style="text-align: center; padding: 2rem;"><h3>🎯 Predicted Segment: {segment}</h3></div>', unsafe_allow_html=True)
                    
                    # Detailed segment explanation
                    if 'High-Value' in segment:
                        st.markdown("<div class='recommendation-item'><h4>💡 High-Value Customer</h4><p>Recently active customers with high purchase frequency and monetary value. These are your best customers!</p><ul><li>Target with loyalty programs</li><li>Offer premium services</li><li>Early access to new products</li></ul></div>", unsafe_allow_html=True)
                    elif 'Regular' in segment:
                        st.markdown("<div class='recommendation-item'><h4>💡 Regular Customer</h4><p>Consistent customers with moderate activity and spending. Great potential for growth!</p><ul><li>Encourage increased engagement</li><li>Target with personalized offers</li><li>Introduce complementary products</li></ul></div>", unsafe_allow_html=True)
                    elif 'Occasional' in segment:
                        st.markdown("<div class='recommendation-item'><h4>💡 Occasional Customer</h4><p>Low activity customers with lower spending. Need attention to increase engagement!</p><ul><li>Reactivation campaigns</li><li>Special incentives</li><li>Target with popular products</li></ul></div>", unsafe_allow_html=True)
                    elif 'At-Risk' in segment:
                        st.markdown("<div class='recommendation-item'><h4>⚠️ At-Risk Customer</h4><p>Inactive customers with low purchase history. Immediate action needed!</p><ul><li>Win-back campaigns</li><li>Special discount offers</li><li>Survey to understand concerns</li></ul></div>", unsafe_allow_html=True)
                    else:
                        st.info(f"Segment {segment} characteristics")
            else:
                st.error("Models not initialized properly")
        
        # Add segment distribution visualization
        if st.checkbox("Show Customer Segment Distribution"):
            if st.session_state.processing_complete and st.session_state.solution.customer_segments is not None:
                seg_counts = st.session_state.solution.customer_segments['Segment'].value_counts()
                
                # Create a more engaging visualization
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="metric-card" style="padding: 1.5rem;"><h4>📊 Segment Distribution</h4></div>', unsafe_allow_html=True)
                    # Create a horizontal bar chart
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(8, 4))
                    bars = ax.barh(seg_counts.index, seg_counts.values, color=['#6a11cb', '#2575fc', '#00dbde', '#fc00ff'])
                    ax.set_xlabel('Number of Customers')
                    # Add value labels on bars
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width, bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                               ha='left', va='center', fontweight='bold')
                    st.pyplot(fig)
                
                with col2:
                    st.markdown('<div class="metric-card" style="padding: 1.5rem;"><h4>📋 Segment Summary</h4></div>', unsafe_allow_html=True)
                    for segment, count in seg_counts.items():
                        percentage = (count / len(st.session_state.solution.customer_segments)) * 100
                        st.markdown(f"<div class='recommendation-item'><strong>{segment}</strong>: {count} customers<br><small>({percentage:.1f}% of total)</small></div>", unsafe_allow_html=True)
            else:
                st.info("Process data first to see segment distribution")
else:
    st.info("Please upload an Online Retail CSV file to get started")
    
    # Show project information
    st.markdown("## 🎯 Project Overview") 
    st.markdown("""
    This e-commerce ML solution includes:
    
    ### 🔹 Customer Segmentation System
    - Uses RFM Analysis + KMeans Clustering
    - Labels customers as: High-Value, Regular, Occasional, At-Risk
    - Based on Recency, Frequency, and Monetary metrics
    
    ### 🔹 Product Recommendation System
    - Uses Item-Based Collaborative Filtering
    - Recommends 5 similar products when user enters a product name
    - Based on customer purchasing patterns
    
    ### 🛠 How to Use:
    1. Upload the online_retail.csv file
    2. Wait for data processing to complete
    3. Use the tabs to access either Product Recommendations or Customer Segmentation
    """)