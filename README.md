# E-commerce Machine Learning Solution

This project implements two key machine learning systems for e-commerce businesses:
1. **Customer Segmentation System** using RFM Analysis and K-Means Clustering
2. **Product Recommendation System** using Collaborative Filtering

## 📋 Problem Statement

E-commerce companies collect huge amounts of data every day — who bought what, when, how often, and how much they spent. The challenge is to use this data to:
- Group customers into meaningful segments (like loyal, regular, at-risk, etc.)
- Recommend similar products to users automatically

## 🎯 Project Goals

### A. Customer Segmentation System
Uses RFM Analysis + KMeans Clustering to label customers as:
- High-Value
- Regular
- Occasional
- At-Risk

### B. Product Recommendation System
Uses Item-Based Collaborative Filtering to:
- Recommend 5 similar products when user enters a product name

## 🛠 Tech Stack

- Python 3.x
- Pandas - Data manipulation
- NumPy - Numerical computations
- Scikit-learn - Machine learning algorithms
- Matplotlib/Seaborn - Data visualization
- Streamlit - Web application framework

## 📁 Files Structure

- `ecommerce_ml_solution.py` - Main implementation of the ML solution
- `streamlit_app.py` - Streamlit web application
- `requirements.txt` - Python package dependencies
- `online_retail.csv` - Dataset (provided)

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Analysis Script
```bash
python ecommerce_ml_solution.py
```

### 3. Run the Streamlit Application
```bash
streamlit run streamlit_app.py
```

## 📊 Methodology

### Step 1: Dataset Understanding
- Loaded and inspected the dataset with columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

### Step 2: Data Preprocessing
- Removed rows where CustomerID is missing
- Removed cancelled invoices (InvoiceNo starts with 'C')
- Removed rows with Quantity <= 0 or UnitPrice <= 0

### Step 3: Exploratory Data Analysis (EDA)
- Explored sales patterns by country
- Analyzed top-selling products
- Examined sales trends over time
- Investigated RFM distributions

### Step 4: Feature Engineering (RFM Analysis)
For each customer calculated:
- **Recency (R)**: Days since last purchase
- **Frequency (F)**: Number of transactions
- **Monetary (M)**: Total money spent

### Step 5: Clustering (Customer Segmentation)
- Scaled RFM using StandardScaler
- Used KMeans clustering algorithm
- Determined optimal k using Elbow Method and Silhouette Score
- Assigned cluster labels and interpreted clusters

### Step 6: Product Recommendation System
- Created customer-product pivot table
- Computed similarity between products using Cosine Similarity
- Implemented collaborative filtering for recommendations

### Step 7: Streamlit App Deployment
- Built interactive web interface with two modules:
  - Product Recommendation module
  - Customer Segmentation module

## 📈 Key Features

### Customer Segmentation
- Automatically determines optimal number of clusters
- Provides meaningful segment labels based on RFM characteristics
- Visualizes customer segments in multiple dimensions

### Product Recommendations
- Uses collaborative filtering to find similar products
- Provides similarity scores for recommendations
- Handles edge cases where products might not exist in the dataset

### Interactive Web Interface
- Easy-to-use Streamlit app
- Real-time predictions for customer segmentation
- Product recommendation functionality
- Responsive design for various screen sizes

## 💡 Business Impact

- **Customer Retention**: Identify at-risk customers early for targeted retention campaigns
- **Personalized Marketing**: Tailor marketing strategies based on customer segments
- **Sales Growth**: Recommend relevant products to increase average order value
- **Inventory Management**: Understand product affinities for better stock decisions