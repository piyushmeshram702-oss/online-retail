# E-commerce ML Solution - Customer Segmentation & Product Recommendations

This project implements a comprehensive e-commerce machine learning solution featuring two core systems:

🔹 **Customer Segmentation System** using RFM Analysis and K-Means Clustering

🔹 **Product Recommendation System** using Item-Based Collaborative Filtering

## 📋 Problem Statement

E-commerce companies collect massive amounts of transactional data daily. The challenge is to leverage this data to:
- Group customers into meaningful segments (loyal, regular, at-risk, etc.)
- Recommend similar products to users automatically

## 🎯 Project Goals

### A. Customer Segmentation System
Uses RFM Analysis + K-Means Clustering to classify customers as:
- **High-Value**: Recently active customers with high purchase frequency and monetary value
- **Regular**: Consistent customers with moderate activity and spending
- **Occasional**: Low activity customers with lower spending
- **At-Risk**: Inactive customers with low purchase history

### B. Product Recommendation System
Uses Item-Based Collaborative Filtering to:
- Recommend 5 similar products when user enters a product name
- Leverage customer purchasing patterns for accurate recommendations

## 🛠 Tech Stack

- **Python 3.x** - Core programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib/Seaborn** - Data visualization
- **Streamlit** - Interactive web application framework

## 📁 Files Structure

- `ecommerce_ml_solution.py` - Core ML solution implementation
- `app.py` - Main and only Streamlit web application with futuristic UI (deployment file)
- `requirements.txt` - Python package dependencies
- `README.md` - Project documentation
- `online_retail.csv` - Dataset for analysis

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Application
```bash
streamlit run streamlit_app.py
```

The application will be accessible at `http://localhost:8501`

## 📊 Methodology

### Step 1: Dataset Understanding
- Loaded the Online Retail dataset with columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

### Step 2: Data Preprocessing
- Removed duplicate transactions
- Converted InvoiceDate to datetime format
- Removed rows with missing CustomerID
- Filtered out cancelled invoices (starting with 'C')
- Excluded invalid transactions (Quantity <= 0 or UnitPrice <= 0)
- Calculated transaction Amount (Quantity × UnitPrice)

### Step 3: Feature Engineering (RFM Analysis)
For each customer calculated:
- **Recency (R)**: Days since last purchase
- **Frequency (F)**: Total number of transactions
- **Monetary (M)**: Total amount spent

### Step 4: Customer Segmentation
- Standardized RFM metrics using StandardScaler
- Determined optimal cluster count using Elbow Method and Silhouette Score
- Applied K-Means clustering algorithm
- Interpreted clusters to assign meaningful segment labels

### Step 5: Product Recommendation System
- Created customer-product matrix using pivot table
- Calculated product similarities using Cosine Similarity
- Implemented collaborative filtering for recommendations

### Step 6: Web Application Development
- Developed interactive Streamlit application
- Added futuristic UI with custom CSS styling
- Implemented real-time prediction capabilities
- Added search functionality for products by code or description

## 🌟 Key Features

### Customer Segmentation
- **RFM Analysis**: Comprehensive analysis of Recency, Frequency, and Monetary metrics
- **Automatic Clustering**: Dynamically determines optimal number of customer segments
- **Meaningful Labels**: Assigns intuitive segment names based on customer behavior
- **Real-time Prediction**: Predict customer segment based on RFM inputs
- **Visual Analytics**: Displays segment distribution and characteristics

### Product Recommendations
- **Collaborative Filtering**: Uses customer purchase patterns for accurate recommendations
- **Dual Search**: Search products by both code and description
- **Similarity Scores**: Provides confidence metrics for recommendations
- **Edge Case Handling**: Manages products not present in the dataset

### Interactive Web Interface
- **Futuristic Design**: Gradient colors, animations, and modern UI elements
- **Google Fonts**: Orbitron for headings and Exo 2 for body text
- **Responsive Layout**: Works across different screen sizes
- **Real-time Updates**: Instant feedback for user interactions
- **Visual Feedback**: Animations and hover effects for better UX

## 🎨 UI Enhancements

- **Custom CSS Styling**: Professional gradient backgrounds and modern design
- **Interactive Elements**: Sliders, animated buttons, and hover effects
- **Product Cards**: Visually appealing product displays with similarity indicators
- **Segment Visualization**: Color-coded customer segments with detailed explanations
- **Search Functionality**: Advanced search supporting both product codes and descriptions
- **Live Stats**: Real-time display of segment distributions and trending products

## 💡 Business Impact

- **Customer Retention**: Early identification of at-risk customers for targeted retention campaigns
- **Personalized Marketing**: Tailored marketing strategies based on customer segments
- **Sales Growth**: Cross-sell and upsell opportunities through product recommendations
- **Inventory Optimization**: Insights into product affinities for better stock management
- **Customer Experience**: Personalized shopping experiences leading to higher satisfaction

## 🚀 Deployment Options

### Local Development
```bash
streamlit run streamlit_app.py
```

### Streamlit Sharing (Cloud Deployment)
1. Clone the repository from [GitHub](https://github.com/piyushmeshram702-oss/online-retail)
2. Connect to share.streamlit.io
3. Configure main script as `app.py`
4. Make sure to include `requirements.txt` for dependency installation

### Cloud Platforms
The application can be deployed on:
- Heroku
- AWS
- Google Cloud Platform
- Azure

## 📈 Expected Outcomes

By implementing this solution, businesses can expect:
- Increased customer lifetime value through targeted marketing
- Higher conversion rates from personalized recommendations
- Improved customer retention through early intervention
- Better inventory management based on product affinity insights
- Enhanced customer satisfaction from personalized experiences