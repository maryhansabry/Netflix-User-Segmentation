# Netflix-User-Segmentation
🎬 Netflix User Segmentation — AML Project

Unsupervised deep learning pipeline for customer segmentation using Autoencoder + K-Means on real Netflix user data.


📦 Dataset
Source: Netflix Users Database — Kaggle
FieldDescriptionAgeUser ageCountryCountry of the userSubscription_TypeBasic / Standard / PremiumFavorite_GenreAction, Drama, Comedy…Watch_Time_HoursAverage daily watch hoursLast_LoginDate of last platform login

👥 Team & Contributions
#NameContribution1Radwa HanyData Loading & Data Cleaning · Data Visualization2Mai HusseinFeature Engineering & Data Preprocessing3Eathar HaithamModel Architecture & Training4Maryhan SabryDeployment · Latent Feature Extraction · GUI5Malak AbdelkawyClustering Using K-Means6Febronia EmadClustering Using Agglomerative

🔄 Pipeline Overview
Raw CSV  →  Cleaning  →  Feature Engineering  →  Encoding & Scaling
    →  Autoencoder (Latent Space)  →  K-Means Clustering  →  Streamlit App

🧱 Project Structure
📁 project/
├── app.py                  # Streamlit entry point
├── config.py               # Constants, cluster metadata, countries
├── features.py             # Feature engineering & model prediction
├── gemini_utils.py         # Gemini AI chat integration
├── ui_single.py            # Single user analysis page
├── ui_batch.py             # Batch analysis page
├── requirements.txt        # Python dependencies
└── models/
    ├── encoder_model.keras  # Trained encoder
    ├── kmeans_model.pkl     # Trained K-Means
    └── scaler.pkl           # Fitted StandardScaler

⚙️ Methodology
1 · Data Loading & Cleaning — Radwa Hany

Loaded dataset via kagglehub
Checked for nulls, duplicates, and data types
Explored distributions (age, watch time, country, genre)

2 · Feature Engineering — Mai Hussein

Converted Last_Login → Days_Since_Last_Login
Created Engagement_Score = Watch_Time_Hours / (Days_Since_Last_Login + 1)
Binned Age → Age_Group (Child / Young / Adult / MidAge / Senior)
Quantile-binned Watch_Level (Low / Medium / High / Very_High)
Quantile-binned Activity_Level (Active / Medium / Inactive)
Flagged Low_Engagement_Premium users (Premium subscribers with below-median engagement)

3 · Model Architecture & Training — Eathar Haitham

One-Hot Encoded categorical features, scaled with StandardScaler
Built Autoencoder in Keras:

Encoder: Input(37) → Dense(16) → Dense(8) → Latent(4)
Decoder: Latent(4) → Dense(8) → Dense(16) → Output(37)


Trained for 50 epochs, batch size 32, 20% validation split
Optimizer: Adam · Loss: MSE

4 · Latent Feature Extraction & Deployment — Maryhan Sabry

Extracted encoder sub-model from trained autoencoder
Generated 4-dimensional latent representation per user
Built full Streamlit GUI with Single User & Batch Analysis modes
Integrated Gemini AI for natural language analyst chat

5 · Clustering — K-Means — Malak Abdelkawy

Selected K=3 via Elbow Method + Silhouette Score analysis
Applied KMeans(n_clusters=3, random_state=42, n_init=10)
Labeled clusters:

ClusterLabelProfile0🔥 Active Heavy WatchersHigh watch time, frequent logins1😴 Inactive UsersLow engagement, high churn risk2📺 Casual UsersModerate usage, irregular activity
6 · Clustering — Agglomerative — Febronia Emad

Plotted dendrogram using Ward linkage to validate cluster count
Applied AgglomerativeClustering(n_clusters=3, linkage='ward')
Compared against K-Means via Silhouette Score


📊 Model Comparison
MetricK-MeansAgglomerativeNumber of Clusters33Algorithm TypeCentroid-basedConnectivity-basedSilhouette Scoresee notebooksee notebook

🚀 Run the App
bash# Install dependencies
pip install -r requirements.txt

# Add Gemini API key (optional, for AI chat)
echo "GEMINI_API_KEY=your_key_here" > .env

# Launch
streamlit run app.py

🤖 Gemini AI Integration
The app uses Gemini 2.5 Flash to provide natural language insights per user segment, with regional market context baked into the system prompt.

Built with Keras 3 · Scikit-learn · Streamlit · Plotly · Google Generative AI
