import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import random
import os

# --- Autoencoder ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# --- Streamlit Config ---
st.set_page_config(page_title="Insider Threat Detection", layout="wide")
st.title("🔐 Insider Threat Detection (CERT Dataset)")

uploaded_logon = st.sidebar.file_uploader("Upload logon.csv", type=["csv"])
uploaded_file = st.sidebar.file_uploader("Upload file.csv", type=["csv"])

# --- Dummy Dataset Generator ---
def generate_dummy_data():
    num_users = 5
    num_days = 10
    pcs = ['PC-1', 'PC-2', 'PC-3']
    files = ['report.docx', 'data.xlsx', 'presentation.pptx', 'notes.txt']
    start_date = datetime.today() - timedelta(days=num_days)

    logon_records = []
    file_records = []
    for user_id in range(1, num_users + 1):
        user = f"user{user_id}"
        for day in range(num_days):
            date = start_date + timedelta(days=day)
            # Logon entries
            for _ in range(random.randint(1,5)):
                logon_records.append({
                    'id': random.randint(1000,9999),
                    'date': date + timedelta(hours=random.randint(0,23)),
                    'user': user,
                    'pc': random.choice(pcs),
                    'activity': random.choice(['Logon','Logoff'])
                })
            # File entries
            for _ in range(random.randint(1,5)):
                file_records.append({
                    'id': random.randint(1000,9999),
                    'date': date + timedelta(hours=random.randint(0,23)),
                    'user': user,
                    'pc': random.choice(pcs),
                    'filename': random.choice(files),
                    'activity': 'File Access',
                    'to_removable_media': random.choice([0,1]),
                    'from_removable_media': random.choice([0,1]),
                    'content': 'dummy_content',
                    'size': random.randint(100,10000)
                })

    logon_df = pd.DataFrame(logon_records)
    file_df = pd.DataFrame(file_records)
    return logon_df, file_df

# --- Load Data ---
if uploaded_logon and uploaded_file:
    logon = pd.read_csv(uploaded_logon)
    file = pd.read_csv(uploaded_file)
else:
    st.info("No CSV uploaded → using dummy dataset for demonstration.")
    logon, file = generate_dummy_data()

# --- Preprocess dates consistently ---
logon['date'] = pd.to_datetime(logon['date']).dt.floor('D')
file['date'] = pd.to_datetime(file['date']).dt.floor('D')

# --- Aggregate logon ---
logon_features = logon.groupby(['user','date']).agg(
    logon_count=('pc','count'),
    unique_pcs=('pc','nunique'),
    failed_logons=('activity', lambda x: (x=='Logoff').sum())
).reset_index()

# --- Aggregate file ---
file_features = file.groupby(['user','date']).agg(
    files_accessed=('filename','count')
).reset_index()

# --- Merge datasets (outer join ensures all users) ---
df = pd.merge(logon_features, file_features, on=['user','date'], how='outer').fillna(0)

# --- Features for ML ---
features = ['logon_count','unique_pcs','failed_logons','files_accessed']
X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Isolation Forest ---
iso = IsolationForest(contamination=0.05, random_state=42)
df['anomaly_iso'] = iso.fit_predict(X_scaled)

# --- Autoencoder ---
input_dim = X_scaled.shape[1]
model = Autoencoder(input_dim)

# Check if model exists
if os.path.exists("models/autoencoder.pth"):
    model.load_state_dict(torch.load("models/autoencoder.pth", map_location=torch.device('cpu')))
    model.eval()
else:
    st.warning("Autoencoder model not found! Please train it first in the notebook.")

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
recon = model(X_tensor).detach().numpy()
mse = np.mean(np.square(X_scaled - recon), axis=1)
threshold = np.percentile(mse, 95)
df['anomaly_auto'] = (mse > threshold).astype(int)
df['score'] = mse

# --- Show Anomaly Detection Table ---
st.subheader("⚠️ Anomaly Detection Table")
st.dataframe(df.sort_values('score', ascending=False))

# --- Scatter plot of anomaly scores ---
fig = px.scatter(df, x='date', y='score', color=df['anomaly_auto'].astype(str),
                 hover_data=['user'], title="Autoencoder Anomaly Scores")
st.plotly_chart(fig, use_container_width=True)

# --- Line chart per user ---
user_select = st.selectbox("Select User", df['user'].unique())
user_df = df[df['user']==user_select].sort_values('date')
st.line_chart(user_df.set_index('date')['score'])
