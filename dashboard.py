#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from datetime import date, timedelta
import compress_pickle as compickle

st.set_page_config(page_title="Retail Demand Forecasting Dashboard", page_icon="📦", layout="wide")

# load model data
@st.cache_data
def load_raw_data():
    try:
        df = pd.read_csv('df_model_sarimax.csv')
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        else:
            df.index = pd.to_datetime(df.index)
        
        df.index.name = 'Date'
        
        return df
    except FileNotFoundError:
        st.error("File 'df_model_sarimax.csv' tidak ditemukan.")
        st.stop()

df_model_sarimax = load_raw_data()
last_historical_date = df_model_sarimax.index.max().date()

@st.cache_resource
def load_objects():
    try:
        with open('preprocessor_dict.pkl', 'rb') as f:
            preprocessor_dict = pickle.load(f)
        
        with open('eval_dict.pkl', 'rb') as f:
            eval_dict = pickle.load(f)

        sarimax_models = {}
        for cat in preprocessor_dict.keys():
             with open(f'sarimax_model_{cat}.pkl.gz', 'rb') as f:
                sarimax_models[cat] = compickle.load(f)

        return preprocessor_dict, eval_dict, sarimax_models
    except FileNotFoundError:
        st.error("File model tidak ditemukan.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat objek: {e}")
        st.stop()

preprocessor_dict, eval_dict, sarimax_models = load_objects()
categories = list(preprocessor_dict.keys())

# Tampilan Dashboard 
st.set_page_config(
    page_title="Demand & Stock Optimization Dashboard",
    layout="wide"
)

st.title("🛒 Demand & Stock Optimization Dashboard")
st.markdown("Dashboard ini memprediksi permintaan harian dan merekomendasikan stok optimal berdasarkan data historis dan skenario masa depan.")

# Sidebar u/ input user
st.sidebar.header("⚙️ Pengaturan Prediksi")

with st.sidebar.form("prediction_form"):
    st.subheader("Rentang Tanggal")
    
    # Prediksi dimulai dari hari setelah data historis
    min_date_pred = last_historical_date + timedelta(days=1)
    # Prediksi maksimal 1 tahun dari tanggal terakhir dataset historis (sesuai data future)
    max_date_pred = last_historical_date + timedelta(days=365) 

    # Membatasi input tanggal sesuai dengan rentang prediksi yang valid (hanya 1 tahun)
    start_date = st.date_input(
        "Tanggal Mulai Prediksi", 
        value=min_date_pred, 
        min_value=min_date_pred, 
        max_value=max_date_pred
    )
    
    end_date = st.date_input(
        "Tanggal Akhir Prediksi", 
        value=min(start_date + timedelta(days=30), max_date_pred),
        min_value=start_date, 
        max_value=max_date_pred
    )
    
    if start_date >= end_date:
        st.error("Tanggal mulai harus lebih awal dari tanggal akhir.")
        st.stop()
    
    st.subheader("Pilihan Kategori")
    selected_category = st.selectbox("Pilih Kategori", categories)

    st.subheader("Skenario Bisnis")
    promo_scenario = st.selectbox("Promosi?", options=[0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    discount_scenario = st.selectbox("Tingkat Diskon", options=['rendah', 'sedang', 'tinggi'])

    submitted = st.form_submit_button("Generate Forecast")

# Logika Prediksi dan Output
if submitted:
    st.header(f"Hasil Prediksi untuk Kategori: **{selected_category}**")
    
    model = sarimax_models[selected_category]
    preprocessor = preprocessor_dict[selected_category]
    mape = eval_dict[selected_category]['MAPE']
    
    # Hitung jumlah hari prediksi
    forecast_days = (end_date - start_date).days + 1
    
    # Buat data eksogen untuk masa depan
    future_index = pd.date_range(start=start_date, periods=forecast_days, freq='D')
    
    exog_cols_raw = ['Weather Condition', 'Seasonality', 'Region', 'Store ID', 'discount_level', 'price_level', 'Promotion', 'Epidemic']
    X_future_raw = pd.DataFrame(index=future_index, columns=exog_cols_raw)

    # Mengisi data eksogen asumsi
    for col in X_future_raw.columns:
        if col == 'discount_level':
            X_future_raw[col] = discount_scenario
        elif col == 'price_level':
            X_future_raw[col] = 'sedang'
        elif col == 'Promotion':
            X_future_raw[col] = promo_scenario
        elif col == 'Epidemic':
            X_future_raw[col] = 0
        elif col == 'Weather Condition':
            X_future_raw[col] = 'Normal'
        elif col == 'Seasonality':
            month = X_future_raw.index.month
            season_map = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn', 12: 'Winter'}
            X_future_raw[col] = [season_map[m] for m in month]
        elif col == 'Region':
            region_mode = pd.Series(df_model_sarimax[df_model_sarimax['Category'] == selected_category]['Region']).mode()[0]
            X_future_raw[col] = region_mode
        elif col == 'Store ID':
            store_mode = pd.Series(df_model_sarimax[df_model_sarimax['Category'] == selected_category]['Store ID']).mode()[0]
            X_future_raw[col] = store_mode
        else:
            X_future_raw[col] = 0

    X_future_encoded = preprocessor.transform(X_future_raw)
    expected_columns_after_transform = preprocessor.get_feature_names_out()
    X_future_df = pd.DataFrame(X_future_encoded, index=future_index, columns=expected_columns_after_transform).fillna(0)

    future_forecast = model.get_forecast(steps=forecast_days, exog=X_future_df.astype(float))
    forecast_df = future_forecast.predicted_mean.to_frame('predicted_demand')
    
    forecast_df['optimal_stock'] = forecast_df['predicted_demand'] * (1 + mape)
    forecast_df['optimal_stock'] = forecast_df['optimal_stock'].apply(lambda x: max(0, x))
    forecast_df = forecast_df.round(2)
    
    st.subheader("Grafik Prediksi Demand & Stok Optimal")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(forecast_df.index, forecast_df['predicted_demand'], label='Predicted Demand', color='green', marker='o', markersize=3)
    ax.plot(forecast_df.index, forecast_df['optimal_stock'], label='Optimal Stock', color='orange', linestyle='--')
    ax.set_title(f"Prediksi Harian {selected_category} ({start_date} s/d {end_date})")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Jumlah Unit")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    st.subheader("Rekomendasi Stok Optimal (Tabel)")
    st.dataframe(forecast_df, use_container_width=True)

st.sidebar.header("💡 Performa Model")
st.sidebar.markdown("Ringkasan MAE & MAPE pada data test.")
for cat, metrics in eval_dict.items():
    st.sidebar.markdown(f"- **{cat}**: MAE={metrics['MAE']:.2f}, MAPE={metrics['MAPE']:.2%}")

# In[ ]:





# In[ ]:




