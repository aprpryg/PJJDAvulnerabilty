import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
import urllib.request
import json

# =====================================================================
# KONFIGURASI HALAMAN
# =====================================================================
st.set_page_config(page_title="Household Vulnerability Monitor", layout="wide")

# =====================================================================
# FUNGSI CACHE: DATA, MODEL, & GEOJSON
# =====================================================================
@st.cache_data
def load_data():
    return pd.read_csv('susenas_ml_ready.csv.gz', compression='gzip')

@st.cache_resource
def train_model(data):
    features = ['M101', 'krt_jk', 'krt_umur', 'krt_pendidikan', 'krt_bekerja', 'M1501', 'penerima_bansos']
    target = 'high_vulnerability'
    df_ml = data[features + [target]].dropna().copy()
    
    # Konversi isian huruf ke angka
    df_ml['krt_bekerja'] = df_ml['krt_bekerja'].apply(lambda x: 1 if str(x).strip() == 'A' else 0)
    
    # One-Hot Encoding
    X = pd.get_dummies(df_ml[features], columns=['M101', 'M1501', 'krt_pendidikan'], drop_first=True)
    y = df_ml[target]
    
    # Latih model
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    rf.fit(X, y)
    
    return rf, X.columns

@st.cache_data
def load_geojson():
    url = "https://raw.githubusercontent.com/superpikar/indonesia-geojson/master/indonesia-province-simple.json"
    try:
        response = urllib.request.urlopen(url)
        return json.loads(response.read())
    except Exception:
        return None

# Eksekusi pemanggilan fungsi
df = load_data()
model, model_features = train_model(df)
geojson_indo = load_geojson()

# Mapping Provinsi
prov_map = {
    11: 'ACEH', 12: 'SUMATERA UTARA', 13: 'SUMATERA BARAT', 14: 'RIAU', 15: 'JAMBI', 
    16: 'SUMATERA SELATAN', 17: 'BENGKULU', 18: 'LAMPUNG', 19: 'BANGKA BELITUNG', 21: 'KEPULAUAN RIAU', 
    31: 'DKI JAKARTA', 32: 'JAWA BARAT', 33: 'JAWA TENGAH', 34: 'DI YOGYAKARTA', 35: 'JAWA TIMUR', 
    36: 'BANTEN', 51: 'BALI', 52: 'NUSA TENGGARA BARAT', 53: 'NUSA TENGGARA TIMUR', 61: 'KALIMANTAN BARAT', 
    62: 'KALIMANTAN TENGAH', 63: 'KALIMANTAN SELATAN', 64: 'KALIMANTAN TIMUR', 65: 'KALIMANTAN UTARA', 
    71: 'SULAWESI UTARA', 72: 'SULAWESI TENGAH', 73: 'SULAWESI SELATAN', 74: 'SULAWESI TENGGARA', 
    75: 'GORONTALO', 76: 'SULAWESI BARAT', 81: 'MALUKU', 82: 'MALUKU UTARA', 91: 'PAPUA BARAT', 
    92: 'PAPUA BARAT DAYA', 94: 'PAPUA', 95: 'PAPUA SELATAN', 96: 'PAPUA TENGAH', 97: 'PAPUA PEGUNUNGAN'
}

# =====================================================================
# SIDEBAR: ML VULNERABILITY SIMULATOR
# =====================================================================
with st.sidebar:
    st.header("🎯 Predictive Simulator")
    st.markdown("Ubah karakteristik rumah tangga di bawah ini untuk melihat estimasi probabilitas risiko kerentanan.")
    
    with st.form("simulator_form"):
        prov_options = sorted(df['M101'].dropna().unique())
        sim_prov = st.selectbox("Provinsi", options=prov_options, format_func=lambda x: prov_map.get(x, str(x)))
        
        st.markdown("**Karakteristik Kepala RT**")
        sim_umur = st.slider("Usia KRT", 18, 90, 45)
        sim_jk = st.selectbox("Jenis Kelamin", [1, 2], format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")
        sim_kerja = st.selectbox("Status Pekerjaan", [1, 0], format_func=lambda x: "Bekerja" if x == 1 else "Tidak Bekerja")
        
        pend_options = [3, 8, 13, 21, 25]
        sim_pend = st.selectbox("Pendidikan Tertinggi", pend_options, 
                                format_func=lambda x: {3: "SD", 8: "SMP", 13: "SMA", 21: "S1", 25: "Tanpa Ijazah SD"}.get(x, str(x)))
        
        sim_rumah = st.selectbox("Status Kepemilikan Rumah", [1, 2, 3], format_func=lambda x: {1: "Milik Sendiri", 2: "Sewa/Kontrak", 3: "Bebas Sewa"}.get(x, str(x)))
        
        st.markdown("**Intervensi Kebijakan**")
        sim_bansos = st.selectbox("Menerima Bansos?", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
        
        submit_btn = st.form_submit_button("Hitung Estimasi Risiko", use_container_width=True)
        
    if submit_btn:
        input_dict = {
            'krt_jk': sim_jk,
            'krt_umur': sim_umur,
            'krt_bekerja': sim_kerja,
            'penerima_bansos': sim_bansos,
            f'M101_{float(sim_prov)}': 1,
            f'M1501_{float(sim_rumah)}': 1,
            f'krt_pendidikan_{float(sim_pend)}': 1
        }
        
        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=model_features, fill_value=0)
        
        prob = model.predict_proba(input_df)[0][1]
        
        st.divider()
        st.write("### Hasil Simulasi:")
        if prob >= 0.5:
            st.error(f"**Status:** Tergolong Rumah Tangga Rentan\n\n**Probabilitas Risiko:** {prob*100:.1f}%")
        else:
            st.success(f"**Status:** Tergolong Rumah Tangga Aman (Non-Rentan)\n\n**Probabilitas Risiko:** {prob*100:.1f}%")
            
        st.caption("⚠️ **Analytical Caveat:** Skor di atas merupakan estimasi pola asosiatif berdasarkan data *cross-sectional* SUSENAS, bukan hubungan sebab-akibat absolut.")

# =====================================================================
# MAIN LAYOUT: BARIS 1 (KPI)
# =====================================================================
total_pop = df['FWT'].sum()
high_vuln_pop = df[df['high_vulnerability'] == 1]['FWT'].sum()
assisted_pop = df[df['penerima_bansos'] == 1]['FWT'].sum()
residual_vuln_pop = df[(df['high_vulnerability'] == 1) & (df['penerima_bansos'] == 1)]['FWT'].sum()
underserved_pop = df[(df['high_vulnerability'] == 1) & (df['penerima_bansos'] == 0)]['FWT'].sum()

st.title("HOUSEHOLD VULNERABILITY MONITOR")
st.markdown("Mengidentifikasi Rumah Tangga Rentan di Luar Indikator Pendapatan dan Cakupan Bantuan Sosial")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total households", f"{total_pop/1_000_000:,.1f} juta")
col2.metric("High vulnerability", f"{(high_vuln_pop/total_pop)*100:.1f}%")
col3.metric("Receiving assistance", f"{(assisted_pop/total_pop)*100:.1f}%")
col4.metric("Assisted but vulnerable", f"{(residual_vuln_pop/total_pop)*100:.1f}%")

st.divider()

# =====================================================================
# MAIN LAYOUT: BARIS 2 (GAP MATRIX & PROFILE)
# =====================================================================
col_matrix, col_profile = st.columns([1.2, 1])

with col_matrix:
    st.subheader("Social Protection Gap Matrix")
    
    kuadran_A = df[(df['high_vulnerability'] == 0) & (df['penerima_bansos'] == 0)]['FWT'].sum() / total_pop * 100
    kuadran_B = underserved_pop / total_pop * 100
    kuadran_C = df[(df['high_vulnerability'] == 0) & (df['penerima_bansos'] == 1)]['FWT'].sum() / total_pop * 100
    kuadran_D = residual_vuln_pop / total_pop * 100

    st.markdown(f"""
    <table style="width:100%; text-align:center; font-size:16px; border-collapse: collapse;">
      <tr>
        <th style="border: none;"></th>
        <th colspan="2" style="text-align:center; background-color:#f0f2f6; padding:8px;">VULNERABILITY</th>
      </tr>
      <tr>
        <th style="border: none;"></th>
        <th style="width:35%; text-align:center; background-color:#f0f2f6;">Low</th>
        <th style="width:35%; text-align:center; background-color:#f0f2f6;">High</th>
      </tr>
      <tr>
        <td style="text-align:right; font-weight:bold; padding:15px; background-color:#f0f2f6;">No assistance</td>
        <td style="border: 1px solid #ddd; padding:15px;">
            <span style="font-size:20px;">A</span><br><span style="font-size:14px; color:gray;">{kuadran_A:.1f}%</span>
        </td>
        <td style="border: 1px solid #ddd; padding:15px; background-color:#fff0f0;">
            <span style="font-size:20px;">🔴 B</span><br>
            <span style="font-size:12px; font-weight:bold;">Potentially Underserved</span><br>
            <span style="font-size:14px; color:gray;">{kuadran_B:.1f}%</span>
        </td>
      </tr>
      <tr>
        <td style="text-align:right; font-weight:bold; padding:15px; background-color:#f0f2f6;">Assistance</td>
        <td style="border: 1px solid #ddd; padding:15px;">
            <span style="font-size:20px;">C</span><br><span style="font-size:14px; color:gray;">{kuadran_C:.1f}%</span>
        </td>
        <td style="border: 1px solid #ddd; padding:15px; background-color:#fff8eb;">
            <span style="font-size:20px;">🟠 D</span><br>
            <span style="font-size:12px; font-weight:bold;">Residual Vulnerability</span><br>
            <span style="font-size:14px; color:gray;">{kuadran_D:.1f}%</span>
        </td>
      </tr>
    </table>
    """, unsafe_allow_html=True)

    st.warning("**Policy Insight:**\n"
           "* **🔴 Kuadran B (Underserved):** Kelompok rentan yang luput dari jaring pengaman sosial. Mengindikasikan perlunya pembaruan data registrasi sosial (*exclusion error*).\n"
           "* **🟠 Kuadran D (Residual Vulnerability):** Bantuan sudah diterima, namun rumah tangga tetap rentan. Bantuan tunai/sembako saat ini belum cukup untuk mengentaskan akar deprivasi multidimensi.")

with col_profile:
    st.subheader("Vulnerability Profile")
    
    def calc_weighted_pct(indicator, condition):
        subset = df[df['high_vulnerability'] == condition]
        if subset['FWT'].sum() == 0: return 0
        return (subset[subset[indicator] == 1]['FWT'].sum() / subset['FWT'].sum()) * 100

    profile_data = {
        "Indicator": [
            "Low expenditure", 
            "Food / Nutritional deprivation", 
            "Poor housing / utilities",
            "Receiving social assistance"
        ],
        "Vulnerable": [
            f"{calc_weighted_pct('econ_vuln', 1):.1f}%",
            f"{calc_weighted_pct('food_vuln', 1):.1f}%",
            f"{calc_weighted_pct('housing_vuln', 1):.1f}%",
            f"{calc_weighted_pct('penerima_bansos', 1):.1f}%"
        ],
        "Non-vulnerable": [
            f"{calc_weighted_pct('econ_vuln', 0):.1f}%",
            f"{calc_weighted_pct('food_vuln', 0):.1f}%",
            f"{calc_weighted_pct('housing_vuln', 0):.1f}%",
            f"{calc_weighted_pct('penerima_bansos', 0):.1f}%"
        ]
    }
    
    st.dataframe(pd.DataFrame(profile_data), hide_index=True, use_container_width=True)
    
    st.info("**Data Insight:**\n"
        "Perhatikan indikator **Food / Nutritional deprivation**. Mayoritas rumah tangga rentan (mendekati 100%) mengalami defisit nutrisi dasar, meskipun persentase 'Low expenditure' mereka lebih rendah. Berada di atas garis kemiskinan moneter tidak menjamin ketahanan pangan.")

st.divider()

# =====================================================================
# MAIN LAYOUT: BARIS 3 (VULNERABILITY BY PROVINCE)
# =====================================================================
st.subheader("Vulnerability by Province")

if 'M101' in df.columns:
    prov_vuln = df[df['high_vulnerability'] == 1].groupby('M101')['FWT'].sum()
    prov_total = df.groupby('M101')['FWT'].sum()
    
    prov_pct = (prov_vuln / prov_total * 100).reset_index(name='Vuln_Pct')
    prov_pct['Vuln_Pct'] = prov_pct['Vuln_Pct'].round(1) 
    prov_pct['Nama_Provinsi'] = prov_pct['M101'].map(prov_map).fillna(prov_pct['M101'].astype(str))
    
    st.info("**Geographical Insight:**\n"
        "Terdapat ketimpangan spasial yang tegas. Kerentanan tertinggi terkonsentrasi di wilayah Indonesia Timur (Nusa Tenggara, Maluku, Papua). Pengentasan kerentanan di wilayah ini membutuhkan intervensi struktural infrastruktur dasar, bukan sekadar distribusi bantuan sosial standar.")
    
    col_map_view, col_bar_view = st.columns(2)
    
    with col_map_view:
        if geojson_indo:
            fig_map = px.choropleth(
                prov_pct,
                geojson=geojson_indo,
                featureidkey="properties.Propinsi",
                locations="Nama_Provinsi",
                color="Vuln_Pct",
                color_continuous_scale="Reds",
                labels={'Vuln_Pct': '% Rentan'}
            )
            fig_map.update_geos(fitbounds="locations", visible=False)
            fig_map.update_traces(hovertemplate="<b>%{location}</b><br>Rentan: %{z}%<extra></extra>")
            fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=650)
            st.plotly_chart(fig_map, use_container_width=True)
            st.caption("Sebaran Geografis. Area abu-abu menandakan perbedaan ejaan base map (geojson).")
            
    with col_bar_view:
        fig_bar = px.bar(
            prov_pct.sort_values('Vuln_Pct', ascending=True), 
            x='Vuln_Pct', y='Nama_Provinsi', orientation='h',
            labels={'Vuln_Pct': '% Rentan', 'Nama_Provinsi': ''},
            color='Vuln_Pct', color_continuous_scale='Reds', 
            height=850 
        )
        fig_bar.update_yaxes(dtick=1)
        fig_bar.update_traces(hovertemplate="<b>%{y}</b><br>Rentan: %{x}%<extra></extra>")
        fig_bar.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.warning("Variabel M101 tidak ditemukan di dataset.")