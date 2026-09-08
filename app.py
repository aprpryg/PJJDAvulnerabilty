import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
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
def train_models_and_evaluate(data):
    features = ['M101', 'krt_jk', 'krt_umur', 'krt_pendidikan', 'krt_bekerja', 'M1501', 'penerima_bansos']
    target = 'high_vulnerability'
    df_ml = data[features + [target]].dropna().copy()
    
    df_ml['krt_bekerja'] = df_ml['krt_bekerja'].apply(lambda x: 1 if str(x).strip() == 'A' else 0)
    
    X = pd.get_dummies(df_ml[features], columns=['M101', 'M1501', 'krt_pendidikan'], drop_first=True)
    y = df_ml[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Model 1: Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    
    # Model 2: Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    
    metrics = {
        'RF': {
            'auc': roc_auc_score(y_test, y_prob_rf),
            'precision': precision_score(y_test, y_pred_rf),
            'recall': recall_score(y_test, y_pred_rf),
            'f1': f1_score(y_test, y_pred_rf)
        },
        'LR': {
            'auc': roc_auc_score(y_test, y_prob_lr),
            'precision': precision_score(y_test, y_pred_lr),
            'recall': recall_score(y_test, y_pred_lr),
            'f1': f1_score(y_test, y_pred_lr)
        }
    }
    
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return rf, X.columns, importances, metrics, cm_rf

@st.cache_data
def load_geojson():
    url = "https://raw.githubusercontent.com/superpikar/indonesia-geojson/master/indonesia-province-simple.json"
    try:
        response = urllib.request.urlopen(url)
        return json.loads(response.read())
    except Exception:
        return None

df = load_data()
model, model_features, feature_importances, model_metrics, cm_rf = train_models_and_evaluate(df)
geojson_indo = load_geojson()

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
# HEADER GLOBAL
# =====================================================================
st.title("HOUSEHOLD VULNERABILITY MONITOR")
st.markdown("Identifikasi *Social Protection Gap* dan Pola Kerentanan Rumah Tangga (SUSENAS).")
st.divider()

# =====================================================================
# ARSITEKTUR TABS
# =====================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "01 — OVERVIEW", 
    "02 — VULNERABILITY GAP", 
    "03 — MODEL & INSIGHTS", 
    "04 — RISK SIMULATOR"
])

# ---------------------------------------------------------------------
# TAB 1: OVERVIEW
# ---------------------------------------------------------------------
with tab1:
    st.subheader("How large is the problem?")
    
    total_pop = df['FWT'].sum()
    high_vuln_pop = df[df['high_vulnerability'] == 1]['FWT'].sum()
    assisted_pop = df[df['penerima_bansos'] == 1]['FWT'].sum()
    residual_vuln_pop = df[(df['high_vulnerability'] == 1) & (df['penerima_bansos'] == 1)]['FWT'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total households", f"{total_pop/1_000_000:,.1f} juta")
    
    col2.metric("High vulnerability", f"{(high_vuln_pop/total_pop)*100:.1f}%")
    col2.caption("📝 **Note:** Keluarga yang mengalami minimal 2 masalah dasar sekaligus (ekonomi lemah, kurang gizi, atau tanpa listrik).")
    
    col3.metric("Assistance coverage", f"{(assisted_pop/total_pop)*100:.1f}%")
    
    col4.metric("Residual vulnerability", f"{(residual_vuln_pop/assisted_pop)*100:.1f}%")
    col4.caption("📝 **Note:** Persentase keluarga yang masih hidup dalam kondisi rentan **meskipun** sudah menerima bansos.")

    st.divider()
    st.subheader("Vulnerability by Province")
    
    if 'M101' in df.columns:
        prov_vuln = df[df['high_vulnerability'] == 1].groupby('M101')['FWT'].sum()
        prov_total = df.groupby('M101')['FWT'].sum()
        
        prov_pct = (prov_vuln / prov_total * 100).reset_index(name='Vuln_Pct')
        prov_pct['Vuln_Pct'] = prov_pct['Vuln_Pct'].round(1) 
        prov_pct['Nama_Provinsi'] = prov_pct['M101'].map(prov_map).fillna(prov_pct['M101'].astype(str))
        
        def categorize_vuln(val):
            if val < 20: return "Low (0-20%)"
            elif val <= 40: return "Moderate (20-40%)"
            else: return "High (>40%)"
        prov_pct['Category'] = prov_pct['Vuln_Pct'].apply(categorize_vuln)
        
        st.info("Top 5 kerentanan tertinggi didominasi wilayah Timur. Intervensi memerlukan pembangunan infrastruktur dasar yang struktural.")
        
        col_map_view, col_bar_view = st.columns([1.2, 1])
        with col_map_view:
            if geojson_indo:
                color_map = {"Low (0-20%)": "#fee5d9", "Moderate (20-40%)": "#fb6a4a", "High (>40%)": "#a50f15"}
                fig_map = px.choropleth(
                    prov_pct, geojson=geojson_indo, featureidkey="properties.Propinsi",
                    locations="Nama_Provinsi", color="Category", color_discrete_map=color_map,
                    category_orders={"Category": ["Low (0-20%)", "Moderate (20-40%)", "High (>40%)"]}
                )
                fig_map.update_geos(fitbounds="locations", visible=False)
                fig_map.update_traces(hovertemplate="<b>%{location}</b><br>Rentan: %{customdata[0]}%<extra></extra>", customdata=prov_pct[['Vuln_Pct']])
                fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=550, legend_title="Risk Level")
                st.plotly_chart(fig_map, use_container_width=True)
                
        with col_bar_view:
            view_all = st.checkbox("View all 38 provinces")
            if view_all:
                plot_df = prov_pct.sort_values('Vuln_Pct', ascending=True)
                h = 750
            else:
                top_10 = prov_pct.sort_values('Vuln_Pct', ascending=False).head(10)
                bottom_10 = prov_pct.sort_values('Vuln_Pct', ascending=True).head(10)
                plot_df = pd.concat([bottom_10, top_10]).sort_values('Vuln_Pct', ascending=True)
                h = 550
                
            fig_bar = px.bar(
                plot_df, x='Vuln_Pct', y='Nama_Provinsi', orientation='h',
                color='Category', color_discrete_map=color_map, height=h 
            )
            fig_bar.update_yaxes(dtick=1)
            fig_bar.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 2: VULNERABILITY GAP
# ---------------------------------------------------------------------
with tab2:
    st.subheader("Who is vulnerable? (Social Protection Gap Matrix)")
    
    underserved_pop = df[(df['high_vulnerability'] == 1) & (df['penerima_bansos'] == 0)]['FWT'].sum()
    kuadran_A = df[(df['high_vulnerability'] == 0) & (df['penerima_bansos'] == 0)]['FWT'].sum() / total_pop * 100
    kuadran_B = underserved_pop / total_pop * 100
    kuadran_C = df[(df['high_vulnerability'] == 0) & (df['penerima_bansos'] == 1)]['FWT'].sum() / total_pop * 100
    kuadran_D = df[(df['high_vulnerability'] == 1) & (df['penerima_bansos'] == 1)]['FWT'].sum() / total_pop * 100

    col_mat_1, col_mat_2 = st.columns([1.5, 1])
    with col_mat_1:
        st.markdown(f"""
        <table style="width:100%; text-align:center; font-size:16px; border-collapse: collapse;">
          <tr>
            <th style="border: none;"></th>
            <th colspan="2" style="text-align:center; background-color:#f0f2f6; padding:8px;">VULNERABILITY (≥ 2 Deprivations)</th>
          </tr>
          <tr>
            <th style="border: none;"></th>
            <th style="width:35%; text-align:center; background-color:#f0f2f6;">Low</th>
            <th style="width:35%; text-align:center; background-color:#f0f2f6;">High</th>
          </tr>
          <tr>
            <td style="text-align:right; font-weight:bold; padding:15px; background-color:#f0f2f6;">No assistance</td>
            <td style="border: 1px solid #ddd; padding:15px;">
                <span style="font-size:20px;">A</span><br>
                <span style="font-size:12px; font-weight:bold;">Low Priority</span><br>
                <span style="font-size:14px; color:gray;">{kuadran_A:.1f}%</span>
            </td>
            <td style="border: 1px solid #ddd; padding:15px; background-color:#fff0f0;">
                <span style="font-size:20px;">🔴 B</span><br>
                <span style="font-size:12px; font-weight:bold;">Potentially Underserved</span><br>
                <span style="font-size:11px; font-style:italic;">(Warga rentan yang luput dari bansos)</span><br>
                <span style="font-size:14px; color:gray;">{kuadran_B:.1f}%</span>
            </td>
          </tr>
          <tr>
            <td style="text-align:right; font-weight:bold; padding:15px; background-color:#f0f2f6;">Assistance</td>
            <td style="border: 1px solid #ddd; padding:15px;">
                <span style="font-size:20px;">C</span><br>
                <span style="font-size:12px; font-weight:bold;">Protected</span><br>
                <span style="font-size:14px; color:gray;">{kuadran_C:.1f}%</span>
            </td>
            <td style="border: 1px solid #ddd; padding:15px; background-color:#fff8eb;">
                <span style="font-size:20px;">🟠 D</span><br>
                <span style="font-size:12px; font-weight:bold;">Residual Vulnerability</span><br>
                <span style="font-size:11px; font-style:italic;">(Sudah dibantu tapi masih rentan)</span><br>
                <span style="font-size:14px; color:gray;">{kuadran_D:.1f}%</span>
            </td>
          </tr>
        </table>
        """, unsafe_allow_html=True)

    with col_mat_2:
        st.warning("**Policy Insights:**\n"
               "* **Kuadran B (Potentially Underserved):** Mengindikasikan *potential social protection gap* yang perlu diverifikasi dengan data Regsosek.\n"
               "* **Kuadran D (Residual Vulnerability):** Menunjukkan perlunya analisis kecukupan nilai transfer, ketepatan sasaran, dan intervensi pendamping.")

    st.divider()
    st.markdown("### Profile Filter")
    sel_kuadran = st.radio("Tampilkan karakteristik demografi dan deprivasi untuk:", 
                           ["Semua Rumah Tangga", "🔴 Kuadran B (Potentially Underserved)", "🟠 Kuadran D (Residual Vulnerability)"], horizontal=True)
    
    if "B" in sel_kuadran:
        df_prof = df[(df['high_vulnerability'] == 1) & (df['penerima_bansos'] == 0)]
    elif "D" in sel_kuadran:
        df_prof = df[(df['high_vulnerability'] == 1) & (df['penerima_bansos'] == 1)]
    else:
        df_prof = df.copy()
        
    avg_umur = (df_prof['krt_umur'] * df_prof['FWT']).sum() / df_prof['FWT'].sum() if len(df_prof)>0 else 0
    krt_wanita = df_prof[df_prof['krt_jk']==2]['FWT'].sum() / df_prof['FWT'].sum() * 100 if len(df_prof)>0 else 0
    
    col_p1, col_p2 = st.columns(2)
    col_p1.metric("Rata-rata Usia Kepala RT", f"{avg_umur:.1f} Tahun")
    col_p2.metric("KRT Perempuan (Female Headed)", f"{krt_wanita:.1f}%")

# ---------------------------------------------------------------------
# TAB 3: MODEL & INSIGHTS
# ---------------------------------------------------------------------
with tab3:
    st.subheader("Model Performance & Comparison")
    
    col_mc1, col_mc2, col_mc3 = st.columns([1.2, 1, 1.5])
    
    with col_mc1:
        st.markdown("**1. Model Comparison**")
        comp_df = pd.DataFrame([
            {"Model": "Random Forest", "AUC": model_metrics['RF']['auc'], "Precision": model_metrics['RF']['precision'], "Recall": model_metrics['RF']['recall'], "F1": model_metrics['RF']['f1']},
            {"Model": "Logistic Regression", "AUC": model_metrics['LR']['auc'], "Precision": model_metrics['LR']['precision'], "Recall": model_metrics['LR']['recall'], "F1": model_metrics['LR']['f1']}
        ])
        st.dataframe(comp_df.style.format({"AUC": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}), hide_index=True)
        st.caption("📝 **Keterangan Metrik:**\n"
                   "- **AUC:** Akurasi model dalam membedakan rumah tangga rentan vs tidak rentan.\n"
                   "- **Precision:** Ketepatan deteksi (persentase tebakan 'rentan' yang benar-benar rentan di lapangan).\n"
                   "- **Recall:** Keberhasilan melacak warga rentan tanpa terlewat.")
    
    with col_mc2:
        st.markdown("**2. RF Confusion Matrix**")
        fig_cm = px.imshow(cm_rf, text_auto=True, color_continuous_scale='Blues',
                           labels=dict(x="Predicted Label", y="True Label"),
                           x=['Low Vuln', 'High Vuln'], y=['Low Vuln', 'High Vuln'])
        fig_cm.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=200)
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("📝 **Note:** Matriks rincian tebakan model yang benar vs salah. Kotak biru gelap menunjukkan tebakan sistem yang tepat sasaran.")
        
    with col_mc3:
        st.markdown("**3. Top Vulnerability Drivers**")
        top_features = feature_importances.head(8).sort_values(by='Importance', ascending=True)
        fig_feat = px.bar(
            top_features, x='Importance', y='Feature', orientation='h',
            color='Importance', color_continuous_scale='Blues'
        )
        fig_feat.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=250)
        st.plotly_chart(fig_feat, use_container_width=True)
        st.caption("📝 **Note:** Urutan faktor-faktor struktural utama yang paling menentukan apakah sebuah keluarga rentan atau tidak.")

# ---------------------------------------------------------------------
# TAB 4: RISK SIMULATOR
# ---------------------------------------------------------------------
with tab4:
    st.subheader("Vulnerability Risk Simulator")
    st.markdown("Explore how household characteristics are associated with estimated vulnerability risk.")
    
    col_form, col_result = st.columns([1, 1])
    
    with col_form:
        with st.form("simulator_form"):
            prov_options = sorted(df['M101'].dropna().unique())
            sim_prov = st.selectbox("Provinsi (Location)", options=prov_options, format_func=lambda x: prov_map.get(x, str(x)))
            sim_umur = st.slider("Age of household head (Usia KRT)", 18, 90, 45)
            sim_jk = st.selectbox("Gender", [1, 2], format_func=lambda x: "Male (Laki-laki)" if x == 1 else "Female (Perempuan)")
            sim_kerja = st.selectbox("Employment", [1, 0], format_func=lambda x: "Working" if x == 1 else "Not Working")
            pend_options = [3, 8, 13, 21, 25]
            sim_pend = st.selectbox("Education", pend_options, format_func=lambda x: {3: "SD", 8: "SMP", 13: "SMA", 21: "S1", 25: "No Primary Education"}.get(x, str(x)))
            sim_rumah = st.selectbox("Housing", [1, 2, 3], format_func=lambda x: {1: "Own House", 2: "Rent", 3: "Rent-Free"}.get(x, str(x)))
            sim_bansos = st.selectbox("Social Assistance", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            
            submit_btn = st.form_submit_button("Estimate Vulnerability", use_container_width=True)
            
    with col_result:
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
            
            st.info("### Output")
            st.write(f"Untuk karakteristik tersebut, sistem mengestimasi probabilitas risiko kerentanan sebesar **{prob*100:.1f}%**.")
            
            if prob >= 0.5:
                st.error("**Risk Category: HIGH (RENTAN)**")
            else:
                st.success("**Risk Category: LOW (AMAN)**")
            
            st.markdown("**Karakteristik Pendorong (Top Drivers):**")
            top_3_global = feature_importances.head(3)['Feature'].tolist()
            st.write("Skor kerentanan pada keluarga ini paling kuat didorong oleh faktor-faktor berikut:")
            for f in top_3_global:
                if f == 'penerima_bansos':
                    st.write(f"- Status Bantuan Sosial ({'Menerima' if sim_bansos==1 else 'Tidak Menerima'})")
                elif f == 'krt_umur':
                    st.write(f"- Usia Kepala Rumah Tangga ({sim_umur} Tahun)")
                elif f.startswith('krt_pendidikan'):
                    st.write(f"- Status Pendidikan KRT")
                
            st.caption("⚠️ **Analytical Caveat:** Angka ini adalah perkiraan risiko berdasarkan pola data SUSENAS masa lalu, bukan alat untuk mengukur dampak sebab-akibat program perlindungan sosial.")