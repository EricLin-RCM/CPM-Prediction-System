import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta
import io

# --- 網頁設定 ---
st.set_page_config(page_title="CRM 產品預測系統", page_icon="📈", layout="wide")

# --- 核心邏輯函數 (v4) ---
def run_product_automation_v4_web(uploaded_file):
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("正在讀取 Excel 檔案...")
    
    try:
        all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
    except Exception as e:
        st.error(f"檔案讀取失敗: {e}")
        return None

    final_summary = []
    total_sheets = len(all_sheets)
    processed_count = 0

    for product_id, df in all_sheets.items():
        processed_count += 1
        progress = int((processed_count / total_sheets) * 100)
        progress_bar.progress(progress)
        status_text.text(f"正在分析產品: {product_id} ({processed_count}/{total_sheets})")

        # A. 資料清洗
        if '單據日期' not in df.columns or '數量' not in df.columns:
            continue
        df['date'] = pd.to_datetime(df['單據日期'], format='%Y.%m.%d', errors='coerce')
        df = df.dropna(subset=['date', '數量']).sort_values('date').reset_index(drop=True)
        
        if df.empty: continue

        # B. 合併訂單 (7天內)
        df['temp_gap'] = df['date'].diff().dt.days.fillna(999)
        df['session_id'] = (df['temp_gap'] > 7).cumsum()
        
        df = df.groupby('session_id').agg({
            'date': 'last',
            '數量': 'sum'
        }).reset_index(drop=True)

        if len(df) < 2: continue 

        # C. 特徵工程
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['days_since_last'] = df['date'].diff().dt.days.fillna(0)
        
        df['rolling_days'] = df['days_since_last'].rolling(window=3, min_periods=1).mean()
        df['rolling_qty'] = df['數量'].rolling(window=3, min_periods=1).mean()
        
        df['target_days'] = df['date'].shift(-1).diff().dt.days.shift(-1)
        df['target_qty'] = df['數量'].shift(-1)

        train_data = df[df['year'] >= 2022].copy()
        if len(train_data) < 5: train_data = df.tail(10).copy()
        
        train_df = train_data.dropna(subset=['target_days', 'target_qty']).copy()
        features = ['數量', 'days_since_last', 'month', 'rolling_days', 'rolling_qty']
        last_row = df.tail(1).copy()

        # D. 混合預測
        sample_count = len(train_df)
        
        if sample_count < 5:
            p_days_1 = df['days_since_last'].median()
            p_qty_1 = df['數量'].median()
            confidence_label = "低 (採統計中位數)"
        else:
            train_df.loc[:, 'weight'] = train_df['year'].apply(lambda x: 1.2 if x >= 2024 else 1.0)
            model_days = RandomForestRegressor(n_estimators=100, random_state=42)
            model_qty = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model_days.fit(train_df[features], train_df['target_days'], sample_weight=train_df['weight'])
            model_qty.fit(train_df[features], train_df['target_qty'], sample_weight=train_df['weight'])
            
            p_days_1 = model_days.predict(last_row[features])[0]
            p_qty_1 = model_qty.predict(last_row[features])[0]
            confidence_label = "高 (AI 模型分析)"

        # E. 安全約束
        history_max_gap = max(df['days_since_last'].max(), 30)
        p_days_1 = min(p_days_1, 540, history_max_gap * 1.2)
        p_days_1 = max(1, int(round(p_days_1)))
        p_qty_1 = max(1, int(round(p_qty_1)))

        last_date = last_row['date'].iloc[0]
        date_1 = last_date + timedelta(days=p_days_1)
        deadline_1 = date_1 + timedelta(days=40)
        
        # F. T+2
        date_2 = date_1 + timedelta(days=p_days_1)

        final_summary.append({
            '產品編號': product_id,
            '分析信心度': confidence_label,
            '最後有效下單日': last_date.strftime('%Y-%m-%d'),
            '【預測1】預計日期': date_1.strftime('%Y-%m-%d'),
            '【預測1】預計數量': p_qty_1,
            '【預測1】追蹤期限': deadline_1.strftime('%Y-%m-%d'),
            '【預測2】預計日期': date_2.strftime('%Y-%m-%d'),
            '預測間隔參考': f"約 {p_days_1} 天下單一次",
            '數據樣本數': sample_count
        })

    status_text.text("分析完成！")
    progress_bar.empty()
    
    if final_summary:
        result_df = pd.DataFrame(final_summary)
        # 確保只選取存在的欄位
        target_cols = ['產品編號', '分析信心度', '最後有效下單日', '【預測1】預計日期', '【預測1】預計數量', '【預測1】追蹤期限', '【預測2】預計日期', '預測間隔參考', '數據樣本數']
        final_cols = [c for c in target_cols if c in result_df.columns]
        return result_df[final_cols]
    else:
        return None

# --- Excel 下載輔助函數 (修正版) ---
def convert_df_to_excel(df):
    output = io.BytesIO()
    # 使用 xlsxwriter 引擎
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='預測結果')
        
        # 嘗試自動調整欄寬 (如果不支援可移除這段 try-except)
        try:
            worksheet = writer.sheets['預測結果']
            for i, col in enumerate(df.columns):
                # 簡單計算最大寬度
                col_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(i, i, col_len)
        except:
            pass # 如果調整欄寬失敗，不影響檔案生成
            
    # 重置指標
    output.seek(0)
    return output.getvalue()

# --- 網頁主介面 ---
def main():
    st.title("📊 CRM 顧客關係管理 - 產品下單預測系統")
    st.markdown("### 自動化 AI 預測引擎")
    st.info("請上傳 Excel 檔案，系統將自動分析並產出未來兩次的建議下單日。")

    uploaded_file = st.file_uploader("📂 上傳 Excel 檔案 (.xlsx)", type=['xlsx'])

    if uploaded_file is not None:
        if st.button("🚀 開始分析", type="primary"):
            result_df = run_product_automation_v4_web(uploaded_file)
            
            if result_df is not None:
                st.success(f"成功分析 {len(result_df)} 筆產品資料！")
                st.dataframe(result_df.head())
                
                excel_data = convert_df_to_excel(result_df)
                
                st.download_button(
                    label="📥 下載預測報告",
                    data=excel_data,
                    file_name='prediction_summary_v4.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                )
            else:
                st.warning("沒有產出結果，請檢查 Excel 內容格式。")

if __name__ == "__main__":
    main()