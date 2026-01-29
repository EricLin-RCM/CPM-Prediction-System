import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta
import io

# --- 網頁設定 ---
st.set_page_config(page_title="CRM 產品預測系統", page_icon="📈", layout="wide")

# --- 1. 生成範例 Excel 的函數 (新功能) ---
def generate_example_file():
    output = io.BytesIO()
    # 建立範例資料
    data = {
        '單據日期': ['2023.01.15', '2023.02.20', '2023.04.10', '2023.06.05', '2024.01.12'],
        '數量': [100, 150, 200, 120, 300]
    }
    df_example = pd.DataFrame(data)
    
    # 使用 xlsxwriter 寫入
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # 建立兩個範例分頁，讓使用者知道可以放多個產品
        df_example.to_excel(writer, index=False, sheet_name='產品A001')
        df_example.to_excel(writer, index=False, sheet_name='產品B002')
        
        # 加入說明分頁 (可選)
        workbook = writer.book
        worksheet = writer.sheets['產品A001']
        # 設定欄寬
        worksheet.set_column('A:B', 15)
        
    output.seek(0)
    return output.getvalue()

# --- 2. 核心邏輯函數 (v4) ---
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
        target_cols = ['產品編號', '分析信心度', '最後有效下單日', '【預測1】預計日期', '【預測1】預計數量', '【預測1】追蹤期限', '【預測2】預計日期', '預測間隔參考', '數據樣本數']
        final_cols = [c for c in target_cols if c in result_df.columns]
        return result_df[final_cols]
    else:
        return None

# --- 3. Excel 下載輔助函數 ---
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='預測結果')
        try:
            worksheet = writer.sheets['預測結果']
            for i, col in enumerate(df.columns):
                col_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(i, i, col_len)
        except:
            pass
    output.seek(0)
    return output.getvalue()

# --- 4. 網頁主介面 ---
def main():
    st.title("📊 CRM 顧客關係管理 - 產品下單預測系統")
    
    # 說明區塊
    with st.expander("📖 系統使用說明 (點擊展開)"):
        st.markdown("""
        **如何使用本系統：**
        1. 下載下方的 **範例格式**。
        2. 將您的產品銷售資料填入，**每一個產品請建立一個獨立的分頁 (Sheet)**。
        3. 分頁名稱請命名為該產品的編號 (例如: P001)。
        4. 欄位必須包含：`單據日期` (格式: 2024.01.01) 與 `數量`。
        5. 上傳檔案並等待 AI 分析。
        """)

    st.markdown("---")

    # --- 新增：下載範例區塊 ---
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("1. 取得格式")
        st.markdown("請先下載範例，依照格式填入資料：")
        
        # 產生範例檔案
        example_file = generate_example_file()
        
        st.download_button(
            label="📥 下載 Excel 範例表單",
            data=example_file,
            file_name="import_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="點擊下載包含標準欄位的 Excel 範本"
        )

    with col2:
        st.subheader("2. 上傳分析")
        uploaded_file = st.file_uploader("📂 上傳填寫好的 Excel 檔案", type=['xlsx'])

    # 執行區塊
    if uploaded_file is not None:
        st.markdown("---")
        st.write("已讀取檔案，準備開始分析...")
        
        if st.button("🚀 開始執行預測分析", type="primary"):
            result_df = run_product_automation_v4_web(uploaded_file)
            
            if result_df is not None:
                st.success(f"✅ 分析完成！共處理 {len(result_df)} 筆產品資料。")
                st.dataframe(result_df.head(), use_container_width=True)
                
                excel_data = convert_df_to_excel(result_df)
                
                st.download_button(
                    label="📥 下載完整預測報告",
                    data=excel_data,
                    file_name='prediction_summary_v4.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                )
            else:
                st.error("❌ 無法產出結果。請檢查 Excel 格式是否與範例一致（需包含 '單據日期' 與 '數量'）。")

if __name__ == "__main__":
    main()