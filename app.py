import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta
import io

# --- 網頁設定 ---
st.set_page_config(page_title="CRM 智慧產品預測系統", page_icon="🧠", layout="wide")

# ==========================================
# 🧠 核心升級 1: 智慧欄位偵測設定
# ==========================================
# 定義程式看得懂的「同義詞」，無論使用者欄位叫什麼，只要在清單內都能抓到
COLUMN_MAPPING = {
    'date': [
        '單據日期', '下單日', '日期', '銷貨日期', '交易日期', '訂單日期', 
        'Date', 'Order Date', 'Txn Date'
    ],
    'qty': [
        '數量', '訂單數量', '銷貨數量', '實際出貨數量', 'Qty', 'Quantity', 
        'Amount', '銷售數量', '出貨數量'
    ],
    'product': [
        '產品編號', '品號', '品名', '料號', 'Product ID', 'Item Code', 
        'Part Number', '產品名稱', '商品代碼'
    ]
}

def find_column(df, target_type):
    """
    在 DataFrame 中尋找符合 target_type (date/qty/product) 的欄位名稱
    回傳: 找到的欄位名稱 (str) 或 None
    """
    candidates = COLUMN_MAPPING.get(target_type, [])
    # 1. 精確比對
    for col in df.columns:
        if col.strip() in candidates:
            return col
    # 2. 模糊比對 (只要欄位名稱包含關鍵字)
    for col in df.columns:
        for candidate in candidates:
            if candidate in col:
                return col
    return None

# ==========================================
# 📦 功能函數區
# ==========================================

def generate_example_file():
    """生成範例 Excel 供使用者下載"""
    output = io.BytesIO()
    data = {
        '單據日期': ['2023.01.15', '2023.02.20', '2023.04.10', '2023.06.05', '2024.01.12'],
        '數量': [100, 150, 200, 120, 300]
    }
    df_example = pd.DataFrame(data)
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_example.to_excel(writer, index=False, sheet_name='範例產品A')
        # 加入說明
        workbook = writer.book
        worksheet = writer.sheets['範例產品A']
        worksheet.set_column('A:B', 15)
        
    output.seek(0)
    return output.getvalue()

def convert_df_to_excel(df):
    """將 DataFrame 轉為 Excel binary"""
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

# ==========================================
# 🤖 核心邏輯函數 (升級版 v5)
# ==========================================
def run_product_automation_v5_web(uploaded_file):
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("正在讀取並解析 Excel 結構...")
    
    try:
        # 讀取所有分頁
        raw_sheets = pd.read_excel(uploaded_file, sheet_name=None)
    except Exception as e:
        st.error(f"檔案讀取失敗: {e}")
        return None

    # --- 🧠 核心升級 2: 自動資料結構標準化 ---
    # 目標：無論使用者上傳的是「多 Sheet 模式」還是「單 Sheet 明細模式」
    # 最終都轉換成 { '產品ID': DataFrame } 的統一格式
    
    processed_dict = {}
    
    for sheet_name, df in raw_sheets.items():
        if df.empty: continue
        
        # 1. 偵測關鍵欄位
        col_date = find_column(df, 'date')
        col_qty = find_column(df, 'qty')
        col_prod = find_column(df, 'product') # 偵測是否有產品編號欄位
        
        if not col_date or not col_qty:
            # 如果連日期或數量都找不到，就跳過這個 Sheet
            continue
            
        # 2. 判斷資料模式
        if col_prod:
            # [模式 A] 明細表模式：一張表包含多個產品 (如達宇)
            # 自動依照「產品欄位」進行拆分
            grouped = df.groupby(col_prod)
            for pid, sub_df in grouped:
                # 建立唯一的 key (避免不同 Sheet 有相同產品名覆蓋)
                unique_key = f"{str(pid).strip()}" 
                # 標準化欄位名稱供後續使用
                sub_df = sub_df.rename(columns={col_date: 'date', col_qty: '數量'})
                processed_dict[unique_key] = sub_df
        else:
            # [模式 B] 獨立分頁模式：一個 Sheet 就是一個產品 (如竟丞/舊版)
            # 使用 Sheet Name 作為產品 ID
            df = df.rename(columns={col_date: 'date', col_qty: '數量'})
            processed_dict[sheet_name] = df

    if not processed_dict:
        st.error("❌ 無法識別任何有效資料。請確認 Excel 中包含代表「日期」與「數量」的欄位。")
        return None

    # --- 開始跑預測迴圈 (邏輯同 v4) ---
    final_summary = []
    total_items = len(processed_dict)
    processed_count = 0
    
    status_text.text(f"成功識別 {total_items} 個產品，開始 AI 分析...")

    for product_id, df in processed_dict.items():
        processed_count += 1
        # 更新進度條 (每 5% 更新一次避免太頻繁)
        if processed_count % max(1, int(total_items/20)) == 0:
            progress = int((processed_count / total_items) * 100)
            progress_bar.progress(progress)
            status_text.text(f"正在分析: {product_id} ({processed_count}/{total_items})")

        # --- 以下邏輯與 v4 完全相同 ---
        
        # A. 資料清洗
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # 確保數量是數字，並過濾掉退貨 (負數) 或 0
        df['數量'] = pd.to_numeric(df['數量'], errors='coerce')
        df = df[df['數量'] > 0] 
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

# ==========================================
# 🖥️ 網頁主介面
# ==========================================
def main():
    st.title("🧠 CRM 智慧產品預測系統 (v5)")
    st.markdown("### 支援多種 Excel 格式的 AI 預測引擎")
    
    with st.expander("📖 支援的欄位格式說明 (系統會自動偵測，無需完全一致)"):
        st.markdown("""
        本系統具備**智慧欄位對照**功能，只要您的 Excel 包含以下概念的欄位即可：
        
        1. **日期欄位**：可命名為 `單據日期`, `下單日`, `日期`, `Date`, `Order Date`...
        2. **數量欄位**：可命名為 `數量`, `訂單數量`, `銷貨數量`, `Qty`, `Quantity`...
        3. **產品欄位 (選用)**：若您的 Excel 是「一張表包含所有產品明細」，請確保有 `品號`, `品名`, `產品編號` 欄位，系統會自動拆分分析。
        """)

    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("1. 取得範本")
        example_file = generate_example_file()
        st.download_button(
            label="📥 下載標準範本 (可選)",
            data=example_file,
            file_name="import_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.info("💡 您也可以直接上傳既有的 ERP 匯出檔，系統會嘗試自動識別！")

    with col2:
        st.subheader("2. 上傳分析")
        uploaded_file = st.file_uploader("📂 上傳 Excel 檔案 (.xlsx)", type=['xlsx'])

    if uploaded_file is not None:
        st.markdown("---")
        if st.button("🚀 啟動 AI 識別與預測", type="primary"):
            result_df = run_product_automation_v5_web(uploaded_file)
            
            if result_df is not None:
                st.success(f"✅ 分析完成！共處理 {len(result_df)} 筆產品預測。")
                st.dataframe(result_df.head(), use_container_width=True)
                
                excel_data = convert_df_to_excel(result_df)
                st.download_button(
                    label="📥 下載完整預測報告",
                    data=excel_data,
                    file_name='prediction_summary_v5.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                )

if __name__ == "__main__":
    main()