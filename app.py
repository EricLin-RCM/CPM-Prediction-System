import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta
import io

# --- 網頁設定 ---
st.set_page_config(page_title="CRM 智慧產品預測系統", page_icon="🧠", layout="wide")

# ==========================================
# 🧠 核心升級: 智慧欄位對照表 (新增客戶欄位)
# ==========================================
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
    ],
    'customer': [
        '客戶', '客戶代號', '客戶簡稱', '客戶名稱', 'Customer', 'Client', 
        'Cust ID', 'Cust Name', 'Buyer'
    ]
}

def find_column(df, target_type):
    """智慧尋找欄位名稱"""
    candidates = COLUMN_MAPPING.get(target_type, [])
    # 1. 精確比對
    for col in df.columns:
        if str(col).strip() in candidates:
            return col
    # 2. 模糊比對
    for col in df.columns:
        for candidate in candidates:
            if candidate in str(col):
                return col
    return None

# ==========================================
# 📦 功能 1: 生成標準範本 (包含四大變數)
# ==========================================
def generate_example_file():
    output = io.BytesIO()
    # 建立包含完整維度的範例
    data = {
        '客戶代號': ['C001', 'C001', 'C002', 'C001', 'C002'],
        '產品編號': ['P-1001', 'P-1001', 'P-1001', 'P-2002', 'P-2002'],
        '單據日期': ['2023.01.15', '2023.02.20', '2023.04.10', '2023.06.05', '2024.01.12'],
        '數量': [100, 150, 200, 120, 300]
    }
    df_example = pd.DataFrame(data)
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_example.to_excel(writer, index=False, sheet_name='銷售明細表')
        
        # 加入格式說明
        workbook = writer.book
        worksheet = writer.sheets['銷售明細表']
        worksheet.set_column('A:D', 15)
        
    output.seek(0)
    return output.getvalue()

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

# ==========================================
# 🔍 功能 2: 資料預檢與結構化 (Audit Phase)
# ==========================================
def audit_and_process_data(uploaded_file):
    """
    讀取檔案，偵測欄位，並將資料轉換為統一的 {key: dataframe} 格式
    回傳: (狀態訊息, 處理後的資料字典, 偵測到的欄位資訊)
    """
    try:
        raw_sheets = pd.read_excel(uploaded_file, sheet_name=None)
    except Exception as e:
        return f"❌ 檔案讀取錯誤: {e}", None, None

    processed_dict = {}
    audit_info = {
        "total_rows": 0,
        "detected_columns": {},
        "grouping_mode": "未知",
        "groups_found": 0
    }

    for sheet_name, df in raw_sheets.items():
        if df.empty: continue
        
        # 1. 偵測欄位
        col_date = find_column(df, 'date')
        col_qty = find_column(df, 'qty')
        col_prod = find_column(df, 'product')
        col_cust = find_column(df, 'customer')
        
        if not col_date or not col_qty:
            continue
            
        audit_info["detected_columns"] = {
            "日期": col_date, "數量": col_qty, 
            "產品": col_prod if col_prod else "(未偵測到 - 使用分頁名)",
            "客戶": col_cust if col_cust else "(未偵測到 - 視為單一客戶)"
        }
        
        # 標準化欄位名
        rename_map = {col_date: 'date', col_qty: '數量'}
        if col_prod: rename_map[col_prod] = 'product_id'
        if col_cust: rename_map[col_cust] = 'customer_id'
        
        df = df.rename(columns=rename_map)
        
        # 2. 資料分組邏輯 (Grouping Logic)
        if col_prod and col_cust:
            # 模式 A: 客戶 + 產品 (最精準)
            audit_info["grouping_mode"] = "精準模式 (客戶 + 產品)"
            grouped = df.groupby(['customer_id', 'product_id'])
            for (cid, pid), sub_df in grouped:
                key = (str(cid).strip(), str(pid).strip()) # Key 為 Tuple
                processed_dict[key] = sub_df
                
        elif col_prod:
            # 模式 B: 僅產品 (忽略客戶差異)
            audit_info["grouping_mode"] = "產品模式 (混合所有客戶)"
            grouped = df.groupby('product_id')
            for pid, sub_df in grouped:
                key = ("全部客戶", str(pid).strip())
                processed_dict[key] = sub_df
                
        else:
            # 模式 C: 僅分頁 (舊模式)
            audit_info["grouping_mode"] = "簡易模式 (以分頁為產品)"
            key = ("預設", sheet_name)
            processed_dict[key] = df

        audit_info["total_rows"] += len(df)

    audit_info["groups_found"] = len(processed_dict)
    
    if not processed_dict:
        return "❌ 找不到有效的 [日期] 與 [數量] 欄位，請檢查 Excel。", None, None
        
    return "OK", processed_dict, audit_info

# ==========================================
# 🤖 功能 3: AI 預測執行 (Prediction Phase)
# ==========================================
def run_prediction_engine(processed_data):
    final_summary = []
    
    # 建立進度條
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(processed_data)
    count = 0

    for (cust_id, prod_id), df in processed_data.items():
        count += 1
        if count % max(1, int(total/20)) == 0:
            progress_bar.progress(int((count / total) * 100))
            status_text.text(f"分析中... {cust_id} - {prod_id}")

        # --- 以下邏輯與 v5 核心相同 ---
        # A. 清洗
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['數量'] = pd.to_numeric(df['數量'], errors='coerce')
        df = df[df['數量'] > 0].dropna(subset=['date', '數量']).sort_values('date').reset_index(drop=True)
        if df.empty: continue

        # B. 合併訂單 (7天)
        df['temp_gap'] = df['date'].diff().dt.days.fillna(999)
        df['session_id'] = (df['temp_gap'] > 7).cumsum()
        df = df.groupby('session_id').agg({'date': 'last', '數量': 'sum'}).reset_index(drop=True)
        if len(df) < 2: continue

        # C. 特徵
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['days_since_last'] = df['date'].diff().dt.days.fillna(0)
        df['rolling_days'] = df['days_since_last'].rolling(3, min_periods=1).mean()
        df['rolling_qty'] = df['數量'].rolling(3, min_periods=1).mean()
        df['target_days'] = df['date'].shift(-1).diff().dt.days.shift(-1)
        df['target_qty'] = df['數量'].shift(-1)

        train_df = df.dropna(subset=['target_days', 'target_qty']).copy()
        if len(train_df) >= 3: # 降低門檻，有3筆就跑
            train_df = train_df[train_df['year'] >= 2022] # 只取近期
            if len(train_df) < 3: train_df = df.tail(10) # 若篩完太少，用全部

        features = ['數量', 'days_since_last', 'month', 'rolling_days', 'rolling_qty']
        last_row = df.tail(1).copy()
        sample_count = len(train_df)

        # D. 混合預測
        if sample_count < 5:
            p_days_1 = df['days_since_last'].median()
            p_qty_1 = df['數量'].median()
            conf_label = "低 (統計中位數)"
        else:
            try:
                model_d = RandomForestRegressor(n_estimators=100, random_state=42)
                model_q = RandomForestRegressor(n_estimators=100, random_state=42)
                model_d.fit(train_df[features], train_df['target_days'])
                model_q.fit(train_df[features], train_df['target_qty'])
                p_days_1 = model_d.predict(last_row[features])[0]
                p_qty_1 = model_q.predict(last_row[features])[0]
                conf_label = "高 (AI 模型)"
            except:
                p_days_1 = df['days_since_last'].median()
                p_qty_1 = df['數量'].median()
                conf_label = "低 (模型錯誤轉統計)"

        # E. 約束
        max_gap = max(df['days_since_last'].max(), 30) * 1.5
        p_days_1 = max(1, int(min(p_days_1, 540, max_gap)))
        p_qty_1 = max(1, int(p_qty_1))

        date_1 = last_row['date'].iloc[0] + timedelta(days=p_days_1)
        date_2 = date_1 + timedelta(days=p_days_1) # T+2 簡化推估

        final_summary.append({
            '客戶名稱': cust_id,
            '產品編號': prod_id,
            '分析信心度': conf_label,
            '最後下單日': last_row['date'].iloc[0].strftime('%Y-%m-%d'),
            '【預測1】日期': date_1.strftime('%Y-%m-%d'),
            '【預測1】數量': p_qty_1,
            '【預測2】日期': date_2.strftime('%Y-%m-%d'),
            '歷史樣本數': sample_count
        })

    progress_bar.empty()
    status_text.empty()
    
    if final_summary:
        return pd.DataFrame(final_summary)
    return None

# ==========================================
# 🖥️ 網頁主介面
# ==========================================
def main():
    st.title("🧠 CRM 智慧產品預測系統 (v6)")
    st.caption("支援：客戶分群預測 • 資料規格預檢 • 智慧欄位偵測")

    # --- 側邊欄：範本下載 ---
    with st.sidebar:
        st.header("1. 準備資料")
        st.markdown("請下載範本，並填入您的銷售數據。")
        ex_file = generate_example_file()
        st.download_button("📥 下載標準範本 (.xlsx)", ex_file, "import_template.xlsx")
        st.markdown("---")
        st.info("**欄位說明**：\n- **客戶/產品**：系統會依此分組。\n- **日期/數量**：核心預測變數。")

    # --- 主畫面：上傳與檢核 ---
    st.header("2. 上傳與檢核")
    uploaded_file = st.file_uploader("請上傳 Excel 檔案", type=['xlsx'])

    if uploaded_file:
        # 1. 執行資料預檢 (Data Audit)
        status, processed_data, audit_info = audit_and_process_data(uploaded_file)

        if status != "OK":
            st.error(status)
        else:
            # 2. 顯示檢核報告 (Confirmation UI)
            st.success("✅ 檔案讀取成功！請確認以下資料規格：")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("總資料筆數", audit_info["total_rows"])
            c2.metric("分析組合數 (客戶x產品)", audit_info["groups_found"])
            c3.info(f"偵測模式：{audit_info['grouping_mode']}")

            with st.expander("🔍 查看詳細欄位偵測結果", expanded=True):
                st.json(audit_info["detected_columns"])
                st.markdown("如果偵測結果正確，請點擊下方按鈕開始分析。")

            # 3. 執行分析按鈕
            if st.button("🚀 確認無誤，開始預測分析", type="primary"):
                result_df = run_prediction_engine(processed_data)
                
                if result_df is not None:
                    st.divider()
                    st.header("3. 分析結果")
                    st.success(f"完成！共產出 {len(result_df)} 筆預測結果。")
                    
                    # 呈現結果表格
                    st.dataframe(result_df.head(), use_container_width=True)
                    
                    # 下載按鈕
                    excel_data = convert_df_to_excel(result_df)
                    st.download_button(
                        "📥 下載完整預測報告 (.xlsx)",
                        excel_data,
                        "prediction_summary_v6.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.warning("⚠️ 分析完成，但因資料量不足 (每組需至少 2 筆交易)，沒有產出預測結果。")

if __name__ == "__main__":
    main()