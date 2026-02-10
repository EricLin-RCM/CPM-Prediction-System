import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta
import io

# --- 網頁設定 ---
st.set_page_config(page_title="CRM 智慧產品預測系統", page_icon="🧠", layout="wide")

# ==========================================
# 🧠 核心升級: 智慧欄位對照表
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
        '客戶', '客戶代號', '客戶簡稱', '客戶名稱', 'Customer', 'Client'
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
# 🧹 ERP 資料清洗核心 (新增功能)
# ==========================================
def clean_messy_erp_file(uploaded_file):
    """
    專門處理格式跑掉的 ERP 報表 (列印格式轉 Excel)
    """
    try:
        # 嘗試讀取為 CSV (很多 ERP 匯出其實是 Tab 分隔或逗號分隔)
        uploaded_file.seek(0)
        df_raw = pd.read_csv(uploaded_file, header=None)
    except:
        try:
            uploaded_file.seek(0)
            df_raw = pd.read_excel(uploaded_file, header=None)
        except:
            return None

    cleaned_rows = []
    current_date = None
    current_customer = None
    
    # 硬編碼關鍵欄位位置 (基於 prn34c.xls 分析)
    # 產品編號通常在第 5 欄 (Index 5)
    
    for i, row in df_raw.iterrows():
        # 1. 偵測日期行
        first_col = str(row[0]).strip()
        if "訂單日期" in first_col:
            val = str(row[2]).strip() # 日期通常在第 3 欄
            if val and val != 'nan':
                current_date = val.replace('.', '-') # 轉成標準格式
            continue

        # 2. 偵測資料行
        # 判斷依據: 第 5 欄有值，且不是標題
        if len(row) > 5:
            prod_id = str(row[5]).strip()
            if prod_id and prod_id != 'nan' and prod_id != "產品編號":
                
                # 處理客戶 (填補空白)
                cust = str(row[0]).strip()
                if cust and cust != 'nan':
                    current_customer = cust
                
                # 處理數量與單價 (最困難的部分：欄位會位移)
                # 策略：找到「產品名稱」後面的「單位」欄位，數值通常在單位後面
                unit_idx = -1
                prod_name_col = 9
                
                # 往後找「單位」(通常是文字且長度短)
                if len(row) > prod_name_col:
                    for c in range(prod_name_col + 1, len(row)):
                        val = str(row[c]).strip()
                        # 判斷是否為數字
                        try:
                            float(val.replace(',', ''))
                            is_num = True
                        except:
                            is_num = False
                        
                        if val and val != 'nan' and not is_num:
                            unit_idx = c
                            break
                
                qty = 0.0
                
                if unit_idx != -1:
                    # 收集單位後面的所有數字
                    nums = []
                    for c in range(unit_idx + 1, len(row)):
                        val = str(row[c]).strip()
                        try:
                            num = float(val.replace(',', ''))
                            nums.append(num)
                        except:
                            pass
                        if len(nums) >= 3: break 
                    
                    # 啟發式規則
                    if len(nums) >= 2:
                        qty = nums[1] # 通常是 [單價, 數量]
                    elif len(nums) == 1:
                        qty = nums[0]
                
                # 排除合計行
                if "合計" not in str(row.values) and "總計" not in str(row.values):
                    cleaned_rows.append({
                        '單據日期': current_date,
                        '客戶名稱': current_customer,
                        '產品編號': prod_id,
                        '數量': qty
                    })

    if not cleaned_rows:
        return None
        
    return pd.DataFrame(cleaned_rows)

# ==========================================
# 📦 輔助功能: 範本與 Excel 輸出
# ==========================================
def generate_example_file():
    output = io.BytesIO()
    data = {
        '單據日期': ['2023.01.15', '2023.02.20'],
        '客戶名稱': ['客戶A', '客戶A'],
        '產品編號': ['P001', 'P001'],
        '數量': [100, 150]
    }
    df = pd.DataFrame(data)
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
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
# 🔍 資料預檢與結構化 (整合清洗邏輯)
# ==========================================
def audit_and_process_data(uploaded_file):
    # 1. 嘗試直接讀取 (標準 Excel)
    try:
        raw_sheets = pd.read_excel(uploaded_file, sheet_name=None)
    except:
        # 如果讀失敗，可能是 CSV 或亂碼檔
        uploaded_file.seek(0)
        try:
            raw_sheets = {'Sheet1': pd.read_csv(uploaded_file)}
        except:
            raw_sheets = {} # 讀取失敗

    processed_dict = {}
    audit_info = {"total_rows": 0, "detected_columns": {}, "grouping_mode": "未知", "groups_found": 0}
    
    # 🚩 判斷是否需要啟動「ERP 清洗模式」
    # 如果讀進來第一欄有很多 NaN，或者找不到標題，很有可能是跑掉的格式
    needs_cleaning = False
    
    # 簡單檢查：如果所有 Sheet 都找不到 'date' 和 'qty'，就假設需要清洗
    valid_sheets = 0
    for _, df in raw_sheets.items():
        if find_column(df, 'date') and find_column(df, 'qty'):
            valid_sheets += 1
            
    if valid_sheets == 0:
        needs_cleaning = True
    
    if needs_cleaning:
        uploaded_file.seek(0)
        st.toast("偵測到非標準格式，正在啟動 ERP 清洗引擎...", icon="🧹")
        df_cleaned = clean_messy_erp_file(uploaded_file)
        
        if df_cleaned is not None and not df_cleaned.empty:
            # 清洗成功，將其視為標準資料繼續處理
            raw_sheets = {'Cleaned_Data': df_cleaned}
        else:
            return "❌ 無法識別檔案格式，且自動清洗失敗。請檢查檔案是否為支援的 Excel/CSV。", None, None

    # --- 以下邏輯與標準化處理相同 ---
    for sheet_name, df in raw_sheets.items():
        if df.empty: continue
        
        col_date = find_column(df, 'date')
        col_qty = find_column(df, 'qty')
        col_prod = find_column(df, 'product')
        col_cust = find_column(df, 'customer')
        
        if not col_date or not col_qty: continue
            
        audit_info["detected_columns"] = {
            "日期": col_date, "數量": col_qty, 
            "產品": col_prod, "客戶": col_cust
        }
        
        rename_map = {col_date: 'date', col_qty: '數量'}
        if col_prod: rename_map[col_prod] = 'product_id'
        if col_cust: rename_map[col_cust] = 'customer_id'
        df = df.rename(columns=rename_map)
        
        # 分組邏輯
        if col_prod and col_cust:
            audit_info["grouping_mode"] = "精準模式 (客戶 + 產品)"
            for (cid, pid), sub_df in df.groupby(['customer_id', 'product_id']):
                key = (str(cid).strip(), str(pid).strip())
                processed_dict[key] = sub_df
        elif col_prod:
            audit_info["grouping_mode"] = "產品模式"
            for pid, sub_df in df.groupby('product_id'):
                key = ("全部客戶", str(pid).strip())
                processed_dict[key] = sub_df
        else:
            audit_info["grouping_mode"] = "簡易模式"
            key = ("預設", sheet_name)
            processed_dict[key] = df
            
        audit_info["total_rows"] += len(df)

    audit_info["groups_found"] = len(processed_dict)
    if not processed_dict:
        return "❌ 找不到有效資料", None, None
        
    return "OK", processed_dict, audit_info

# ==========================================
# 🤖 預測引擎 (核心邏輯)
# ==========================================
def run_prediction_engine(processed_data):
    final_summary = []
    progress_bar = st.progress(0)
    total = len(processed_data)
    count = 0

    for (cust_id, prod_id), df in processed_data.items():
        count += 1
        progress_bar.progress(int((count / total) * 100))

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['數量'] = pd.to_numeric(df['數量'], errors='coerce')
        df = df[df['數量'] > 0].dropna(subset=['date', '數量']).sort_values('date').reset_index(drop=True)
        if df.empty: continue

        # 合併訂單 (7天)
        df['temp_gap'] = df['date'].diff().dt.days.fillna(999)
        df['session_id'] = (df['temp_gap'] > 7).cumsum()
        df = df.groupby('session_id').agg({'date': 'last', '數量': 'sum'}).reset_index(drop=True)
        if len(df) < 2: continue

        # 特徵工程
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['days_since_last'] = df['date'].diff().dt.days.fillna(0)
        df['target_days'] = df['date'].shift(-1).diff().dt.days.shift(-1)
        df['target_qty'] = df['數量'].shift(-1)
        
        train_df = df.dropna(subset=['target_days', 'target_qty']).copy()
        if len(train_df) >= 3:
            train_df = train_df[train_df['year'] >= 2022]
            if len(train_df) < 3: train_df = df.tail(10)

        last_row = df.tail(1).copy()
        
        # 混合預測 (AI + 統計)
        if len(train_df) < 5:
            p_days = df['days_since_last'].median()
            p_qty = df['數量'].median()
            conf = "低 (統計)"
        else:
            try:
                model_d = RandomForestRegressor(n_estimators=100, random_state=42)
                model_d.fit(train_df[['數量', 'days_since_last', 'month']], train_df['target_days'])
                p_days = model_d.predict(last_row[['數量', 'days_since_last', 'month']])[0]
                p_qty = df['數量'].median() # 簡化數量預測以求穩
                conf = "高 (AI)"
            except:
                p_days = df['days_since_last'].median()
                p_qty = df['數量'].median()
                conf = "低 (錯誤)"

        p_days = max(1, int(p_days))
        date_1 = last_row['date'].iloc[0] + timedelta(days=p_days)
        
        final_summary.append({
            '客戶名稱': cust_id, '產品編號': prod_id, '分析信心度': conf,
            '最後下單日': last_row['date'].iloc[0].strftime('%Y-%m-%d'),
            '【預測1】日期': date_1.strftime('%Y-%m-%d'),
            '【預測1】數量': int(p_qty),
            '歷史訂單數': len(df)
        })

    progress_bar.empty()
    if final_summary: return pd.DataFrame(final_summary)
    return None

# ==========================================
# 🖥️ 網頁主介面
# ==========================================
def main():
    st.title("🧠 CRM 智慧產品預測系統 (v7)")
    st.caption("支援功能：自動清洗 ERP 報表 • 客戶分群預測 • 智慧欄位偵測")

    with st.sidebar:
        st.header("1. 準備資料")
        ex_file = generate_example_file()
        st.download_button("📥 下載標準範本", ex_file, "template.xlsx")
        st.info("💡 提示：您也可以直接上傳從 ERP 匯出的原始報表 (如 prn34c.xls)，系統會自動嘗試整理格式。")

    st.header("2. 上傳與檢核")
    uploaded_file = st.file_uploader("上傳 Excel/CSV 檔案", type=['xlsx', 'csv', 'xls'])

    if uploaded_file:
        status, processed_data, audit_info = audit_and_process_data(uploaded_file)

        if status != "OK":
            st.error(status)
        else:
            st.success("✅ 檔案讀取成功！")
            
            # --- 資料預覽與確認區 ---
            st.subheader("🧐 資料預覽")
            c1, c2, c3 = st.columns(3)
            c1.metric("總資料筆數", audit_info["total_rows"])
            c2.metric("分析組合數", audit_info["groups_found"])
            c3.info(f"模式：{audit_info['grouping_mode']}")
            
            st.markdown("請檢查下方的**【數量】**與**【產品】**是否正確：")
            
            # 抓出前 5 筆預覽
            if processed_data:
                preview_list = []
                for k, df in list(processed_data.items())[:5]:
                    temp = df.head(2).copy()
                    temp['Group_Key'] = str(k)
                    preview_list.append(temp)
                if preview_list:
                    preview_df = pd.concat(preview_list)
                    st.dataframe(preview_df.head(10), use_container_width=True)

            if st.button("🚀 確認無誤，開始預測", type="primary"):
                result_df = run_prediction_engine(processed_data)
                if result_df is not None:
                    st.divider()
                    st.success(f"完成！共產出 {len(result_df)} 筆預測。")
                    st.dataframe(result_df, use_container_width=True)
                    excel_data = convert_df_to_excel(result_df)
                    st.download_button("📥 下載完整報告", excel_data, "prediction_v7.xlsx")
                else:
                    st.warning("⚠️ 無法產出結果 (可能原因是歷史資料不足，每項產品需至少 2 筆交易)。")

if __name__ == "__main__":
    main()