import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta
import io
import re

# --- 網頁設定 ---
st.set_page_config(page_title="CRM 智慧產品預測系統", page_icon="🧠", layout="wide")

# ==========================================
# 🧠 核心升級: 智慧欄位對照表
# ==========================================
COLUMN_MAPPING = {
    'date': ['單據日期', '下單日', '日期', '銷貨日期', '交易日期', '訂單日期', 'Date', 'Order Date'],
    'qty': ['數量', '訂單數量', '銷貨數量', 'Qty', 'Quantity', 'Amount'],
    'product': ['產品編號', '品號', '品名', '料號', 'Product ID'],
    'customer': ['客戶', '客戶代號', '客戶簡稱', '客戶名稱', 'Customer']
}

def find_column(df, target_type):
    candidates = COLUMN_MAPPING.get(target_type, [])
    for col in df.columns:
        if str(col).strip() in candidates: return col
    for col in df.columns:
        for candidate in candidates:
            if candidate in str(col): return col
    return None

# ==========================================
# 🧹 ERP 資料清洗核心 (v8 強力版)
# ==========================================
def try_read_content(uploaded_file):
    """
    暴力嘗試讀取檔案內容，解決 Big5/UTF-8 編碼問題
    """
    bytes_data = uploaded_file.getvalue()
    
    # 1. 嘗試常見編碼
    encodings = ['utf-8', 'cp950', 'big5', 'gbk', 'utf-16']
    
    for enc in encodings:
        try:
            # 嘗試解碼並按行切割
            content = bytes_data.decode(enc)
            lines = content.splitlines()
            return lines
        except:
            continue
    return None

def clean_messy_erp_file(uploaded_file):
    """
    v8: 純文字解析模式，不依賴 Pandas 的 read_csv，
    專門對付格式極度混亂的 ERP 報表。
    """
    lines = try_read_content(uploaded_file)
    
    if not lines:
        return None

    cleaned_rows = []
    current_date = None
    current_customer = None
    
    # 逐行解析
    for line in lines:
        # 去除引號中的逗號 (避免 CSV 分割錯誤)，簡單處理
        # 這裡假設金額裡的逗號是干擾源，先簡單移除引號
        line_clean = line.replace('"', '').replace("'", "")
        parts = line_clean.split(',')
        
        # 移除前後空白
        parts = [p.strip() for p in parts]
        
        # 如果切出來欄位太少，可能是空行
        if len(parts) < 3: continue

        # 1. 偵測日期行
        # 檢查第 0 欄是否包含 "訂單日期"
        if "訂單日期" in parts[0]:
            # 日期通常在第 2 或第 3 個位置
            for p in parts[1:5]: 
                # 簡單正則：抓 202x.xx.xx
                if re.search(r'202\d', p):
                    current_date = p.replace('.', '-').strip()
                    break
            continue

        # 2. 偵測資料行
        # 條件：第 5 欄 (Index 5) 是產品編號，且不為空，且不是標題
        if len(parts) > 6:
            prod_id = parts[5]
            
            # 過濾條件
            if prod_id and prod_id != "產品編號" and "合計" not in line and "總計" not in line:
                
                # 抓客戶 (如果第 0 欄有字，就是新客戶；沒字就沿用舊的)
                if parts[0]:
                    current_customer = parts[0]
                
                # 抓數量 (最難的部分)
                # 策略：從後面往前找，找到「單位」(MPS/KG/箱) 之後的數字
                
                qty = 0.0
                
                # 尋找單位的位置
                unit_candidates = ["MPS", "KG", "PCS", "SET", "箱", "台", "支", "一般包裝"]
                unit_idx = -1
                
                # 掃描這一行，找單位
                for idx, val in enumerate(parts):
                    if val in unit_candidates:
                        unit_idx = idx
                        break
                
                # 如果找不到常見單位，嘗試找「產品名稱」(Index 9) 後面的非數字欄位
                if unit_idx == -1 and len(parts) > 10:
                     for idx in range(10, len(parts)):
                         # 找一個長度短的非數字字串當作單位
                         if parts[idx] and not parts[idx].replace('.','').isdigit() and len(parts[idx]) < 5:
                             unit_idx = idx
                             break
                
                # 如果找到了單位，數量通常在單位後面 1~3 格內
                if unit_idx != -1:
                    potential_nums = []
                    for k in range(unit_idx + 1, min(unit_idx + 5, len(parts))):
                        val = parts[k].replace(',', '') # 去除千分位
                        try:
                            f_val = float(val)
                            potential_nums.append(f_val)
                        except:
                            pass
                    
                    # 邏輯：如果有 2 個數字，通常是 [單價, 數量] -> 取第 2 個
                    # 如果只有 1 個數字，就是數量 -> 取第 1 個
                    if len(potential_nums) >= 2:
                        qty = potential_nums[1]
                    elif len(potential_nums) == 1:
                        qty = potential_nums[0]
                
                # 如果還是沒抓到，嘗試直接抓第 20~25 欄位的數字 (Blind guess)
                if qty == 0 and len(parts) > 20:
                     try:
                         # 嘗試讀取 prn34c.xls 結構中的數量位置
                         candidate = parts[21].replace(',', '') # 假設位置
                         if candidate: qty = float(candidate)
                     except:
                         pass

                if qty > 0:
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
# 📦 輔助功能
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
# 🔍 資料預檢與結構化
# ==========================================
def audit_and_process_data(uploaded_file):
    # 嘗試讀取
    processed_dict = {}
    audit_info = {"total_rows": 0, "detected_columns": {}, "grouping_mode": "未知", "groups_found": 0}
    
    # 1. 先嘗試標準讀取
    raw_sheets = {}
    try:
        raw_sheets = pd.read_excel(uploaded_file, sheet_name=None)
    except:
        pass # 失敗也沒關係，後面會處理

    # 2. 判斷是否需要清洗
    needs_cleaning = True
    if raw_sheets:
        for _, df in raw_sheets.items():
            if find_column(df, 'date') and find_column(df, 'qty'):
                needs_cleaning = False # 有標準欄位，不用洗
                break
    
    if needs_cleaning:
        uploaded_file.seek(0)
        # st.toast("啟動強力清洗模式 (Big5/UTF-8)...", icon="🧹")
        df_cleaned = clean_messy_erp_file(uploaded_file)
        
        if df_cleaned is not None and not df_cleaned.empty:
            raw_sheets = {'Cleaned_Data': df_cleaned}
        else:
            return "❌ 檔案讀取失敗。請確認檔案不是損壞的，或嘗試將檔案另存為標準 CSV (UTF-8) 格式。", None, None

    # --- 以下標準化流程 ---
    for sheet_name, df in raw_sheets.items():
        if df.empty: continue
        
        col_date = find_column(df, 'date')
        col_qty = find_column(df, 'qty')
        col_prod = find_column(df, 'product')
        col_cust = find_column(df, 'customer')
        
        if not col_date or not col_qty: continue
            
        audit_info["detected_columns"] = {"日期": col_date, "數量": col_qty, "產品": col_prod, "客戶": col_cust}
        
        rename_map = {col_date: 'date', col_qty: '數量'}
        if col_prod: rename_map[col_prod] = 'product_id'
        if col_cust: rename_map[col_cust] = 'customer_id'
        df = df.rename(columns=rename_map)
        
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
# 🤖 預測引擎
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

        df['temp_gap'] = df['date'].diff().dt.days.fillna(999)
        df['session_id'] = (df['temp_gap'] > 7).cumsum()
        df = df.groupby('session_id').agg({'date': 'last', '數量': 'sum'}).reset_index(drop=True)
        if len(df) < 2: continue

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
        
        if len(train_df) < 5:
            p_days = df['days_since_last'].median()
            p_qty = df['數量'].median()
            conf = "低 (統計)"
        else:
            try:
                model_d = RandomForestRegressor(n_estimators=100, random_state=42)
                model_d.fit(train_df[['數量', 'days_since_last', 'month']], train_df['target_days'])
                p_days = model_d.predict(last_row[['數量', 'days_since_last', 'month']])[0]
                p_qty = df['數量'].median()
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
    st.title("🧠 CRM 智慧產品預測系統 (v8)")
    st.caption("支援功能：強力清洗 ERP 亂碼報表 • 客戶分群預測 • 智慧欄位偵測")

    with st.sidebar:
        st.header("1. 準備資料")
        ex_file = generate_example_file()
        st.download_button("📥 下載標準範本", ex_file, "template.xlsx")

    st.header("2. 上傳與檢核")
    uploaded_file = st.file_uploader("上傳 Excel/CSV (支援 prn/txt 匯出檔)", type=['xlsx', 'csv', 'xls', 'txt'])

    if uploaded_file:
        status, processed_data, audit_info = audit_and_process_data(uploaded_file)

        if status != "OK":
            st.error(status)
            st.warning("💡 提示：如果依然無法讀取，請將該檔案在 Excel 中開啟，並『另存新檔』為 CSV (UTF-8) 格式後再上傳。")
        else:
            st.success("✅ 檔案讀取成功！")
            
            st.subheader("🧐 資料預覽 (請確認數量是否正確)")
            
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
            
            st.info(f"偵測到 {audit_info['groups_found']} 組產品，共 {audit_info['total_rows']} 筆交易。")

            if st.button("🚀 確認無誤，開始預測", type="primary"):
                result_df = run_prediction_engine(processed_data)
                if result_df is not None:
                    st.divider()
                    st.success(f"完成！共產出 {len(result_df)} 筆預測。")
                    st.dataframe(result_df, use_container_width=True)
                    excel_data = convert_df_to_excel(result_df)
                    st.download_button("📥 下載完整報告", excel_data, "prediction_v8.xlsx")
                else:
                    st.warning("⚠️ 無法產出結果 (歷史資料不足)。")

if __name__ == "__main__":
    main()