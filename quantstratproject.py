import gc
import numpy as np
import pandas as pd

#Load DataFrames from Existing Parquet Files
stock_returns_df = pd.read_parquet('stock_returns.parquet')
fama_french_df = pd.read_parquet('fama_french_factors.parquet')
industry_dummies_df = pd.read_parquet('industry_dummies.parquet')
stock_characteristics_ranked_df = pd.read_parquet('stock_characteristics_ranked.parquet')
stock_characteristics_df = pd.read_parquet('stock_characteristics.parquet')
welch_goyal_df = pd.read_parquet('welch_goyal_macros.parquet')

#Function to Perform Chunked OLS Regression
def ols_regression_chunked(returns, ff3, industry, characteristics, welch_goyal, chunk_size=200000, l2_penalty=1e-4):
    #Standardize Each DataFrame
    def _standardize(df):
        if df.index.name is not None and any(
            k in str(df.index.name).lower() for k in ("date", "month", "yyyymm")
        ):
            df = df.reset_index()
        df.columns = df.columns.astype(str).str.strip().str.lower()
        return df

    returns = _standardize(returns.copy())
    ff3 = _standardize(ff3.copy())
    industry = _standardize(industry.copy())
    characteristics = _standardize(characteristics.copy())
    welch_goyal = _standardize(welch_goyal.copy())

    #Parse Date Columns & Convert to Same Format
    def _parse_dates(df):
        cols_to_check = [c for c in ("date", "yyyymm", "month") if c in df.columns]
        if not cols_to_check:
            raise KeyError("No date/yyyymm/month column found in DataFrame.")
        col = cols_to_check[0]
        s_str = df[col].astype(str).str.split(".").str[0].str.strip()
        parsed = pd.to_datetime(s_str, format="%Y%m", errors="coerce")
        #Try Other Formats if Initial Parsing Fails
        if parsed.isna().all():
            parsed = pd.to_datetime(s_str, format="%Y%m%d", errors="coerce")
        if parsed.isna().all():
            parsed = pd.to_datetime(df[col], errors="coerce")
        df["date"] = parsed.dt.to_period("M")
        return df

    #Parse Dates for All DataFrames
    for df in (returns, ff3, industry, characteristics, welch_goyal):
        _parse_dates(df)

    #Sort Returns & Merge with FF3 Rates
    returns = returns.sort_values(["permno", "date"])
    ff3_clean = ff3.drop(columns=["month", "yyyymm"], errors="ignore")

    returns = returns.merge(ff3_clean[["date", "rf"]], on="date", how="left")
    returns["target_ret"] = (
        (returns["ret"] - returns["rf"]).groupby(returns["permno"]).shift(-1)
    )
    returns = returns.dropna(subset=["target_ret"])[["permno", "date", "target_ret"]]
    returns["target_ret"] = returns["target_ret"].astype(np.float32)
    gc.collect()

    #Define Target Characteristics to Use in Regression
    target_characteristics = [
        'absacc', 'acc', 'aeavol', 'age', 'agr', 'baspread', 'beta', 'betasq',
        'bm', 'bm_ia', 'cash', 'cashdebt', 'cashpr', 'cfp', 'cfp_ia', 'chatoia',
        'chcsho', 'chempia', 'chinv', 'chmom', 'chpmia', 'chtx', 'cinvest',
        'convind', 'divi', 'divo', 'dolvol', 'dy', 'ear', 'egr', 'ep', 'gma',
        'grcapx', 'grltnoa', 'herf', 'hire', 'idiovol', 'ill', 'indmom',
        'invest', 'lev', 'lgr', 'maxret', 'mom12m', 'mom1m', 'mom36m',
        'mom6m', 'ms', 'mvel1', 'mve_ia', 'nincr', 'operprof', 'orgcap',
        'pctacc', 'pricedelay', 'ps', 'rd', 'rd_mve', 'rd_sale',
        'realestate', 'retvol', 'roaq', 'roeq', 'rsup', 'saleinv',
        'sgr', 'sin', 'sp', 'std_dolvol', 'std_turn', 'tang',
        'turn', 'zerotrade'
    ]

    #Filter for Relevant Firm Characteristics
    z_firm_signals = [c for c in target_characteristics if c in characteristics.columns]
    print(f"[*] Using {len(z_firm_signals)} firm characteristics.")

    #Slice Characteristics DataFrame
    characteristics_sliced = characteristics[["permno", "date"] + z_firm_signals].copy()
    for c in z_firm_signals:
        characteristics_sliced[c] = characteristics_sliced[c].astype(np.float32)

    del characteristics
    gc.collect()

    #Clean Industry DataFrame & Columns
    industry_clean = industry.drop(columns=["month", "yyyymm"], errors="ignore")
    del industry

    industry_cols = [c for c in industry_clean.columns if c not in ("permno", "date")]
    for c in industry_cols:
        industry_clean[c] = industry_clean[c].astype(np.float32)

    #Merge Firm Characteristics with Industry Dummies
    firm_panel = characteristics_sliced.merge(
        industry_clean, on=["permno", "date"], how="inner"
    )
    del characteristics_sliced, industry_clean
    gc.collect()

    #Clean Welch-Goyal Macroeconomic DataFrame
    wg_clean = welch_goyal.drop(columns=["month", "yyyymm", "rf_welch_goyal"], errors="ignore")
    del welch_goyal

    ff3_factors = ff3_clean.drop(columns=["rf"], errors="ignore")

    #Check for Duplicate Columns Between Firm Panel and Macro Panel
    duplicates = (set(firm_panel.columns) & set(wg_clean.columns)) - {"date"}
    if duplicates:
        wg_clean = wg_clean.rename(columns={c: f"wg_{c}" for c in duplicates})

    #Merge Macro Panel with Fama-French Factors
    macro_panel = wg_clean.merge(ff3_factors, on="date", how="inner")
    del wg_clean, ff3_factors, ff3_clean, ff3
    gc.collect()

    macro_cols = [c for c in macro_panel.columns if c != "date"]
    for c in macro_cols:
        macro_panel[c] = macro_panel[c].astype(np.float32)

    #Merge Firm Panel with Macro Panel
    merged_features = firm_panel.merge(macro_panel, on="date", how="inner", suffixes=("", "_macro"))
    del firm_panel, macro_panel
    gc.collect()

    #Merge Returns with Merged Features to Create Final Dataset
    final_dataset = returns.merge(merged_features, on=["permno", "date"], how="inner")
    del returns, merged_features
    gc.collect()

    if final_dataset.empty:
        raise ValueError("Merged dataset is empty.")

    z_cols = [c for c in z_firm_signals if c in final_dataset.columns]
    
    #Filter Industry Columns for Existing Columns in Final Dataset
    industry_cols_final = [c for c in industry_cols if c in final_dataset.columns]
    if len(industry_cols_final) > 1:
        industry_cols_final = industry_cols_final[:-1]

    x_cols = [c for c in macro_cols if c in final_dataset.columns]

    #Count Observations and Features
    n_obs = len(final_dataset)
    n_z = len(z_cols)
    n_ind = len(industry_cols_final)
    n_x = len(x_cols)
    n_features = 1 + n_ind + n_z + (n_z * n_x)

    #Initialize Accumulators for XtX and Xty
    XtX = np.zeros((n_features, n_features), dtype=np.float64)
    Xty = np.zeros((n_features,), dtype=np.float64)
    sum_y = 0.0
    sum_y2 = 0.0

    #Process Data in Chunks to Avoid Memory Issues
    for i in range(0, n_obs, chunk_size):
        #Create DataFrame Chunk for Current Iteration
        chunk_df = final_dataset.iloc[i : i + chunk_size]
        n_c = len(chunk_df)

        X_chunk = np.empty((n_c, n_features), dtype=np.float32)
        
        col = 0
        X_chunk[:, col] = 1.0  
        col += 1

        #Add Industry Dummies to Feature Matrix
        if n_ind:
            X_chunk[:, col:col + n_ind] = chunk_df[industry_cols_final].to_numpy(dtype=np.float32)
            col += n_ind

        #Add Firm Characteristics to Feature Matrix
        Z_chunk = chunk_df[z_cols].to_numpy(dtype=np.float32)
        X_chunk[:, col:col + n_z] = Z_chunk
        col += n_z

        #Add Interaction Terms Between Firm Characteristics and Macro Variables
        for x_col in x_cols:
            macro_vec = chunk_df[x_col].to_numpy(dtype=np.float32)
            X_chunk[:, col:col + n_z] = Z_chunk * macro_vec[:, None]
            col += n_z

        y_chunk = chunk_df["target_ret"].to_numpy(dtype=np.float64)

        #Accumulate XtX and Xty for Current Chunk
        XtX += X_chunk.T @ X_chunk
        Xty += X_chunk.T @ y_chunk

        sum_y += np.sum(y_chunk)
        sum_y2 += np.sum(y_chunk ** 2)

        del X_chunk, Z_chunk, y_chunk, chunk_df
        gc.collect()

    del final_dataset
    gc.collect()

    #Print Results 
    print(f"Accumulated Matrix Size : {n_features} x {n_features}")

    #Add L2 Regularization to Matrix
    diag_indices = np.arange(1, n_features)
    XtX[diag_indices, diag_indices] += l2_penalty

    #Use Try-Except Block to Handle Potential Singular Matrix Issues
    #Need to Include to Avoid Singular Matrix Error
    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        print("[!] Matrix near-singular. Falling back to SVD Pseudo-Inverse...")
        beta = np.linalg.lstsq(XtX, Xty, rcond=1e-5)[0]

    #Compute R² and RMSE
    y_mean = sum_y / n_obs
    ss_tot = sum_y2 - n_obs * (y_mean ** 2)
    ss_res = beta.T @ XtX @ beta - 2 * (beta.T @ Xty) + sum_y2
    
    r2 = 1.0 - (ss_res / ss_tot)
    rmse = np.sqrt(max(0.0, ss_res / n_obs))

    print(f"Model Results:")
    print(f"  Features     : {n_features}")
    print(f"  R² Score : {r2:.6f}")
    print(f"  RMSE     : {rmse:.6f}")

    return beta

#Execution
results = ols_regression_chunked(
    stock_returns_df,
    fama_french_df,
    industry_dummies_df,
    stock_characteristics_df,
    welch_goyal_df,
    chunk_size=200000,
    l2_penalty=1e-4  
)

"""
*********OUTPUT*********
[*] Using 73 firm characteristics.

Accumulated Matrix Size : 1170 x 1170
Model Results:
  Features     : 1170
  R² Score : 0.012969
  RMSE     : 0.171215
"""
