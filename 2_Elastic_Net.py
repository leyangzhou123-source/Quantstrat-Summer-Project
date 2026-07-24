import gc
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

#Load DataFrames from Existing Parquet Files
stock_returns_df = pd.read_parquet('stock_returns.parquet')
fama_french_df = pd.read_parquet('fama_french_factors.parquet')
industry_dummies_df = pd.read_parquet('industry_dummies.parquet')
stock_characteristics_ranked_df = pd.read_parquet('stock_characteristics_ranked.parquet')
stock_characteristics_df = pd.read_parquet('stock_characteristics.parquet')
welch_goyal_df = pd.read_parquet('welch_goyal_macros.parquet')

def elastic_net_regression_chunked(
    returns, 
    ff3, 
    industry, 
    characteristics, 
    welch_goyal, 
    chunk_size=200000, 
    alpha=1e-3, 
    l1_ratio=0.5,
    max_epochs=3
):
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

    def _parse_dates(df):
        cols_to_check = [c for c in ("date", "yyyymm", "month") if c in df.columns]
        if not cols_to_check:
            raise KeyError("No date/yyyymm/month column found in DataFrame.")
        col = cols_to_check[0]
        s_str = df[col].astype(str).str.split(".").str[0].str.strip()
        parsed = pd.to_datetime(s_str, format="%Y%m", errors="coerce")
        if parsed.isna().all():
            parsed = pd.to_datetime(s_str, format="%Y%m%d", errors="coerce")
        if parsed.isna().all():
            parsed = pd.to_datetime(df[col], errors="coerce")
        df["date"] = parsed.dt.to_period("M")
        return df

    for df in (returns, ff3, industry, characteristics, welch_goyal):
        _parse_dates(df)

    returns = returns.sort_values(["permno", "date"])
    ff3_clean = ff3.drop(columns=["month", "yyyymm"], errors="ignore")

    returns = returns.merge(ff3_clean[["date", "rf"]], on="date", how="left")
    returns["target_ret"] = (
        (returns["ret"] - returns["rf"]).groupby(returns["permno"]).shift(-1)
    )
    returns = returns.dropna(subset=["target_ret"])[["permno", "date", "target_ret"]]
    returns["target_ret"] = returns["target_ret"].astype(np.float32)
    gc.collect()

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

    z_firm_signals = [c for c in target_characteristics if c in characteristics.columns]
    print(f"[*] Using {len(z_firm_signals)} firm characteristics.")

    characteristics_sliced = characteristics[["permno", "date"] + z_firm_signals].copy()
    for c in z_firm_signals:
        characteristics_sliced[c] = characteristics_sliced[c].astype(np.float32)

    del characteristics
    gc.collect()

    industry_clean = industry.drop(columns=["month", "yyyymm"], errors="ignore")
    del industry

    industry_cols = [c for c in industry_clean.columns if c not in ("permno", "date")]
    for c in industry_cols:
        industry_clean[c] = industry_clean[c].astype(np.float32)

    firm_panel = characteristics_sliced.merge(
        industry_clean, on=["permno", "date"], how="inner"
    )
    del characteristics_sliced, industry_clean
    gc.collect()

    wg_clean = welch_goyal.drop(columns=["month", "yyyymm", "rf_welch_goyal"], errors="ignore")
    del welch_goyal

    ff3_factors = ff3_clean.drop(columns=["rf"], errors="ignore")

    duplicates = (set(firm_panel.columns) & set(wg_clean.columns)) - {"date"}
    if duplicates:
        wg_clean = wg_clean.rename(columns={c: f"wg_{c}" for c in duplicates})

    macro_panel = wg_clean.merge(ff3_factors, on="date", how="inner")
    del wg_clean, ff3_factors, ff3_clean, ff3
    gc.collect()

    macro_cols = [c for c in macro_panel.columns if c != "date"]
    for c in macro_cols:
        macro_panel[c] = macro_panel[c].astype(np.float32)

    merged_features = firm_panel.merge(macro_panel, on="date", how="inner", suffixes=("", "_macro"))
    del firm_panel, macro_panel
    gc.collect()

    final_dataset = returns.merge(merged_features, on=["permno", "date"], how="inner")
    del returns, merged_features
    gc.collect()

    if final_dataset.empty:
        raise ValueError("Merged dataset is empty.")

    z_cols = [c for c in z_firm_signals if c in final_dataset.columns]
    
    industry_cols_final = [c for c in industry_cols if c in final_dataset.columns]
    if len(industry_cols_final) > 1:
        industry_cols_final = industry_cols_final[:-1]

    x_cols = [c for c in macro_cols if c in final_dataset.columns]

    n_obs = len(final_dataset)
    n_z = len(z_cols)
    n_ind = len(industry_cols_final)
    n_x = len(x_cols)
    n_features = n_ind + n_z + (n_z * n_x)

    print(f"[*] Total Features: {n_features}")
    print(f"[*] Training Elastic Net across {n_obs:,} observations...")

    # Initialize Scaler & SGD Regressor with Stable Learning Rate
    scaler = StandardScaler()
    
    model = SGDRegressor(
        penalty="elasticnet",
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        max_iter=1,
        warm_start=True,
        learning_rate="adaptive",
        eta0=1e-4,  # Small initial step size to prevent explosion
        random_state=42
    )

    # Pass 1: Fit Scaler incrementally across batches
    print("[*] Standardizing features across batches...")
    for i in range(0, n_obs, chunk_size):
        chunk_df = final_dataset.iloc[i : i + chunk_size]
        n_c = len(chunk_df)

        X_chunk = np.empty((n_c, n_features), dtype=np.float32)
        col = 0

        if n_ind:
            X_chunk[:, col:col + n_ind] = chunk_df[industry_cols_final].to_numpy(dtype=np.float32)
            col += n_ind

        Z_chunk = chunk_df[z_cols].to_numpy(dtype=np.float32)
        X_chunk[:, col:col + n_z] = Z_chunk
        col += n_z

        for x_col in x_cols:
            macro_vec = chunk_df[x_col].to_numpy(dtype=np.float32)
            X_chunk[:, col:col + n_z] = Z_chunk * macro_vec[:, None]
            col += n_z

        # Clean infs/NaNs before scaling
        np.nan_to_num(X_chunk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        scaler.partial_fit(X_chunk)

        del X_chunk, Z_chunk, chunk_df
        gc.collect()

    # Pass 2: Train Model using Scaled Chunks
    print("[*] Training Elastic Net Model...")
    for epoch in range(max_epochs):
        for i in range(0, n_obs, chunk_size):
            chunk_df = final_dataset.iloc[i : i + chunk_size]
            n_c = len(chunk_df)

            X_chunk = np.empty((n_c, n_features), dtype=np.float32)
            col = 0

            if n_ind:
                X_chunk[:, col:col + n_ind] = chunk_df[industry_cols_final].to_numpy(dtype=np.float32)
                col += n_ind

            Z_chunk = chunk_df[z_cols].to_numpy(dtype=np.float32)
            X_chunk[:, col:col + n_z] = Z_chunk
            col += n_z

            for x_col in x_cols:
                macro_vec = chunk_df[x_col].to_numpy(dtype=np.float32)
                X_chunk[:, col:col + n_z] = Z_chunk * macro_vec[:, None]
                col += n_z

            np.nan_to_num(X_chunk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            X_scaled = scaler.transform(X_chunk)
            y_chunk = chunk_df["target_ret"].to_numpy(dtype=np.float32)

            model.partial_fit(X_scaled, y_chunk)

            del X_chunk, X_scaled, Z_chunk, y_chunk, chunk_df
            gc.collect()

    print("Elastic Net Results:")

    y_preds = []
    y_trues = []

    for i in range(0, n_obs, chunk_size):
        chunk_df = final_dataset.iloc[i : i + chunk_size]
        n_c = len(chunk_df)

        X_chunk = np.empty((n_c, n_features), dtype=np.float32)
        col = 0

        if n_ind:
            X_chunk[:, col:col + n_ind] = chunk_df[industry_cols_final].to_numpy(dtype=np.float32)
            col += n_ind

        Z_chunk = chunk_df[z_cols].to_numpy(dtype=np.float32)
        X_chunk[:, col:col + n_z] = Z_chunk
        col += n_z

        for x_col in x_cols:
            macro_vec = chunk_df[x_col].to_numpy(dtype=np.float32)
            X_chunk[:, col:col + n_z] = Z_chunk * macro_vec[:, None]
            col += n_z

        np.nan_to_num(X_chunk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = scaler.transform(X_chunk)
        
        y_preds.append(model.predict(X_scaled))
        y_trues.append(chunk_df["target_ret"].to_numpy(dtype=np.float32))

        del X_chunk, X_scaled, Z_chunk, chunk_df
        gc.collect()

    y_pred = np.concatenate(y_preds)
    y_true = np.concatenate(y_trues)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    zero_weights = np.sum(np.abs(model.coef_) < 1e-5)

    print(f"Model Results:")
    print(f"  Total Features     : {n_features}")
    print(f"  Zeroed Coefficients: {zero_weights} / {n_features}")
    print(f"  R² Score           : {r2:.6f}")
    print(f"  RMSE               : {rmse:.6f}")

    return model

# Execution
model = elastic_net_regression_chunked(
    stock_returns_df,
    fama_french_df,
    industry_dummies_df,
    stock_characteristics_df,
    welch_goyal_df,
    chunk_size=200000,
    alpha=1e-3,     # Regularization strength
    l1_ratio=0.5,   # Balance of Lasso (L1) and Ridge (L2)
    max_epochs=3
)