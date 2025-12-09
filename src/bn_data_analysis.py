"""
This file has all the functions used during the data analysis for the Bayesian Network.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------------------------------------------------------

"""
    This fucntion is used in order to determine the autocorrelation of the numerical columns per machine. (temporal aspect)
    df: dataframe com machine_id e timestamp ordenados
    cols: lista de colunas numéricas
    lags: lags em número de linhas (p.ex. 1 = 5 min, 6 ~ 30 min)
"""
def autocorr_by_machine(df, cols, lags=(1, 6, 12)):
    records = []
    for mid, g in df.groupby("machine_id"):
        g = g.sort_values("timestamp")
        for col in cols:
            for lag in lags:
                ac = g[col].autocorr(lag=lag)
                records.append({
                    "machine_id": mid,
                    "variable": col,
                    "lag": lag,
                    "autocorr": ac
                })
    return pd.DataFrame(records)

# ---------------------------------------------------------------
 
"""
    Lists sensor pairs with |correlation| >= threshold.
    corr_matrix: pandas DataFrame (square)
    threshold: correlation magnitude threshold
"""

def list_strong_pairs(corr_matrix, threshold=0.9):
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_upper = corr_matrix.where(mask)

    pairs = (
        corr_upper.stack()
        .reset_index()
        .rename(columns={"level_0": "var1", "level_1": "var2", 0: "corr"})
    )
    strong = pairs[pairs["corr"].abs() >= threshold] \
             .sort_values(by="corr", key=lambda s: s.abs(), ascending=False)
    return strong

# ---------------------------------------------------------------

"""
    For a machine and (optionally) a maintenance type, plots all metrics in vertical subplots, with vertical
    lines at maintenance times.
"""
def plot_all_features_with_maintenance(telemetry_df, maintenance_df, machine_id, sensor_cols, time_col="timestamp",
    machine_col="machine_id", maintenance_type_col="action_type", maintenance_filter=None):

    # --- garantir datetime e ordenar ---
    tel = telemetry_df.copy()
    tel[time_col] = pd.to_datetime(tel[time_col])

    maint = maintenance_df.copy()
    maint[time_col] = pd.to_datetime(maint[time_col])

    # filtrar máquina
    tel_m = tel[tel[machine_col] == machine_id].sort_values(time_col)
    maint_m = maint[maint[machine_col] == machine_id].sort_values(time_col)

    if tel_m.empty:
        print(f"No telemetry data for machine {machine_id}.")
        return

    # filtrar tipo(s) de manutenção, se for o caso
    if maintenance_filter is not None:
        maint_m = maint_m[maint_m[maintenance_type_col].isin(maintenance_filter)]

    if maint_m.empty:
        print(f"No maintenance events for machine {machine_id} with the selected filter.")
        return

    event_times = maint_m[time_col].values
    event_types = maint_m[maintenance_type_col].values

    # --- subplots: um por métrica ---
    n_feats = len(sensor_cols)
    fig, axes = plt.subplots(
        n_feats, 1,
        figsize=(20, 2.3 * n_feats),
        sharex=True
    )

    if n_feats == 1:
        axes = [axes]

    for ax, feat in zip(axes, sensor_cols):
        if feat not in tel_m.columns:
            ax.set_visible(False)
            continue

        sns.lineplot(data=tel_m, x=time_col, y=feat, ax=ax)

        # linhas de manutenção
        ymax = tel_m[feat].max()
        for t0, mtype in zip(event_times, event_types):
            ax.axvline(t0, color="red", linestyle="--", alpha=0.6)
            ax.text(
                t0, ymax,
                mtype,
                rotation=90,
                verticalalignment="bottom",
                fontsize=7,
                color="red"
            )

        ax.set_ylabel(feat)

    title = f"Telemetry and maintenance events — Machine {machine_id}"
    if maintenance_filter is not None:
        title += f" — types: {', '.join(map(str, maintenance_filter))}"
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------

"""
    For one machine and one telemetry feature:
      1) compute mean value in a window before and after each maintenance event
         (optionally filtered by maintenance action type);
      2) print a per-event summary and return a DataFrame with the impacts;
      3) plot the time series with maintenance events marked.

    Parameters
    ----------
    telemetry_df : DataFrame
        Telemetry data (must contain machine_col, time_col and the feature).
    maintenance_df : DataFrame
        Maintenance log (must contain machine_col, time_col, maintenance_type_col).
    machine_id : str
        Machine to analyse (e.g. "M-A").
    feature : str
        Telemetry variable to analyse (e.g. "coolant_flow", "vibration_rms").
    before_hours, after_hours : int or float
        Time window before/after each maintenance event to compute means.
    split_by : {"month", "week", None}
        If not None, plots are split into subplots by calendar month or week.
    maintenance_filter : list or None
        If provided, only these maintenance types are considered in the analysis.
    min_points_before, min_points_after : int
        Minimum number of telemetry points required in the before/after window
        to compute an impact for that event.
"""
def plot_machine_feature_with_maintenance_and_impact(
    telemetry_df,
    maintenance_df,
    machine_id,
    feature,
    time_col="timestamp",
    machine_col="machine_id",
    maintenance_type_col="action_type",   # adjust to your column name
    before_hours=24,
    after_hours=24,
    split_by=None,                        # None, "month", or "week"
    maintenance_filter=None,              # e.g. ["clean_filter", "replace_bearing"]
    min_points_before=5,
    min_points_after=5,
):

    # --- Copy and ensure datetime ---
    telemetry = telemetry_df.copy()
    telemetry[time_col] = pd.to_datetime(telemetry[time_col])

    maintenance = maintenance_df.copy()
    maintenance[time_col] = pd.to_datetime(maintenance[time_col])

    # --- Filter by machine ---
    m_data = telemetry[telemetry[machine_col] == machine_id].sort_values(time_col)
    m_events = maintenance[maintenance[machine_col] == machine_id].sort_values(time_col)

    if m_data.empty:
        print(f"No telemetry data for machine {machine_id}.")
        return None

    # Optional filter by maintenance type(s)
    if maintenance_filter is not None:
        m_events = m_events[m_events[maintenance_type_col].isin(maintenance_filter)]

    if m_events.empty:
        print(f"No maintenance events for machine {machine_id} with the selected filter.")
        return None

    # ---------- IMPACT CALCULATION BEFORE / AFTER ----------
    impacts = []

    print(f"\nImpact of maintenance on machine {machine_id} for feature '{feature}':")

    for _, ev in m_events.iterrows():
        t0 = ev[time_col]
        maint_type = ev[maintenance_type_col]

        before_start = t0 - pd.Timedelta(hours=before_hours)
        after_end    = t0 + pd.Timedelta(hours=after_hours)

        mask_before = (m_data[time_col] >= before_start) & (m_data[time_col] < t0)
        mask_after  = (m_data[time_col] > t0) & (m_data[time_col] <= after_end)

        vals_before = m_data.loc[mask_before, feature]
        vals_after  = m_data.loc[mask_after, feature]

        # Only compute impact if we have enough data points on both sides
        if len(vals_before) < min_points_before or len(vals_after) < min_points_after:
            print(f"- {t0} | {maint_type}: insufficient data before/after ("
                  f"{len(vals_before)} / {len(vals_after)} points)")
            continue

        mean_before = vals_before.mean()
        mean_after  = vals_after.mean()
        delta = mean_after - mean_before
        delta_pct = 100 * delta / abs(mean_before) if mean_before != 0 else float("nan")

        print(f"- {t0} | {maint_type}")
        print(f"    mean {feature} over {before_hours}h BEFORE : {mean_before:.3f}")
        print(f"    mean {feature} over {after_hours}h AFTER  : {mean_after:.3f}")
        print(f"    Δ absolute: {delta:+.3f}   Δ relative: {delta_pct:+.1f}%")

        impacts.append({
            "machine_id": machine_id,
            "maintenance_time": t0,
            "maintenance_type": maint_type,
            "feature": feature,
            "mean_before": mean_before,
            "mean_after": mean_after,
            "delta": delta,
            "delta_pct": delta_pct,
            "n_before": len(vals_before),
            "n_after": len(vals_after),
        })

    impact_df = pd.DataFrame(impacts)
    if impact_df.empty:
        print("No events with sufficient data to compute impact.")
    else:
        print("\nSummary by maintenance type:")
        print(
            impact_df
            .groupby("maintenance_type")[["delta", "delta_pct"]]
            .mean()
            .round(3)
        )

    # ---------- PREPARATION FOR PLOTTING ----------
    plot_data = m_data.copy()

    if split_by == "month":
        plot_data["period"] = plot_data[time_col].dt.to_period("M")
        period_label = "month"
    elif split_by == "week":
        plot_data["period"] = plot_data[time_col].dt.to_period("W")
        period_label = "week"
    else:
        plot_data["period"] = "all"
        period_label = None

    periods = sorted(plot_data["period"].unique(), key=lambda x: str(x))

    # ---------- CREATE SUBPLOTS ----------
    n_periods = len(periods)
    fig, axes = plt.subplots(
        n_periods, 1,
        figsize=(20, 3 * n_periods),
        sharey=True
    )
    if n_periods == 1:
        axes = [axes]

    for ax, p in zip(axes, periods):
        sub = plot_data[plot_data["period"] == p]

        sns.lineplot(data=sub, x=time_col, y=feature, ax=ax, label=feature)

        # Maintenance events within this period
        if period_label is None:
            evs_in_period = m_events
        else:
            if split_by == "month":
                evs_in_period = m_events[m_events[time_col].dt.to_period("M") == p]
            elif split_by == "week":
                evs_in_period = m_events[m_events[time_col].dt.to_period("W") == p]
            else:
                evs_in_period = m_events

        # Mark maintenance events
        if not evs_in_period.empty:
            ymax = sub[feature].max()
            for _, ev in evs_in_period.iterrows():
                t0 = ev[time_col]
                maint_type = ev[maintenance_type_col]

                ax.axvline(t0, color="red", linestyle="--", alpha=0.7)
                ax.text(
                    t0,
                    ymax,
                    maint_type,
                    rotation=90,
                    verticalalignment="bottom",
                    fontsize=8,
                    color="red"
                )

        title = f"{feature} — Machine {machine_id}"
        if period_label is not None:
            title += f" — {period_label}: {p}"
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel(feature)

    plt.tight_layout()
    plt.show()

    return impact_df

# ---------------------------------------------------------------
# ----------------- Removed from the project --------------------
# ---------------------------------------------------------------


"""
    Plot raw time series and smoothed versions for different smoothing windows.

    Parameters
    ----------
    df : DataFrame
        Dataset containing telemetry data.
    machine_id : str
        Machine ID to filter, e.g., "M-D".
    feature : str
        Column name to smooth, e.g., "spindle_temp".
    windows : dict
        Example: {"15m": 3, "1h": 12, "6h": 72, "24h": 288}
    date_start : str or None
        Optional start date filter.
    date_end : str or None
        Optional end date filter.
    time_col : str
        Name of timestamp column.
    machine_col : str
        Name of machine ID column.
"""
def plot_smoothing_windows(df, machine_id, feature, windows, date_start=None, date_end=None, 
    time_col="timestamp", machine_col="machine_id"):

    # Filter machine
    m = df[df[machine_col] == machine_id].copy()

    # Optional date filtering
    if date_start is not None:
        m = m[m[time_col] >= pd.to_datetime(date_start)]
    if date_end is not None:
        m = m[m[time_col] <= pd.to_datetime(date_end)]

    # Compute smoothed columns
    for label, w in windows.items():
        smooth_col = f"{feature}_smooth_{label}"
        m[smooth_col] = (
            m[feature]
            .rolling(window=w, min_periods=1)
            .mean()
        )

    # Plot raw series
    plt.figure(figsize=(14, 4))
    sns.lineplot(data=m, x=time_col, y=feature)
    plt.title(f"Raw {feature} over time — {machine_id}")
    plt.xlabel("Time")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()

    # Plot each smoothing window separately
    for label, w in windows.items():
        smooth_col = f"{feature}_smooth_{label}"

        plt.figure(figsize=(14, 4))
        sns.lineplot(data=m, x=time_col, y=feature, alpha=0.3, label="raw")
        sns.lineplot(data=m, x=time_col, y=smooth_col, label=f"smooth {label}")

        plt.title(f"{feature} — smoothing window {label} ({w} points) — {machine_id}")
        plt.xlabel("Time")
        plt.ylabel(feature)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return m  # returns dataframe with all smooth columns added

# ---------------------------------------------------------------

"""
    Applies smoothing (rolling mean) by machine to the specified features
    and returns a DataFrame with new columns *_smooth_<label>.

    df           : Original DataFrame
    features     : List of numeric columns to smooth, e.g., ["spindle_temp", "vibration_rms"]
                   (also accepts a single string)
    windows      : dict with {label: window_size}, e.g., {"1h": 12, "24h": 288}
    time_col     : name of the timestamp column
    machine_col  : name of the machine ID column
    plot_machine : if not None, ID of the machine for which to show plots
    plot_feature : if not None, feature for which to show plots
    date_start   : optional start date filter (string or Timestamp)
    date_end     : optional end date filter (string or Timestamp)
"""
def smooth_features_by_machine(df, features, windows, time_col="timestamp", machine_col="machine_id",
    plot_machine=None, plot_feature=None, date_start=None, date_end=None):

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([machine_col, time_col])

    # ensure features is a list
    if isinstance(features, str):
        features = [features]

    # Apply smoothing by machine to each feature
    for feature in features:
        for label, w in windows.items():
            smooth_col = f"{feature}_smooth_{label}"
            df[smooth_col] = (
                df.groupby(machine_col)[feature]
                  .transform(lambda x: x.rolling(window=w, min_periods=1).mean())
            )

    # Optional plots for a machine + feature
    if plot_machine is not None and plot_feature is not None:
        m = df[df[machine_col] == plot_machine].copy()

        # temporal filter only for plots (does not affect the returned df)
        if date_start is not None:
            m = m[m[time_col] >= pd.to_datetime(date_start)]
        if date_end is not None:
            m = m[m[time_col] <= pd.to_datetime(date_end)]

        # raw plot
        plt.figure(figsize=(14, 4))
        sns.lineplot(data=m, x=time_col, y=plot_feature)
        plt.title(f"Raw {plot_feature} over time — {plot_machine}")
        plt.xlabel("Time")
        plt.ylabel(plot_feature)
        plt.tight_layout()
        plt.show()

        # plots by window
        for label, w in windows.items():
            smooth_col = f"{plot_feature}_smooth_{label}"

            plt.figure(figsize=(14, 4))
            sns.lineplot(data=m, x=time_col, y=plot_feature, alpha=0.3, label="raw")
            sns.lineplot(data=m, x=time_col, y=smooth_col, label=f"smooth {label}")

            plt.title(f"{plot_feature} — smoothing window {label} ({w} points) — {plot_machine}")
            plt.xlabel("Time")
            plt.ylabel(plot_feature)
            plt.legend()
            plt.tight_layout()
            plt.show()

    # returns the complete df with the smoothed columns
    return df
