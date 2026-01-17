"""
This file has all the functions used during the data analysis for the Bayesian Network.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

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

"""
    Plot Metrics with Maintenance Events and Spindle Overheat for a Machine with color-coded points and maintenance lines.
    Parameters
    ----------
    df             : DataFrame with telemetry data (must have a 'spindle_overheat' column)
    maintenance_df : DataFrame with maintenance events
    machine_id     : ID of the machine to analyze (e.g., "M-D")
    features       : list of metrics (columns) to plot
    palette: dict {overheat: color}, e.g., {0:"green", 1:"red"}
"""
def plot_metrics_with_maintenance_and_overheat(df, maintenance_df, machine_id, features,
    palette={0: "green", 1: "red"}, time_col="timestamp", machine_col="machine_id",
    action_col="action_type", date_start=None, date_end=None,
    alpha_0=0.12,      # <- transparência dos 0
    alpha_1=0.95,      # <- transparência dos 1
    s_0=8,             # <- tamanho dos 0
    s_1=16             # <- tamanho dos 1 (um pouco maior p/ destacar)
):

    # Ensure features is a list
    if isinstance(features, str):
        features = [features]

    # Filter data for the specified machine
    m = df[df[machine_col] == machine_id].copy()
    m[time_col] = pd.to_datetime(m[time_col])
    m.sort_values(time_col, inplace=True)

    # Optional time filter
    if date_start is not None:
        m = m[m[time_col] >= pd.to_datetime(date_start)]
    if date_end is not None:
        m = m[m[time_col] <= pd.to_datetime(date_end)]

    # Filter maintenance data for the specified machine
    maint = maintenance_df[maintenance_df[machine_col] == machine_id].copy()
    maint[time_col] = pd.to_datetime(maint[time_col])

    # If time interval was filtered, also filter maintenance events
    if date_start is not None:
        maint = maint[maint[time_col] >= pd.to_datetime(date_start)]
    if date_end is not None:
        maint = maint[maint[time_col] <= pd.to_datetime(date_end)]

    # Create figure with subplots
    n_feat = len(features)
    fig, axes = plt.subplots(
        n_feat, 1,
        figsize=(18, 3.5 * n_feat),
        sharex=True
    )
    if n_feat == 1:
        axes = [axes]

    # garantir 0/1
    m["spindle_overheat"] = m["spindle_overheat"].astype(int)

    # máscaras
    m0 = m[m["spindle_overheat"] == 0]
    m1 = m[m["spindle_overheat"] == 1]

    for ax, feature in zip(axes, features):

        # 0 -> bem transparente (fica "de fundo")
        ax.scatter(
            m0[time_col],
            m0[feature],
            c=palette.get(0, "green"),
            s=s_0,
            alpha=alpha_0,
            zorder=1
        )

        # 1 -> por cima, mais visível
        ax.scatter(
            m1[time_col],
            m1[feature],
            c=palette.get(1, "red"),
            s=s_1,
            alpha=alpha_1,
            zorder=3
        )

        # maintenance lines
        for _, row in maint.iterrows():
            t0 = row[time_col]
            ax.axvline(t0, color="red", linestyle="--", linewidth=1, alpha=0.8, zorder=2)
            ax.text(
                t0,
                m[feature].max(),
                row[action_col],
                rotation=90,
                fontsize=8,
                color="red",
                va="top"
            )

        ax.set_ylabel(feature)
        ax.set_title(f"{feature} — Machine {machine_id}")

    axes[-1].set_xlabel("Time")

    # legend (usa o mesmo alpha para refletir o gráfico)
    overheat_handles = [
        Patch(color=palette.get(0, "green"), label="overheat = 0", alpha=alpha_0),
        Patch(color=palette.get(1, "red"),   label="overheat = 1", alpha=alpha_1),
    ]
    axes[0].legend(handles=overheat_handles, loc="upper right")

    plt.tight_layout()
    plt.show()
