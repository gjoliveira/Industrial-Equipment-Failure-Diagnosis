"""
This file has all the functions used during the data preprocessing for the Bayesian Network.
"""

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from matplotlib.patches import Patch
from sklearn.tree import DecisionTreeClassifier, export_text

# ---------------------------------------------------------------

"""
    Apply GMM Clustering to Dataset, using Specified Features
    
    Parameters
    ----------
    df : DataFrame
        Original Dataset
    features : list of str
        Columns to use for clustering.
    n_clusters : int
        NNumber of clusters for the GMM.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    df_out : DataFrame
        A copy of the original dataset with an extra column: 'cluster'
    centres_original : array
        Cluster centers (in the original feature scale)
    gmm_model : GaussianMixture
        The trained model (optional, in case you want to use it later)
    scaler : StandardScaler
        The trained scaler (in case you need to transform new data)
"""
def apply_gmm_clustering(df, features, n_clusters=3, random_state=42):
    df_out = df.copy()
    
    # Extract features
    X = df_out[features].copy()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train GMM
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        random_state=random_state
    )
    gmm.fit(X_scaled)

    # Predict clusters
    df_out["cluster"] = gmm.predict(X_scaled)

    # Get cluster centers in original scale
    centres_original = scaler.inverse_transform(gmm.means_)
    print("Cluster centres in original units:")
    print(centres_original)

    return df_out, centres_original, gmm, scaler

# ---------------------------------------------------------------

"""
    Creates a pairplot of features colored by clusters.

    Parameters
    ----------
    df : DataFrame
        Dataset containing the features and the cluster column.
    features : list of str
        List of features to include in the pairplot.
    cluster_palette : dict
        Dictionary cluster → color. Ex: {0:"gold", 1:"red", 2:"green"}
    cluster_col : str
        Name of the cluster column in the dataset.
"""
def plot_cluster_pairplot(df, features, cluster_palette, cluster_col="cluster", title="Pairplot of clusters"):
    # garantir que a coluna de cluster está incluída
    features_to_plot = features + [cluster_col]

    sns.pairplot(
        df[features_to_plot],
        hue=cluster_col,
        corner=True,
        diag_kind="kde",
        palette=cluster_palette
    )

    plt.suptitle(title, y=1.02)
    plt.show()

# ---------------------------------------------------------------

"""
    Creates a 3D scatter plot of 3 features colored by clusters.

    Parameters
    ----------
    df : DataFrame
        Dataset containing the features and the clusters.
    features : list of str (length 3)
        The 3 features to use as X, Y, and Z axes.
    cluster_palette : dict
        Cluster → color map. Ex: {0:"gold", 1:"red", 2:"green"}
    cluster_col : str
        Name of the column containing the cluster.
"""
def plot_cluster_3d(df, features, cluster_palette, cluster_col="cluster", title="3D view of clusters"):

    if len(features) != 3:
        raise ValueError("The list 'features' must contain exactly 3 variables for the 3D plot.")

    # Maps clusters → colors
    point_colors = df[cluster_col].map(cluster_palette)

    # create figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # scatter 3D
    ax.scatter(
        df[features[0]],
        df[features[1]],
        df[features[2]],
        c=point_colors,
        alpha=0.6
    )

    # labels
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])
    plt.title(title)

    plt.show()

# ---------------------------------------------------------------

"""
    Plot Metrics with Maintenance Events and Clusters for a Machine with color-coded points and maintenance lines.
    Parameters
    ----------
    df             : DataFrame with telemetry data (must have a 'cluster' column)
    maintenance_df : DataFrame with maintenance events
    machine_id     : ID of the machine to analyze (e.g., "M-D")
    features       : list of metrics (columns) to plot
    cluster_palette: dict {cluster: color}, e.g., {0:"gold", 1:"red", 2:"green"}
"""
def plot_metrics_with_maintenance_and_clusters(df, maintenance_df, machine_id, features, cluster_palette, time_col="timestamp",
    machine_col="machine_id", action_col="action_type", date_start=None, date_end=None):

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

    # colors mapped by cluster
    point_colors = m["cluster"].map(cluster_palette)

    for ax, feature in zip(axes, features):
        # scatter raw points, colored by cluster
        ax.scatter(
            m[time_col],
            m[feature],
            c=point_colors,
            s=8,
            alpha=0.7
        )

        # if there is any smooth column for this feature, plot it on top
        smooth_cols = [c for c in m.columns if c.startswith(feature + "_smooth")]
        if smooth_cols:
            # use the last one (usually the largest window)
            best_smooth = sorted(smooth_cols)[-1]
            sns.lineplot(
                data=m,
                x=time_col,
                y=best_smooth,
                color="black",
                linewidth=2,
                ax=ax,
                label="smooth"
            )

        # maintenance lines
        for _, row in maint.iterrows():
            t0 = row[time_col]
            ax.axvline(t0, color="red", linestyle="--", linewidth=1, alpha=0.8)
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

    # legend for clusters (once only)
    cluster_handles = [
        Patch(color=color, label=f"cluster {cl}")
        for cl, color in cluster_palette.items()
    ]
    axes[0].legend(handles=cluster_handles + [Patch(color="black", label="smooth")],
                   loc="upper right")

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------


"""
    Extract Decision Rules Explaining Clusters Using a Surrogate Decision Tree Model

    Parameters
    ----------
    df : DataFrame
        Dataset with features and the cluster column.
    features : list of str
        Features used to explain the clusters.
    cluster_col : str
        Name of the cluster column in the dataset.
    max_depth : int
        Maximum depth of the tree (controls complexity of the rules).
    min_samples_leaf : int
        Minimum number of samples in a leaf.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    rules : str
        Decision rules in text format.
    tree_model : DecisionTreeClassifier
        Trained model (if necessary to inspect it later).
"""
def extract_cluster_rules(df, features, cluster_col="cluster", max_depth=3, min_samples_leaf=50, random_state=42):
    
    # Prepare data (X and y)
    X = df[features]
    y = df[cluster_col]

    # Train surrogate decision tree model
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    tree.fit(X, y)

    # Extract rules
    rules = export_text(tree, feature_names=features)

    # Print rules
    print("\n=========== Surrogate Model Rules ===========\n")
    print(rules)
    print("==============================================\n")

    return rules, tree


# ---------------------------------------------------------------
