import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

# Load data
df = pd.read_excel("./dataset/dataset.xlsx")

# Drop non-numeric or irrelevant columns
df = df.drop(columns=["Run", "non-lysed LNP", "Lysed LNP"])

# Define input and output columns
FEATURES = ["pH", "Helper lipid (%)", "PEG lipid (%)", "Cholesterol (%)", "Ionizable lipid (%)"]
TARGETS = ["MFI", "%Positive cells", "%EE", "Size (nm)", "PDI"]

X = df[FEATURES].values

# Position mapping for subplots (2 rows × 3 columns)
positions = [(0,0), (0,1), (0,2),
             (1,0), (1,1), (1,2)]

# Plot scatter plot for each target
for target in TARGETS:
    y = df[target]

    fig, ax = plt.subplots(2, 3, figsize=(12, 5))
    y_formatter = mticker.StrMethodFormatter('{x:,.0f}')
    y_formatter_PDI = mticker.StrMethodFormatter('{x:.2f}')

    for idx, feature in enumerate(FEATURES):
        i, j = positions[idx]
        z = df[feature]

        r_value, p_value = stats.spearmanr(z, y)
        r2 = r_value ** 2

        if target != "PDI":
            sns.regplot(x=feature, y=target, data=df, scatter_kws=dict(s=5, color='cornflowerblue'), line_kws=dict(lw=1, color='crimson'),
                        ax=ax[i, j])
            ax[i, j].set_title(f"{feature} VS {target}")
            ax[i, j].set_xlabel(feature)
            ax[i, j].set_ylabel(target)
            ax[i, j].yaxis.set_major_formatter(y_formatter)
            ax[i, j].annotate(f"$R^2 = {r2:.2f}$", xy=(0.05, 0.9), xycoords='axes fraction')
            if feature == "pH":
                ax[i, j].set_xticks([3, 4, 5], [3, 4, 5])
            elif feature == "PEG lipid (%)":
                ax[i, j].set_xticks([1.0, 1.50, 2.0], [1.0, 1.50, 2.0])
            elif feature == "Cholesterol (%)":
                ax[i, j].set_xticks([33.5, 38.5, 43.5], [33.5, 38.5, 43.5])
            elif feature == "Helper lipid (%)":
                ax[i, j].set_xticks([12.5, 15.0, 17.5], [12.5, 15.0, 17.5])
        else:
            sns.regplot(x=feature, y=target, data=df, scatter_kws=dict(s=5, color='cornflowerblue'), line_kws=dict(lw=1, color='crimson'),
                        ax=ax[i, j])
            ax[i, j].set_title(f"{feature} VS {target}")
            ax[i, j].set_xlabel(feature)
            ax[i, j].set_ylabel(target)
            ax[i, j].yaxis.set_major_formatter(y_formatter_PDI)
            ax[i, j].annotate(f"$R^2 = {r2:.2f}$", xy=(0.05, 0.9), xycoords='axes fraction')
            if feature == "Helper lipid (%)":
                ax[i, j].set_xticks([12.5, 15.0, 17.5], [12.5, 15.0, 17.5])
            elif feature == "PEG lipid (%)":
                ax[i, j].set_xticks([1.0, 1.50, 2.0], [1.0, 1.50, 2.0])
            elif feature == "Cholesterol (%)":
                ax[i, j].set_xticks([33.5, 38.5, 43.5], [33.5, 38.5, 43.5])
            elif feature == "pH":
                ax[i, j].set_xticks([3, 4, 5], [3, 4, 5])


    fig.delaxes(ax[1, 2]) #Remove the plot[1, 2]
    plt.tight_layout() #Adjust plot layout
    plt.savefig(f"saved file")
    plt.show()
