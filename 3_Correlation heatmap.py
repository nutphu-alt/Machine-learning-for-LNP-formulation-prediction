import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Load your Excel file
df = pd.read_excel("./dataset/dataset.xlsx")

# Drop rows with missing values (optional)
df = df.dropna()

# Select input and output columns
input = df[["Helper lipid (%)", "PEG lipid (%)", "Cholesterol (%)", "Ionizable lipid (%)", "N:P Ratio"]]
output = df[["MFI", "%Positive cells", "%EE", "Size (nm)", "PDI"]]

# Select only numeric columns
input_cols = input.select_dtypes(include=[np.number]).columns
output_cols = output.select_dtypes(include=[np.number]).columns

# Select only numeric columns
cols = df.select_dtypes(include=[np.number]).columns

# Initialize matrices
corr_matrix = pd.DataFrame(index=input_cols, columns=output_cols)
pval_matrix = pd.DataFrame(index=input_cols, columns=output_cols)
annotations = pd.DataFrame(index=input_cols, columns=output_cols)

# Calculate correlation and p-value
for col1 in input_cols:
    for col2 in output_cols:
        corr, pval = spearmanr(input[col1], output[col2])
        corr_matrix.loc[col1, col2] = corr
        pval_matrix.loc[col1, col2] = pval

        # Add stars based on significance level
        if pval <= 0.001:
            sig = f"{pval:.4f}***"
        elif pval <= 0.01:
            sig = f"{pval:.4f}**"
        elif pval <= 0.05:
            sig = f"{pval:.4f}*"
        else:
            sig = f"{pval:.4f}"
        # Combine correlation value and stars
        annotations.loc[col1, col2] = sig

# Convert matrices to float
corr_matrix = corr_matrix.astype(float)
pval_matrix = pval_matrix.astype(float)

# Plot heatmap
sns.heatmap(corr_matrix, annot=annotations, fmt="", cmap='coolwarm', square=True, cbar=True, vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()
