# +
import anndata as ad
import pandas as pd
import plotly.express as px
from scipy.sparse import csr_matrix

import azimuthpy
from azimuthpy import umap

# -

path_to_data = "/workspaces/panhumanpy/queries/test_obj.h5ad"
adata_in = ad.read_h5ad(path_to_data)

azimuth = azimuthpy.panhuman_azimuth()
adata_out = azimuth.run(adata_in)

azimuth.umap = False


# +
plot_df = pd.DataFrame(adata_out.obsm["X_azimuth-umap"], columns=["umap1", "umap2"])
plot_df["Cell Type"] = adata_out.obs["azimuth_labels_refined"].tolist()
plot_df = plot_df.sort_values("Cell Type")

px.scatter(plot_df, x="umap1", y="umap2", color="Cell Type")
