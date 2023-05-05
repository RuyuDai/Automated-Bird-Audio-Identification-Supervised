import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

def get_boxplot_by_species(total_pcp, overlap_percent, sort_by = 'median'):
    fig, axs = plt.subplots(nrows=5, ncols=4, figsize=(16, 15))
    species_list = total_pcp['species'].drop_duplicates().to_list()
    species_to_int_dict = {y:x for y,x in zip(species_list,range(len(species_list)))}
    total_pcp_int = total_pcp.drop(columns=['segment'])
    total_pcp_int['species'] = total_pcp_int['species'].apply(lambda elem: species_to_int_dict[elem])
    for features_on_y,ax in zip(total_pcp_int.drop(columns='species').columns,axs.ravel()):
        df_plot = pd.DataFrame({col:vals[features_on_y] for col, vals in total_pcp_int.groupby('species')})
        index_provider = df_plot.median().sort_values(ascending=False)
        df_plot[index_provider.index].boxplot(rot=0, ax=ax)
        # ax.set_xticks(rotation=45, fontsize=12)
        # ax.set_yticks(fontsize=12)
        ax.set_title(features_on_y)
        if ax==axs.ravel()[-1]:
            ax.axis('off')
    fig.tight_layout()
    fig.savefig('features-%s.png'%overlap_percent)



def get_scree_plot(total_pcp, plot=True):
    a= total_pcp.iloc[:, 0:13].copy()
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(a), columns=a.columns)
    pca = PCA(n_components=5)
    pca_fit = pca.fit(scaled_df)
    pc_values = np.arange(pca.n_components_) + 1

    if plot:
        plt.figure(figsize=(5, 5))
        plt.plot(pc_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')
        plt.show()

    return pca
