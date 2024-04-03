from neuprint import Client
from neuprint import fetch_neurons, fetch_adjacencies
from neuprint import NeuronCriteria as NC
from neuprint import SynapseCriteria as SC

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

putative_crz = [14161, 11828, 13190, 16132] 
CRZ_PLUS =[14161, 11828, 13190, 16132 , 17702, 14938] 
CINNABAR = '#db544b'
LAPIS = '#2d66a5'
BYZANTINE = '#D342BE'
SHAMROCK = '#33A358'

def cosine_similarity(matrix):
    """
    Calculate the cosine similarity for a matrix.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Calculate cosine similarity between rows
    cosine_sim_matrix = cosine_similarity(matrix)

    # Create a DataFrame with cosine similarity matrix
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=matrix.index, columns=matrix.index)

    return cosine_sim_df



def main(auth_token : str, savepath : str):
    with open(auth_token, 'r') as f:
        token = f.read()

    c = Client('neuprint.janelia.org', dataset='manc:v1.0', token=token)
    c.fetch_version()

    _, crz_targets = fetch_adjacencies(
        sources = NC(
            bodyId=putative_crz,
        ),
    )

    ## Similarity measures

    matrix = crz_targets.pivot_table(index='bodyId_pre', columns='bodyId_post', values='weight', aggfunc='sum', fill_value=0)

    # sort by cosine similarity
    cosine_sim_matrix = cosine_similarity(matrix)
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=matrix.index, columns=matrix.index)
    reference_row = cosine_sim_df.index[0]
    sorted_rows = cosine_sim_df[reference_row].sort_values(ascending=False).index
    sorted_matrix = matrix.loc[sorted_rows]
    sorted_matrix = sorted_matrix[sorted_rows]

    sns.heatmap(
        cosine_sim_df[sorted_rows].loc[sorted_rows],
        #matrix[putative_crz].loc[putative_crz],#.values,#sorted_matrix,
        #matrix.apply(lambda x: x/x.sum(), axis=1),
        cmap='viridis',
        square=True,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={'label': 'Cosine similarity of output vectors'},
        vmin=0,
        #vmax=0.3,
    )

    sp = Path(savepath)

    plt.savefig(str(sp / 'crz_crz_cosine_distance.svg'), format='svg')

    # Let's do hierarchical clustering

    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist, squareform

    # Calculate the distance matrix
    distance_matrix = pdist(matrix/matrix.sum(axis=0), metric='euclidean')

    # Convert the distance matrix to a square-form distance matrix
    distance_sqmatrix = squareform(distance_matrix)

    # Calculate the linkage matrix
    Z = linkage(distance_sqmatrix, method='ward')

    # Plot the dendrogram
    plt.figure(figsize=(10, 10))
    dendrogram(Z, labels=matrix.index, orientation='left', leaf_font_size=10)
    plt.xlabel('Distance')
    plt.ylabel('Neuron ID')
    plt.title('Dendrogram of CRZ neurons')
    #plt.show()

    plt.gcf().savefig(str(sp / 'crz_crz_dendrogram.svg'), format='svg')

    # Output vectors

    crz_post_ns, crz_post = fetch_adjacencies(
        sources = NC(
            bodyId=putative_crz,
        ),
        min_total_weight=5,
    )

    matrix = crz_post.pivot_table(index='bodyId_pre', columns='bodyId_post', values='weight', aggfunc='sum', fill_value=0)

    plt.gcf().set_size_inches(10,2)

    main_targets = matrix.aggregate('sum', axis=0).sort_values(ascending=False).index
    #Normalize the rows
    normed_matrix = matrix.apply(lambda x: x/x.sum(), axis=1)


    sns.heatmap(
        normed_matrix[main_targets].loc[putative_crz],
        cmap='viridis',
        square=False,
        xticklabels=False,
        yticklabels=True,
        cbar_kws={'label': 'Fraction of synapses'},
        vmin=0,
        #vmax=0.3,
    )

    plt.gcf().savefig('crz_main_targets.svg', format='svg')

    summed_out = matrix[main_targets].T.sum(axis=1)
    plt.plot(
        np.arange(len(main_targets)).reshape(-1,1),
        summed_out.values,
        'ok',
    )

    outputs_df = fetch_neurons(NC(bodyId = summed_out.index))[0][['bodyId', 'ntGlutamateProb', 'ntGabaProb', 'ntAcetylcholineProb', 'predictedNtProb', 'predictedNt', 'pre', 'post', 'type']]

    CINNABAR = '#db544b'

    # # plot uncertain neurons in pink
    uncertain = outputs_df[outputs_df['predictedNtProb'] <= 0.7]['bodyId'].values
    uncertain_locvals= np.array([(summed_out.index.get_loc(cell), summed_out[cell]) for cell in uncertain]).T
    plt.plot(
        uncertain_locvals[0],
        uncertain_locvals[1],
        'o',
        color = BYZANTINE,
    )

    #plot descending neurons in yellow
    descending = outputs_df[
        outputs_df['type'].notna() 
        & (
            outputs_df['type'].str.contains('MN.*', regex=True)
            | outputs_df['type'].str.contains('EN.*', regex=True)
        )
        & ~(
            outputs_df['bodyId'].isin(CRZ_PLUS)
            | outputs_df['bodyId'].isin(CrzR)
        )]['bodyId'].values
    descending_locvals= np.array([(summed_out.index.get_loc(cell), summed_out[cell]) for cell in descending]).T
    plt.plot(
        descending_locvals[0],
        descending_locvals[1],
        'o',
        color = '#CEC019',
    )

    # plot putative CrzR in red

    CrzR = [11092, 12614, 168470, 1200, 11765, 11469, 11584]

    endf, _ = fetch_neurons(
        NC(
            bodyId = CrzR
        ),
        client=c
    )

    en00 = [cell for cell in summed_out.index.values if cell in endf['bodyId'].values]
    en_locvals= np.array([(summed_out.index.get_loc(cell), summed_out[cell]) for cell in en00]).T
    plt.plot(
        en_locvals[0],
        en_locvals[1],
        'o',
        color = CINNABAR,
    )

    # plot ascending neurons in blue
    ascending = outputs_df[
        outputs_df['type'].notna() 
        & (
            outputs_df['type'].str.contains('AN.*', regex=True)
        )]['bodyId'].values
    ascending_locvals= np.array([(summed_out.index.get_loc(cell), summed_out[cell]) for cell in ascending]).T
    plt.plot(
        ascending_locvals[0],
        ascending_locvals[1],
        'o',
        color = LAPIS,
    )

    local_neurons = outputs_df[
        ~(
            outputs_df['type'].str.contains('AN.*', regex=True)
            | outputs_df['type'].str.contains('MN.*', regex=True)
            | outputs_df['type'].str.contains('EN.*', regex=True)
            | outputs_df['bodyId'].isin(CRZ_PLUS)
            | outputs_df['bodyId'].isin(CrzR)
        ) & outputs_df['outputRois'].apply(lambda x: all(item == 'ANm' or item.startswith('Ab') for item in x))]

    # plot local neurons in green
    gaba_local = local_neurons[
        local_neurons['ntGabaProb'] > 0.7]['bodyId'].values
    gaba_local_locvals= np.array([(summed_out.index.get_loc(cell), summed_out[cell]) for cell in gaba_local]).T

    choline_local = local_neurons[
        local_neurons['ntAcetylcholineProb'] > 0.7]['bodyId'].values
    choline_local_locvals= np.array([(summed_out.index.get_loc(cell), summed_out[cell]) for cell in choline_local]).T

    glutamate_local = local_neurons[
        local_neurons['ntGlutamateProb'] > 0.7]['bodyId'].values
    glutamate_local_locvals= np.array([(summed_out.index.get_loc(cell), summed_out[cell]) for cell in glutamate_local]).T

    unknown_local = local_neurons[
        local_neurons['predictedNtProb'] <= 0.7]['bodyId'].values
    unknown_local_locvals= np.array([(summed_out.index.get_loc(cell), summed_out[cell]) for cell in unknown_local]).T

    from matplotlib.patches import Rectangle

    bar_plot_f, bar_plot_x = plt.subplots(1,1, figsize=(10,2))

    bar_plot_x.set_ylim(0, 1)
    bar_plot_x.set_xlim(0, 1)

    x, y = 0, 0

    width = summed_out.loc[descending].sum() / summed_out.sum()
    print(f"Descending: {width}")

    bar_plot_x.add_patch(
        Rectangle((x,y), width, 1, color = '#CEC019')
    )
    x+= width

    width = summed_out.loc[ascending].sum() / summed_out.sum()
    print(f"Ascending: {width}")

    bar_plot_x.add_patch(
        Rectangle((x,y), width, 1, color = LAPIS)
    )
    x+=width

    width = summed_out.loc[en00].sum() / summed_out.sum()
    print(f"CrzR: {width}")

    bar_plot_x.add_patch(
        Rectangle((x,y), width, 1, color = CINNABAR)
    )
    x+= width

    width = summed_out.loc[CRZ_PLUS].sum() / summed_out.sum()
    print(f"Crz+: {width}")

    bar_plot_x.add_patch(
        Rectangle((x,y), width, 1, color = '#00A8A3')
    )
    x+= width

    width = summed_out.loc[unknown_local].sum() / summed_out.sum()
    print(f"Unknown: {width}")

    bar_plot_x.add_patch(
        Rectangle((x,y), width, 1, color = BYZANTINE)
    )
    x+= width

    width = summed_out.loc[gaba_local].sum() / summed_out.sum()
    print(f"GABA: {width}")

    bar_plot_x.add_patch(
        Rectangle((x,y), width, 1, color = SHAMROCK)
    )
    x+= width

    width = summed_out.loc[glutamate_local].sum() / summed_out.sum()
    print(f"Glutamate: {width}")

    bar_plot_x.add_patch(
        Rectangle((x,y), width, 1, color = '#CE200F')
    )

    x+= width

    width = summed_out.loc[choline_local].sum() / summed_out.sum()
    print(f"Choline: {width}")

    bar_plot_x.add_patch(
        Rectangle((x,y), width, 1, color = '#5E24CE')
    )

    x+= width

    bar_plot_x.set_xticks([])
    bar_plot_x.set_yticks([])

    bar_plot_f.savefig('crz_output_bar_plot.svg', format='svg')


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print('Usage: python generate_crz_output_vectors.py <auth_token> <savepath>')
        exit(1)
    main(sys.argv[1], sys.argv[2])