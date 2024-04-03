from neuprint import Client
from neuprint import fetch_neurons, fetch_synapses
from neuprint import NeuronCriteria as NC

from bokeh.models import Range1d
from bokeh.plotting import figure
import numpy as np

# If true, analyze from the excel sheet,
# the larger set of putative Crz neurons.

ALL_PUTATIVE = False

if ALL_PUTATIVE:
    import pandas as pd


def main(auth_token : str, savepath : str):
    with open(auth_token, 'r') as f:
        token = f.read()

    c = Client('neuprint.janelia.org', dataset='manc:v1.0', token=token)
    c.fetch_version()

    if ALL_PUTATIVE:
        xl = pd.read_excel('Putative IDs.xlsx')
        xl.columns = xl.loc[0,:]
        xl.drop(0, inplace=True)
        # drop first column
        xl.drop(xl.columns[0], axis=1, inplace=True)
        putative_crz =  xl['Crz'].loc[~xl['Crz'].isnull()].values.astype(int)
    else:
        putative_crz = [14161, 11828, 13190, 16132] 


    pcrz = fetch_neurons(
        NC(
            bodyId=putative_crz,
        ),
        client=c
    )[0]

    BYZANTINE = '#D342BE'
    SHAMROCK = '#33A358'

    roi_mesh = c.fetch_roi_mesh('ANm')

    def plot_neurons(source_df, idx : int, flattened_ax : int = 1):

        skelee = c.fetch_skeleton(source_df.iloc[idx]['bodyId'])

        skelee.radius = skelee.radius.astype(float)/(20)
        # Join parent/child nodes for plotting as line segments below.
        # (Using each row's 'link' (parent) ID, find the row with matching rowId.)
        skelee = skelee.merge(skelee, 'inner',
                                left_on=['link'],
                                right_on=['rowId'],
                                suffixes=['_child', '_parent'])
        p = figure(
            title = 'Body ID: ' + str(source_df.iloc[idx]['bodyId'])
                + f' (Ach : {round(source_df.iloc[idx]["ntAcetylcholineProb"],2)}, '
                + f'Glu : {round(source_df.iloc[idx]["ntGlutamateProb"],2)}, '
                + f'GABA: {round(source_df.iloc[idx]["ntGabaProb"],2)})',
            match_aspect = True
        )
        p.y_range.flipped = False

        axes = [x for x in range(3) if x != flattened_ax]
        verts = np.array([
            [float(row.split(' ')[axes[0]+1]),float(row.split(' ')[axes[1]+1])]
            for row in str(roi_mesh).split('\\n') if row.startswith('v')
        ])

        speckle_size = 1.0
        p.scatter(verts[:,0],verts[:,1],size=speckle_size,line_alpha=0.0, fill_color = (0,0,0), fill_alpha = 0.5)

        str_labels = ['x', 'y', 'z']

        as_str = [str_labels[i] for i in axes]

        # Plot skeleton segments (in 2D)
        p.segment(
            x0=f'{as_str[0]}_child', x1=f'{as_str[0]}_parent',
            y0=f'{as_str[1]}_child', y1=f'{as_str[1]}_parent',
            line_width = 'radius_parent',
            alpha = 0.8,
            color = '#19657F',
            source=skelee
        )
        
        syns = fetch_synapses(NC(bodyId=source_df.iloc[idx]['bodyId']))
        pre = syns.loc[syns['type'] == 'pre']
        p.scatter(
            pre[as_str[0]], pre[as_str[1]],
            size = 3, fill_color=BYZANTINE,
            fill_alpha = 0.9, line_color = None,
            line_alpha = 0.0, line_width=0,
        )
        
        post = syns.loc[syns['type'] == 'post']
        
        p.scatter(
            post[as_str[0]], post[as_str[1]],
            size = 3, fill_color=SHAMROCK,
            fill_alpha = 0.8, line_color=None,
            line_width = 0,
        )

        p.renderers.insert(0,p.renderers.pop())

        lims = {
            'x' : Range1d(18000,30000),
            'y' : Range1d(0,20000),
            'z' : Range1d(0,20000)
        }

        if as_str[0] in lims:
            p.x_range = lims[as_str[0]]
        if as_str[1] in lims:
            p.y_range = lims[as_str[1]]

        #p.y_range = Range1d(0,20000)
        #p.x_range = Range1d(18000,30000)
        p.xgrid.visible = False
        p.ygrid.visible = False
        p.axis.visible = False
        return p

    from bokeh.plotting import gridplot

    from bokeh.io import export_png

    #output_file(filename='/Users/stephen/Desktop/Data/SICode/manc/some_neurons.png')
    # make a grid
    grid = gridplot([[plot_neurons(pcrz,idx, k) for k in range(3)] for idx in range(len(pcrz))])

    export_png(grid, filename = f'{savepath}.png')

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print('Usage: python generate_putative_crz.py <auth_token> <savepath>')
        exit(1)
    main(sys.argv[1], sys.argv[2])