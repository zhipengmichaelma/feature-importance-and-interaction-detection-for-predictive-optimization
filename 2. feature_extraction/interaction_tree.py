from graphviz import Digraph
import re

def interaction_tree(feature_names, interaction_features, filename = 'graph', method = 'all'):
    g = Digraph('G', filename=filename)
    g.graph_attr['rankdir'] = 'LR'
    A1 = list(feature_names)
    A2 = list(interaction_features)

    for i in range(len(A2)):
        A2[i] = re.sub(r'[0-9, <, >, =, .]+', '', A2[i])

    if method=='all':
        for i in range(len(A1)):
            g.node(f'node{i}', label=A1[i])
        for i in range(len(A2)):
            for j in range(len(A1)):
                if A2[i]==A1[j]:
                    A2[i] = j
            if i>0:
                g.edge(f'node{A2[i-1]}',f'node{A2[i]}')

    if method=='interaction':
        for i in range(len(A2)):
            g.node(f'node{i}', label=A2[i])
        for i in range(len(A2)):
            if i>0:
                g.edge(f'node{i-1}',f'node{i}')

    output = g
    g.render(filename, format='jpg', view=True)
    return output


