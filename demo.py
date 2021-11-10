import os
import pandas as pd
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import shutil
import json
import h5py
import attr
from pathlib import Path
from torchbiggraph.config import parse_config
from torchbiggraph.converters.importers import TSVEdgelistReader, convert_input_data
from torchbiggraph.train import train
from torchbiggraph.util import SubprocessInitializer, setup_logging
from torchbiggraph.eval import do_eval
from sklearn.manifold import TSNE

# dir = 'C:/Users/Jiawen/OneDrive/2021-2022_1/昆明公安厅/复杂网络样本(1).xlsx'
DATA_DIR = './PBG/data/example_3'
MODEL_DIR = './PBG/model_3'



def main(args):
    #df = dataloader(args)
    pbg(args)
    plot(args)

    return None

def pbg(args):
    GRAPH_PATH = args.data_format
    raw_config = dict(
        # I/O data
        entity_path=DATA_DIR,
        edge_paths=[
            DATA_DIR + '/edges_partitioned',
        ],
        checkpoint_path=MODEL_DIR,
        # Graph structure
        entities={
            "WHATEVER": {"num_partitions": 1}
        },
        relations=[
            {
                "name": "phonecall",
                "lhs": "WHATEVER",
                "rhs": "WHATEVER",
                "operator": "complex_diagonal",
            }
        ],
        dynamic_relations=False,
        dimension=50,
        global_emb=False,
        comparator="dot",
        num_epochs=50,
        num_uniform_negs=1000,
        loss_fn="softmax",
        lr=0.01,
        regularization_coef=1e-3,
        eval_fraction=0.,
    )

    setup_logging()
    config = parse_config(raw_config)
    subprocess_init = SubprocessInitializer()
    input_edge_paths = [Path(GRAPH_PATH)]

    convert_input_data(
        config.entities,
        config.relations,
        config.entity_path,
        config.edge_paths,
        input_edge_paths,
        TSVEdgelistReader(lhs_col=0, rel_col=None, rhs_col=1),
        dynamic_relations=config.dynamic_relations,
    )
    train(config, subprocess_init=subprocess_init)


    # relations = [attr.evolve(r, all_negs=True) for r in raw_config['relations']]
    # eval_config = attr.evolve(
    #     config, edge_paths='./data/example_3/edges_partitioned', relations=relations, num_uniform_negs=0
    # )

    # do_eval(eval_config, subprocess_init=subprocess_init)

    return None


def stat(args):
    df = dataloader(args)
    G = nx.from_pandas_edgelist(df, '节点A', '节点B',
                        edge_attr=True, create_using=nx.DiGraph())
    #pos = nx.random_layout(G, seed=23)
    #nx.draw(G, pos=pos, with_labels=False)
    #labels = {e:G.edges[e]['连接次数'] for e in G.edges}
    #nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels)  # 在图上标出labels, 这里的pos要和上面的pos保持一致，否则边的权重会乱

    N, K = G.order(), G.size()  # 获取节点的数量，边的数量
    avg_deg = float(K)/N        # 计算average degree(这是有向网络) 平均度是每个节点的度的总和除以节点总数
    print(N, K, avg_deg)

    # 绘制幂律分布图
    in_degrees = G.in_degree()   # 统计每个节点的in_degree
    out_degress = G.out_degree()
    inDegrees = {}
    outDegree = {}
    for i in in_degrees:
        inDegrees[i[0]] = i[1]
    for i in out_degress:
        outDegree[i[0]] = i[1]
    in_values = sorted(set(inDegrees.values()))
    inDegrees_values = list(inDegrees.values())
    in_hist = [inDegrees_values.count(x) for x in in_values]  # 统计每种度的数量
    out_values = sorted(set(outDegree.values()))
    outDegrees_values = list(outDegree.values())
    out_hist = [outDegrees_values.count(x) for x in out_values]
    plt.figure()
    plt.grid(True)
    plt.loglog(in_values, in_hist, 'ro-')   # 绘制双对数曲线；幂律分布图
    plt.loglog(out_values, out_hist, 'bv-')
    plt.legend(['In-degree', 'Out-degree'])
    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    plt.savefig('degrees.pdf')


    # 分析聚类系数
    G_ud = G.to_undirected()
    print('clust of 0:', nx.clustering(G_ud, '0'))  # '0'节点的聚类系数
    clust_coefficients = nx.clustering(G_ud)
    avg_clust = sum(clust_coefficients.values())/len(clust_coefficients)
    print('avg_clust:', avg_clust)  # 平均聚类系数  0.004901714070718665
    print('avg_clust:', nx.average_clustering(G_ud))
    return None

def dataloader(args):
    #df = pd.DataFrame()
    data = pd.read_excel(args.data_in)
    data = data.to_numpy()


    with open(args.data_format, 'w') as f:
        for edge in data:
            for i in range(edge[-1]):
                f.write('\t'.join(edge[:2]) + '\n')

    return data
    

def plot(args):
    loss, reg, violators_lhs, violators_rhs = [], [], [], []
    with open('./PBG/model_3/training_stats.json', 'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            loss.append(data['stats']['metrics']['loss'])
            reg.append(data['stats']['metrics']['reg'])
            violators_lhs.append(data['stats']['metrics']['violators_lhs'])
            violators_rhs.append(data['stats']['metrics']['violators_rhs'])

    plt.figure()
    plt.plot(range(len(loss)),loss,label ='loss')
    plt.legend()
    plt.title('PBG training loss') 
    plt.xlabel('Epochs')
    plt.savefig('./result/pbg_loss.png')

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(range(len(reg)),reg,label ='reg')
    plt.legend()
    plt.title('PBG training records') 
    plt.ylabel('REG')
    plt.xlabel('Epochs')
    plt.subplot(1,2,2)
    plt.plot(range(len(violators_lhs)),violators_lhs,label ='violators_lhs')
    plt.plot(range(len(violators_rhs)),violators_rhs,label ='violators_rhs')
    plt.legend()
    plt.title('PBG training records')
    plt.ylabel('violators')
    plt.xlabel('Epochs')
    plt.savefig('./result/pbg.png')

    
    nodes_path = DATA_DIR + '/entity_names_WHATEVER_0.json'
    embeddings_path = MODEL_DIR + "/embeddings_WHATEVER_0.v{NUMBER_OF_EPOCHS}.h5" \
        .format(NUMBER_OF_EPOCHS=50)

    with open(nodes_path, 'r') as f:
        node_names = json.load(f)
    with h5py.File(embeddings_path, 'r') as g:
        embeddings = g['embeddings'][:]
    node2embedding = dict(zip(node_names, embeddings))

    data_np = np.array([item[1] for item in node2embedding.items()])
    tsne = TSNE(n_components=2)
    Y = tsne.fit_transform(data_np)
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
    plt.savefig('./result/embedding.png')
    print(data_np.shape)




if __name__ == "__main__":
    
    args = argparse.ArgumentParser(description='Decision Stump')
    args.add_argument('--data_in', default='./data/复杂网络样本(1).xlsx', type=str,
                    help='path to the training input .xlsx file')
    args.add_argument('--data_format', default='./data/large_format.tsv', type=str,
            help='path to the training input .xlsx file')        
    args.add_argument('--out_dir', default='./result/demo.xlsx', type=str,
                    help='path to the result .xlsx file')

    args = args.parse_args()
    main(args)
