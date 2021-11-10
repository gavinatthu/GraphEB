from torchbiggraph.config import parse_config
from torchbiggraph.converters.importers import TSVEdgelistReader, convert_input_data
from torchbiggraph.train import train
from torchbiggraph.util import SubprocessInitializer, setup_logging
from torchbiggraph.eval import do_eval
from pathlib import Path


class PBG():
    def __init__(self, DATA_DIR, MODEL_DIR, GRAPH_PATH):
        self.DATA_DIR = DATA_DIR
        self.MODEL_DIR = MODEL_DIR
        self.GRAPH_PATH = GRAPH_PATH


    def train(self):
        raw_config = dict(
            # I/O data
            entity_path=self.DATA_DIR,
            edge_paths=[
                self.DATA_DIR + '/edges_partitioned',
            ],
            checkpoint_path=self.MODEL_DIR,
            # Graph structure
            entities={
                "WHATEVER": {"num_partitions": 1}
            },
            relations=[
                {
                    "name": "doesnt_matter",
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
        input_edge_paths = [Path(self.GRAPH_PATH)]
        

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

    def eval(self):
        
        return None

