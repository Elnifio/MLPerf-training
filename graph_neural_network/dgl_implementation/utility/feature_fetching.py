import torch
import os
import concurrent.futures
import os.path as osp
import numpy as np

class IGBHeteroGraphStructure:
    """
    Synchronously (optionally parallelly) loads the edge relations for IGBH. 
    Current IGBH edge relations are not yet converted to torch tensor. 
    """
    def __init__(
            self,
            # dataset metadata 
            path, dataset_size, num_classes=2983,

            # in-memory and memory-related optimizations
            in_memory=True,
            separate_sampling_aggregation=False,
            
            # perf related
            multithreading=True
        ):
        
        self.dir = path
        self.dataset_size = dataset_size
        self.in_memory = in_memory
        self.label_file = f'node_label_{"19" if num_classes != 2983 else "2K"}.npy'

        self.use_journal_conference = (dataset_size in ['large', 'full'])
        self.separate_sampling_aggregation=separate_sampling_aggregation

        self.torch_tensor_input_dir = path
        self.torch_tensor_input = self.torch_tensor_input_dir != ""

        self.multithreading = multithreading

        # This class only stores the edge data, labels, and the train/val indices
        self.edge_dict = self.load_edge_dict()
        self.label = self.load_labels()
        self.full_num_trainable_nodes = (227130858 if num_classes != 2983 else 157675969)
        self.train_indices, self.val_indices = self.get_train_val_test_indices()

    def load_edge_dict(self):
        mmap_mode = None if self.in_memory else "r"
        
        edges = ["paper__cites__paper", "paper__written_by__author", "author__affiliated_to__institute", "paper__topic__fos"]
        if self.use_journal_conference: 
            edges += ["paper__published__journal", "paper__venue__conference"]
        
        loaded_edges = None
        def load_edge(edge, mmap=mmap_mode, parent_path=osp.join(self.dir, self.dataset_size, "processed")):
            return edge, torch.from_numpy(np.load(osp.join(parent_path, edge, "edge_index.npy"), mmap_mode=mmap))
        
        if self.multithreading:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                loaded_edges = executor.map(load_edge, edges)
            loaded_edges = {
                tuple(edge.split("__")): (edge_index[:, 0], edge_index[:, 1]) for edge, edge_index in loaded_edges
            }
        else:
            loaded_edges = {
                tuple(edge.split("__")): (edge_index[:, 0], edge_index[:, 1])
                for edge, edge_index in map(load_edge, edges)
            }

        return self.augment_edges(loaded_edges)
    
    def load_labels(self):
        if self.dataset_size not in ['full', 'large']:
            return torch.from_numpy(
                np.load(
                    osp.join(
                        self.dir, 
                        self.dataset_size, 
                        'processed', 
                        'paper', 
                        self.label_file)
                )
            ).to(torch.long)
        else: 
            return torch.from_numpy(
                np.memmap(
                    osp.join(
                        self.dir, 
                        self.dataset_size, 
                        'processed', 
                        'paper', 
                        self.label_file
                    ), 
                    dtype='float32', 
                    mode='r', 
                    shape=(
                        (269346174 if self.dataset_size == "full" else 100000000)
                    )
                )
            ).to(torch.long)

    def augment_edges(self, edge_dict):
        # Adds reverse edge connections to the graph
        # add rev_{edge} to every edge except paper-cites-paper
        edge_dict.update(
            {
                (dst, f"rev_{edge}", src): (dst_idx, src_idx)
                for (src, edge, dst), (src_idx, dst_idx) in edge_dict.items()
                if src != dst
            }
        )

        paper_cites_paper = edge_dict[("paper", 'cites', 'paper')]

        edge_dict[("paper", 'cites', 'paper')] = (
            torch.cat((paper_cites_paper[0], paper_cites_paper[1])),
            torch.cat((paper_cites_paper[1], paper_cites_paper[0]))
        )

        return edge_dict

    def get_train_val_test_indices(self):
        base_dir = osp.join(self.dir, self.dataset_size, "processed")
        assert osp.exists(osp.join(base_dir, "train_idx.pt")) and osp.exists(osp.join(base_dir, "val_idx.pt")), \
            "Train and validation indices not found. Please run GLT's split_seeds.py first."
        
        return (
            torch.load(osp.join(self.dir, self.dataset_size, "processed", "train_idx.pt")), 
            torch.load(osp.join(self.dir, self.dataset_size, "processed", "val_idx.pt"))
        )
        

class Features:
    """
    Lazily initializes the features for IGBH. 

    Features will be initialized only when *build_features* is called. 

    Features will be placed into shared memory when *share_features* is called
    or if the features are built (either mmap-ed or loaded in memory) 
    and *torch.multiprocessing.spawn* is called
    """
    def __init__(self, path, dataset_size, in_memory=True, use_fp16=True):
        self.path = path
        self.dataset_size = dataset_size
        self.in_memory=in_memory
        self.use_fp16=use_fp16
        self.feature = {}

    def build_features(self, use_journal_conference=False, multithreading=False):
        node_types = ['paper', 'author', 'institute', 'fos']
        if use_journal_conference or self.dataset_size in ['large', 'full']:
            node_types += ['conference', 'journal']

        if multithreading:
            def load_feature(feature_store, feature_name):
                return feature_store.load(feature_name), feature_name

            with concurrent.futures.ThreadPoolExecutor() as executor:
                loaded_features = executor.map(load_feature, [(self, ntype) for ntype in node_types])
                self.feature = {
                    node_type: feature_value for feature_value, node_type in loaded_features
                }
        else:
            for node_type in node_types:
                self.feature[node_type] = self.load(node_type)

    def share_features(self):
        for node_type in self.feature:
            self.feature[node_type] = self.feature[node_type].share_memory_()

    def load_from_tensor(self, node):
        return torch.load(osp.join(self.path, self.dataset_size, "processed", node, "node_feat_fp16.pt"))
        
    def load_in_memory_numpy(self, node):
        return torch.from_numpy(np.load(osp.join(self.path, self.dataset_size, 'processed', node, 'node_feat.npy')))
        
    def load_mmap_numpy(self, node):
        """
        Loads a given numpy array through mmap_mode="r"
        """
        return torch.from_numpy(np.load(osp.join(self.path, self.dataset_size, "processed", node, "node_feat.npy"), mmap_mode="r" ))

    def memmap_mmap_numpy(self, node):
        """
        Loads a given NumPy array through memory-mapping np.memmap. 
        
        This is the same code as the one provided in IGB codebase. 
        """
        shape = [None, 1024]
        if self.dataset_size == "full":
            if node == "paper":
                shape[0] = 269346174
            elif node == "author":
                shape[0] = 277220883
        elif self.dataset_size == "large":
            if node == "paper":
                shape[0] = 100000000
            elif node == "author":
                shape[0] = 116959896

        assert shape[0] is not None
        return torch.from_numpy(np.memmap(osp.join(self.path, self.dataset_size, "processed", node, "node_feat.npy"), dtype="float32", mode='r', shape=shape))

    def load(self, node):
        if self.in_memory:
            if self.use_fp16:
                return self.load_from_tensor(node)
            else:
                if self.dataset_size in ['large', 'full'] and node in ['paper', 'author']:
                    return self.memmap_mmap_numpy(node)
                else:
                    return self.load_in_memory_numpy(node)
        else:
            if self.dataset_size in ['large', 'full'] and node in ['paper', 'author']:
                return self.memmap_mmap_numpy(node)
            else:
                return self.load_mmap_numpy(node)

    def get_input_features(self, input_dict, device): 
        # fetches the batch inputs
        # moving it here so so that future modifications could be easier
        return {
            key: self.feature[key][value.to(torch.device("cpu")), :].to(device).to(torch.float32) 
            for key, value in input_dict.items()
        }
        


