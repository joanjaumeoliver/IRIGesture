import os
import re
import glob
import base64
import shutil
import numpy as np

# Imports needed for GitHub dataset downloading.
from github import Github
from github import GithubException

# Imports for PyTorch Geometric Dataset class.
import torch
from torch_geometric.data import (InMemoryDataset, Data)

class IRIGesture(InMemoryDataset):
    r"""The IRIGesture dataset

    .. note::
        TO DO

    Args:
        root (string): Root directory where the dataset should be saved.
        token (string, optional): GitHub token needed in order to download 
            IRIGesture dataset. (By default uses 'GITHUB_TOKEN' environment 
            variable)
        categories (list, optional): List of categories to include in the
            dataset. Can include the categories :obj:`"attention"`, :obj:`"right"`,
            :obj:`"left"`, :obj:`"stop"`, :obj:`"yes"`, :obj:`"shrug"`,
            :obj:`"random"`, :obj:`"static"`. If set to
            :obj:`None`, the dataset will contain all categories. (default:
            :obj:`None`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """
    # We will use this information in order to access GitHub.
    __owner = "RamonRL"
    __repo = "GESTURE-PROJECT"
    __serverPath = "dataset/BodyGestureDataset"
    __categories = ['attention', 'right', 'left', 'stop', 'yes', 'shrug', 'random', 'static']
    __token = os.environ.get("GITHUB_TOKEN", None)

    def __init__(self, root, token=None, categories=None,
                 transform=None, pre_transform=None, pre_filter=None):
        
        self.includeStatic = True
        self.includeDynamic = False
        
        token = self.__token if token is None else token
        self.__token = token

        if categories is not None:
            categories = [gestures.lower() for gestures in categories]
            for gestures in categories:
                assert gestures in self.__categories
            self.__categories = categories 
        
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # We look for two random files in order to decide if dataset needs to be downloaded.
        return ['S1_attention_1_1m_upper.npy', 'S6_stop_2_4m_full.npy']

    @property
    def processed_file_names(self):
        # We generate a *.pt file with name composition of each gesture.
        name = '_'.join([gesture[:2] for gesture in self.__categories])
        return f'{name}.pt'

    def download(self):
        git = Github(self.__token)
        owner = git.get_user(self.__owner)
        repository = owner.get_repo(self.__repo)
        if os.path.exists(self.raw_dir): shutil.rmtree(self.raw_dir)
        os.makedirs(self.raw_dir)
                 
        self.__recursiveDownload(repository, self.__serverPath, self.raw_dir)

    def __recursiveDownload(self, repository, server_path, local_path, content_prefix = ""):
        contents = repository.get_contents(server_path)

        for content in contents:
            if content.type == 'dir' and not ("videos" in content.name):
                # We use a RegEX to store subject folder recursively.
                prefix = content.name if re.search("^S\d{0,}$", content.name) else content_prefix
                self.__recursiveDownload(repository, content.path, local_path, prefix)
            elif content.type == 'file' and (".npy" in content.name):
                try:
                    path = content.path
                    if (("3Djoints" in path) and self.includeStatic) or (("dynamic_joints" in path) and self.includeDynamic):
                        file_name = content_prefix + "_" + content.name
                        file_content = repository.get_contents(path)
                        file_data = base64.b64decode(file_content.content)
                        file_out = open(os.path.join(local_path, file_name),"wb")
                        file_out.write(file_data)
                        file_out.close()
                except (GithubException, IOError) as exc:
                    print('Error downloading %s: %s', content.path, exc)   

    def process(self):
        data_list = []
        
        # We create an extremly connected graph.
        CCO = [[i,j] for i in range(0, len(self.__categories) - 1) for j in range(0, len(self.__categories) - 1)]
        
        for gesture in self.__categories:
            paths = glob.glob(os.path.join(self.raw_dir, f'*{gesture}*.npy'))
            paths = sorted(paths, key=lambda e: (len(e), e))

            for path in paths:
                pose = np.load(path, allow_pickle=True)[-1,][0]

                x = []
                # There's 33 landmarks to take into account.
                for landmark in range(0,33):
                    x.append([pose.landmark[landmark].x, pose.landmark[landmark].y, 
                              pose.landmark[landmark].z, pose.landmark[landmark].visibility])
                    
                x = torch.tensor(x, dtype=torch.float)
                edge_index = torch.tensor(CCO)
                y = torch.tensor(self.__categories.index(gesture))
                
                data = Data(x=x, edge_index=edge_index, edge_attr=None, y=y)
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                    
                data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])