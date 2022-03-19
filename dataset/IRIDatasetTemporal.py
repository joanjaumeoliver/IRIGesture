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
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

class IRIGestureTemporal(InMemoryDataset):
    r"""The IRIGesture dataset
    .. note::
        TO DO
    Args:
        root (string): Root directory where the dataset should be saved.
        temporal (boolean): If set to true dataset with temporal component
            would be returned.
        dataTypes (string): Use 'Dynamic' or 'Static'.
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
        
    def __init__(self, root, dataTypes = "Static", token=None, categories=None,
                 transform=None, pre_transform=None, pre_filter=None):
        
        if dataTypes == "Dynamic":
            self.StaticData = True
            self.DynamicData = False
        else:
            self.StaticData = False
            self.DynamicData = True
        
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
                    if (("3Djoints" in path) and self.StaticData) or (("dynamic_joints" in path) and self.DynamicData):
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
        self.features = []
        self.targets = []
               
        # We create an extremly connected graph.
        self.CCO = np.swapaxes([[i,j] for i in range(0, 33) for j in range(0, 33)], 0, 1)
        
        for gesture in self.__categories:
            paths = glob.glob(os.path.join(self.raw_dir, f'*{gesture}*.npy'))
            paths = sorted(paths, key=lambda e: (len(e), e))

            for path in paths:
                gesture_seq = np.load(path, allow_pickle=True)
                number_of_frames = gesture_seq.shape[0]

                x = np.empty([33,4,0])
                for frame in range(0, number_of_frames):
                    pose = gesture_seq[frame,][0]
                    frame_landmark = np.empty([0,4])

                    # There's 33 landmarks to take into account.
                    for landmark in range(0,33):
                        frame_landmark = np.append(frame_landmark, np.expand_dims([pose.landmark[landmark].x, pose.landmark[landmark].y, pose.landmark[landmark].z, pose.landmark[landmark].visibility], axis = 0), axis=0)

                    x = np.append(x,  np.expand_dims(frame_landmark, axis = 2), axis = 2) #[33, 4, number_of_frames]

                #x = np.swapaxes(np.swapaxes(x, 0, 2), 1, 2)
                x = np.swapaxes(x, 0, 1)
                self.features.append(x) #[4, 33, number_of_frames]
                x = np.swapaxes(x, 0, 2)      
                x = torch.tensor(x, dtype=torch.float) #[number_of_frames, 33, 4]

                edge_index = torch.tensor(self.CCO) #[2, 1089]
                y = self.__categories.index(gesture)
                
                target = np.zeros([number_of_frames, len(self.__categories)])
                target[:,y] = 1                
                self.targets.append(target)
                
                y = torch.tensor(y) #[1]
                
                data = Data(x=x, edge_index=edge_index, edge_attr=None, y=y)
                if self.pre_filter is not None and not self.pre_filter(data):
                  continue
                if self.pre_transform is not None:
                  data = self.pre_transform(data)

                data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
        
        self.__totalElements = len(data_list)
        
    def getTemporalDataset(self) -> object:
        if self.DynamicData:
            return self.__getDynamicTemporalDataset
        else:
            return self.__getStaticTemporalDataset
        
    def __getDynamicTemporalDataset(self) -> DynamicGraphTemporalSignal:
        """Returning the IRIGesture data iterator.

        Args types:
            * **frames** *(int)* - The number of consecutive frames T, default 16.
        Return types:
            * **dataset** *(DynamicGraphTemporalSignal)* - The IRIGestureTemporal dataset.
        """      
        self._get_edges() #List of CCO [2, 1089]
        self._get_edge_weights() #List of ones (1089, )
        #self.features() # List each item (4, 33, frames)
        #self.targets() #List each item (frames, gestures)

        dataset = DynamicGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset
        
    def __getStaticTemporalDataset(self) -> StaticGraphTemporalSignal:
        """Returning the IRIGesture data iterator.

        Args types:
            * **frames** *(int)* - The number of consecutive frames T, default 16.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The IRIGestureTemporal dataset.
        """      
        self._get_edges() #CCO [2, 1089]
        self._get_edge_weights() #Ones (1089, )
        #self.features() # List each item (4, 33, frames)
        #self.targets() #List each item (frames, gestures)
        
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset
    
    def _get_edges(self):
        if self.DynamicData:
            self._edges = [self.CCO] * self.__totalElements
        else:
            self._edges = self.CCO
    
    def _get_edge_weights(self):
        ones = np.ones((1089, ))
        
        if self.DynamicData:
            self._edge_weights = [ones] * self.__totalElements
        else:
            self._edge_weights = ones