import os
import re
import glob
import base64
import shutil
from typing import Tuple
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Union

# Imports needed for GitHub dataset downloading.
from github import Github
from github import GithubException

# Imports for PyTorch Geometric Dataset class.
import torch
from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

Edge_Indices = List[Union[np.ndarray, None]]
Edge_Weights = List[Union[np.ndarray, None]]
Node_Features = List[Union[np.ndarray, None]]
Targets = List[Union[np.ndarray, None]]
Additional_Features = List[np.ndarray]


class CustomDynamicGraphTemporalSignal(DynamicGraphTemporalSignal):
    def __init__(self,
                 videos,
                 edge_indices: Edge_Indices,
                 edge_weights: Edge_Weights,
                 features: Node_Features,
                 targets: Targets,
                 **kwargs: Additional_Features):
        self.videos_paths = videos
        super(CustomDynamicGraphTemporalSignal, self).__init__(edge_indices, edge_weights, features, targets, **kwargs)


class IRIGestureTemporal(InMemoryDataset):
    r"""The IRIGesture dataset
    .. note::
        TO DO
    Args:
        root (string): Root directory where the dataset should be saved.
            would be returned.
        testSubject (string): Subject to exclude from Dataset in order to use as Test.
        dataTypes (string): Use 'Dynamic' or 'Static'.
        token (string, optional): GitHub token needed in order to download 
            IRIGesture dataset. (By default uses 'GITHUB_TOKEN' environment 
            variable)
        alsoDownloadVideos (bool, optional): If set to true, videos would also be downloaded from GitHub.
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
    __categoriesDynamic = ['greeting', 'continue', 'turnback', 'no', 'slowdown', 'come', 'back']
    __token = os.environ.get("GITHUB_TOKEN", None)
    __processed = False

    __train_features = []
    __train_targets = []
    __train_videos = []

    __test_targets = []
    __test_features = []
    __test_videos = []

    __nodes_to_use = [0,  # nose
                      # 1,       # left_eye_inner
                      # 2,       # left_eye
                      # 3,       # left_eye_outer
                      # 4,       # right_eye_inner
                      # 5,       # right_eye
                      # 6,       # right_eye_outer
                      # 7,       # left_ear
                      # 8,       # right_ear
                      # 9,       # mouth_left
                      # 10,      # mouth_right
                      11,  # left_shoulder
                      12,  # right_shoulder
                      13,  # left_elbow
                      14,  # right_elbow
                      15,  # left_wrist
                      16,  # right_wrist
                      17,  # left_pinky
                      18,  # right_pinky
                      19,  # left_index
                      20,  # right_index
                      21,  # left_thumb
                      22,  # right_thumb
                      23,  # left_hip
                      24  # right_hip
                      # 25,      # left_knee
                      # 26,      # right_knee
                      # 27,      # left_ankle
                      # 28,      # right_ankle
                      # 29,      # left_heel
                      # 30,      # right_heel
                      # 31,      # left_foot_index
                      # 32]      # right_foot_index
                      ]

    number_nodes = 15
    number_targets = -1
    number_frames = 30
    frames_gap = 3
    __testSubject = 'S2'
    alsoDownloadVideos = False

    def __init__(self, root, dataTypes="Static", testSubject="S2", alsoDownloadVideos=False, token=None,
                 categories=None,
                 transform=None, pre_transform=None, pre_filter=None):

        if dataTypes == "Dynamic":
            self.StaticData = False
            self.DynamicData = True
        else:
            self.StaticData = True
            self.DynamicData = False

        self.dataTypes = dataTypes
        self.__testSubject = testSubject
        self.alsoDownloadVideos = alsoDownloadVideos

        token = self.__token if token is None else token
        self.__token = token

        if categories is not None:
            categories = [gestures.lower() for gestures in categories]
            for gestures in categories:
                assert gestures in self.__categories
            self.__categories = categories
        else:
            if self.DynamicData:
                self.__categories = self.__categoriesDynamic

        self.number_targets = len(self.__categories)

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        if not self.__processed:
            self.features = torch.load(os.path.join(self.processed_dir, f'{self.dataTypes[:3]}_feat.pt'))
            self.targets = torch.load(os.path.join(self.processed_dir, f'{self.dataTypes[:3]}_trgs.pt'))
            self.__test_features = torch.load(os.path.join(self.processed_dir, f'{self.dataTypes[:3]}_tsfeat.pt'))
            self.__train_features = torch.load(os.path.join(self.processed_dir, f'{self.dataTypes[:3]}_trfeat.pt'))
            self.__test_targets = torch.load(os.path.join(self.processed_dir, f'{self.dataTypes[:3]}_tstrgs.pt'))
            self.__train_targets = torch.load(os.path.join(self.processed_dir,
                                                           f'{self.dataTypes[:3]}_trtrgs.pt'))
            self.__test_videos = torch.load(os.path.join(self.processed_dir, f'{self.dataTypes[:3]}_tsvids.pt'))
            self.__train_videos = torch.load(os.path.join(self.processed_dir, f'{self.dataTypes[:3]}_trvids.pt'))
            self.CCO = torch.load(os.path.join(self.processed_dir, f'{self.dataTypes[:3]}_CCO.pt'))
            self.__totalElements = torch.load(os.path.join(self.processed_dir, f'{self.dataTypes[:3]}_tels.pt'))

    @property
    def raw_file_names(self):
        # We look for two random files in order to decide if dataset needs to be downloaded.
        if self.StaticData:
            raw_file_names_list = ['S1_attention_1_1m_upper.npy', 'S6_stop_2_4m_full.npy']
            if self.alsoDownloadVideos:
                raw_file_names_list.append(os.path.join('videos', 'S1_attention_1_1m_upper.avi'))
            return raw_file_names_list
        else:
            raw_file_names_list = ['S1_continue_3_6m_full.npy', 'S10_turnback_1_1m_upper.npy']
            if self.alsoDownloadVideos:
                raw_file_names_list.append(os.path.join('videos', 'S1_continue_3_6m_full.avi'))
            return raw_file_names_list

    @property
    def processed_file_names(self):
        # We generate a *.pt file with name composition of each gesture.
        name = '_'.join([gesture[:2] for gesture in self.__categories])
        return f'{self.dataTypes[:3]}_{str(self.alsoDownloadVideos)[:1]}_{name}.pt'

    def download(self):
        git = Github(self.__token)
        owner = git.get_user(self.__owner)
        repository = owner.get_repo(self.__repo)
        if os.path.exists(self.raw_dir):
            shutil.rmtree(self.raw_dir)
        os.makedirs(self.raw_dir)
        if self.alsoDownloadVideos:
            os.makedirs(os.path.join(self.raw_dir, "videos"))

        self.__recursive_download(repository, self.__serverPath, self.raw_dir)

    def __recursive_download(self, repository, server_path, local_path, content_prefix=""):
        contents = repository.get_contents(server_path)

        for content in contents:
            if content.type == 'dir':
                # We use a RegEX to store subject folder recursively.
                prefix = content.name if re.search("^S\d{0,}$", content.name) else content_prefix
                self.__recursive_download(repository, content.path, local_path, prefix)
            elif content.type == 'file' and (
                    ".npy" in content.name or (self.alsoDownloadVideos and ".avi" in content.name)):
                try:
                    path = content.path
                    if (self.StaticData and (("3Djoints" in path)
                                             or ("videos" in path and not ("dynamic_videos" in path)))) \
                            or (self.DynamicData and ("dynamic" in path)):
                        file_name = content_prefix + "_" + content.name
                        file_content = repository.get_contents(path)
                        file_data = base64.b64decode(file_content.content)
                        if ".avi" in content.name:
                            file_out = open(os.path.join(local_path, "videos", file_name), "wb")
                        else:
                            file_out = open(os.path.join(local_path, file_name), "wb")
                        file_out.write(file_data)
                        file_out.close()

                except (GithubException, IOError) as exc:
                    print('Error downloading %s: %s', content.path, exc)

    def process(self):
        if os.path.exists(self.processed_dir):
            shutil.rmtree(self.processed_dir)
        os.makedirs(self.processed_dir)
        if self.alsoDownloadVideos:
            os.makedirs(os.path.join(self.processed_dir, 'videos'))

        data_list = []
        self.features = []
        self.targets = []
        self.__test_features = []
        self.__train_features = []
        self.__test_targets = []
        self.__train_targets = []
        self.__test_videos = []
        self.__train_videos = []

        # We create an extremly connected graph.
        self.CCO = np.swapaxes([[i, j] for i in range(0, self.number_nodes) for j in range(0, self.number_nodes)], 0, 1)

        for gesture in self.__categories:
            paths = glob.glob(os.path.join(self.raw_dir, f'*{gesture}*.npy'))
            paths = sorted(paths, key=lambda e: (len(e), e))

            for path in paths:
                is_test_subject = path.__contains__(self.__testSubject)
                gesture_seq = np.load(path, allow_pickle=True)
                number_of_sequences = (gesture_seq.shape[0] - self.number_frames) // self.frames_gap

                video_name = None
                input_video = None
                if self.alsoDownloadVideos:
                    video_name = os.path.splitext(os.path.basename(path))[0]
                    input_video = cv2.VideoCapture(f'{os.path.join(self.raw_dir, "videos", video_name)}.avi')

                for seq in range(0, number_of_sequences):
                    x = np.empty([self.number_nodes, 4, 0])
                    init_frame = seq * self.frames_gap
                    end_frame = init_frame + self.number_frames

                    output_video = None
                    output_video_path = None
                    if self.alsoDownloadVideos:
                        output_video_path = f'{os.path.join(self.processed_dir, "videos", video_name)}' \
                                            f'_{init_frame}_{end_frame}.avi'
                        output_video = cv2.VideoWriter(output_video_path,
                                                       cv2.VideoWriter_fourcc(*'XVID'),
                                                       int(input_video.get(cv2.CAP_PROP_FPS)),
                                                       (int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                        int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

                    for frame in range(init_frame, end_frame):
                        pose = gesture_seq[frame, ][0]
                        frame_landmark = np.empty([0, 4])

                        if self.alsoDownloadVideos:
                            input_video.set(cv2.CAP_PROP_POS_FRAMES, frame)
                            ret, video_frame = input_video.read()
                            if ret:
                                mp.solutions.drawing_utils.draw_landmarks(video_frame,
                                                                          pose, mp.solutions.pose.POSE_CONNECTIONS)
                                output_video.write(video_frame)

                        # There's 33 landmarks in total.
                        for landmark in range(0, 33):
                            if self.__nodes_to_use.__contains__(landmark):
                                frame_landmark = np.append(frame_landmark, np.expand_dims(
                                    [pose.landmark[landmark].x, pose.landmark[landmark].y, pose.landmark[landmark].z,
                                     pose.landmark[landmark].visibility], axis=0), axis=0)

                        # x = [n° nodes, 4, number_of_frames]
                        x = np.append(x, np.expand_dims(frame_landmark, axis=2), axis=2)

                    if self.alsoDownloadVideos:
                        output_video.release()

                    x = np.swapaxes(x, 0, 1)
                    self.features.append(x)  # [4, number_nodes, number_of_frames]

                    if is_test_subject:
                        self.__test_features.append(x)
                        self.__test_videos.append(output_video_path)
                    else:
                        self.__train_features.append(x)
                        self.__train_videos.append(output_video_path)

                    x = np.swapaxes(x, 0, 2)
                    x = torch.tensor(x, dtype=torch.float)  # [number_of_frames, number_nodes, 4]

                    edge_index = torch.tensor(self.CCO)  # [2, 1089]
                    y = self.__categories.index(gesture)

                    target = np.zeros(len(self.__categories))
                    target[y] = 1
                    self.targets.append(target)

                    if is_test_subject:
                        self.__test_targets.append(target)
                    else:
                        self.__train_targets.append(target)

                    y = torch.tensor(y)  # [1]

                    data = Data(x=x, edge_index=edge_index, edge_attr=None, y=y)
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)

                if self.alsoDownloadVideos:
                    input_video.release()

        torch.save(self.collate(data_list), self.processed_paths[0])
        torch.save(self.features, os.path.join(self.processed_dir, f'{self.dataTypes[:3]}_feat.pt'))
        torch.save(self.__test_features, os.path.join(self.processed_dir, f'{self.dataTypes[:3]}_tsfeat.pt'))
        torch.save(self.__train_features, os.path.join(self.processed_dir, f'{self.dataTypes[:3]}_trfeat.pt'))
        torch.save(self.targets, os.path.join(self.processed_dir, f'{self.dataTypes[:3]}_trgs.pt'))
        torch.save(self.__test_targets, os.path.join(self.processed_dir, f'{self.dataTypes[:3]}_tstrgs.pt'))
        torch.save(self.__train_targets, os.path.join(self.processed_dir, f'{self.dataTypes[:3]}_trtrgs.pt'))
        torch.save(self.__test_videos, os.path.join(self.processed_dir, f'{self.dataTypes[:3]}_tsvids.pt'))
        torch.save(self.__train_videos, os.path.join(self.processed_dir, f'{self.dataTypes[:3]}_trvids.pt'))
        torch.save(self.CCO, os.path.join(self.processed_dir, f'{self.dataTypes[:3]}_CCO.pt'))

        self.__totalElements = len(data_list)
        torch.save(self.__totalElements, os.path.join(self.processed_dir, f'{self.dataTypes[:3]}_tels.pt'))

        self.__processed = True

    def get_dataset(self) -> Tuple[CustomDynamicGraphTemporalSignal, CustomDynamicGraphTemporalSignal]:
        """Returning the IRIGesture data iterator.

        Return types:
            * **(train_dataset, test_dataset)** *(tuple of DynamicGraphTemporalSignal)* - The IRIGestureTemporalDataset.
        """

        test_dataset = CustomDynamicGraphTemporalSignal(
            self.__test_videos,
            self.__get_edges(number_elements=len(self.__test_features)),  # List of CCO [2, self.number_nodes**2]
            self.__get_edge_weights(number_elements=len(self.__test_features)),  # List of ones (self.number_nodes**2, )
            self.__test_features,  # List each item (4, self.number_nodes, frames)
            self.__test_targets  # List each item (frames, gestures)
        )

        train_dataset = CustomDynamicGraphTemporalSignal(
            self.__train_videos,
            self.__get_edges(number_elements=len(self.__train_features)),  # List of CCO [2, self.number_nodes**2]
            self.__get_edge_weights(number_elements=len(self.__train_features)),  # List of ones (
            # self.number_nodes**2, )
            self.__train_features,  # List each item (4, self.number_nodes, frames)
            self.__train_targets  # List each item (gestures)
        )

        return train_dataset, test_dataset

    def get_all_dataset(self) -> DynamicGraphTemporalSignal:
        """Returning the IRIGesture data iterator.

        Args types:
            * **frames** *(int)* - The number of consecutive frames T, default 16.
        Return types:
            * **dataset** *(DynamicGraphTemporalSignal)* - The IRIGestureTemporal dataset.
        """
        dataset = DynamicGraphTemporalSignal(
            self.__get_edges(),  # List of CCO [2, self.number_nodes**2]
            self.__get_edge_weights(),  # List of ones (self.number_nodes**2, )
            self.features,  # List each item (4, self.number_nodes, frames)
            self.targets  # List each item (gestures)
        )
        return dataset

    def __get_edges(self, number_elements=None):
        number_of_elements = self.__totalElements if number_elements is None else number_elements
        edges = [self.CCO] * number_of_elements
        return edges

    def __get_edge_weights(self, number_elements=None):
        number_of_elements = self.__totalElements if number_elements is None else number_elements
        edge_weights = [np.ones((self.number_nodes ** 2,))] * number_of_elements
        return edge_weights

    @property
    def categories(self) -> List[str]:
        return self.__categories
