import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam,SGD, RMSprop, Adagrad
from tqdm import tqdm
import numpy as np

from utils.softargmax import SoftArgmax2D, create_meshgrid
from utils.preprocessing import augment_data, create_images_dict
from utils.image_utils import create_gaussian_heatmap_template, create_dist_mat, \
        preprocess_image_for_segmentation, pad, resize
from utils.dataloader import SceneDataset, scene_collate
from test import evaluate
from train import train

# Transformer import
from transformer.decoder import Decoder
from transformer.multihead_attention import MultiHeadAttention
from transformer.positional_encoding import PositionalEncoding
from transformer.pointerwise_feedforward import PointerwiseFeedforward
from transformer.encoder_decoder import EncoderDecoder
from transformer.encoder import Encoder
from transformer.encoder_layer import EncoderLayer
from transformer.decoder_layer import DecoderLayer
from transformer.batch import subsequent_mask
from transformer.noam_opt import NoamOpt
import baselineUtils
import scipy.io
import os
import copy
import math


class YNetEncoder(nn.Module):
    def __init__(self, in_channels, channels=(64, 128, 256, 512, 512)):
        """
        Encoder model
        :param in_channels: int, semantic_classes + obs_len
        :param channels: list, hidden layer channels
        """
        super(YNetEncoder, self).__init__()
        self.stages = nn.ModuleList()

        # First block
        self.stages.append(nn.Sequential(
                nn.Conv2d(in_channels, channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True),
        ))

        # Subsequent blocks, each starting with MaxPool
        for i in range(len(channels)-1):
            self.stages.append(nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                    nn.Conv2d(channels[i], channels[i+1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels[i+1], channels[i+1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.ReLU(inplace=True)))

        # Last MaxPool layer before passing the features into decoder
        self.stages.append(nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)))

    def forward(self, x):
        # Saves the feature maps Tensor of each layer into a list, as we will later need them again for the decoder
        features = []
        for stage in self.stages:
                x = stage(x)
                features.append(x)
        return features


class YNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, output_len, traj=False):
        """
        Decoder models
        :param encoder_channels: list, encoder channels, used for skip connections
        :param decoder_channels: list, decoder channels
        :param output_len: int, pred_len
        :param traj: False or int, if False -> Goal and waypoint predictor, if int -> number of waypoints
        """
        super(YNetDecoder, self).__init__()

        # The trajectory decoder takes in addition the conditioned goal and waypoints as an additional image channel
        if traj:
            encoder_channels = [channel+traj for channel in encoder_channels]
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder
        center_channels = encoder_channels[0]

        decoder_channels = decoder_channels

        # The center layer (the layer with the smallest feature map size)
        self.center = nn.Sequential(
                nn.Conv2d(center_channels, center_channels*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(center_channels*2, center_channels*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True)
        )

        # Determine the upsample channel dimensions
        upsample_channels_in = [center_channels*2] + decoder_channels[:-1]
        upsample_channels_out = [num_channel // 2 for num_channel in upsample_channels_in]

        # Upsampling consists of bilinear upsampling + 3x3 Conv, here the 3x3 Conv is defined
        self.upsample_conv = [
                nn.Conv2d(in_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                for in_channels_, out_channels_ in zip(upsample_channels_in, upsample_channels_out)]
        self.upsample_conv = nn.ModuleList(self.upsample_conv)

        # Determine the input and output channel dimensions of each layer in the decoder
        # As we concat the encoded feature and decoded features we have to sum both dims
        in_channels = [enc + dec for enc, dec in zip(encoder_channels, upsample_channels_out)]
        out_channels = decoder_channels

        self.decoder = [nn.Sequential(
                nn.Conv2d(in_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True))
                for in_channels_, out_channels_ in zip(in_channels, out_channels)]
        self.decoder = nn.ModuleList(self.decoder)


        # Final 1x1 Conv prediction to get our heatmap logits (before softmax)
        self.predictor = nn.Conv2d(in_channels=decoder_channels[-1], out_channels=output_len, kernel_size=1, stride=1, padding=0)

    def forward(self, features):
        # Takes in the list of feature maps from the encoder. Trajectory predictor in addition the goal and waypoint heatmaps
        features = features[::-1]  # reverse the order of encoded features, as the decoder starts from the smallest image
        center_feature = features[0]
        x = self.center(center_feature)
        for i, (feature, module, upsample_conv) in enumerate(zip(features[1:], self.decoder, self.upsample_conv)):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # bilinear interpolation for upsampling
            x = upsample_conv(x)  # 3x3 conv for upsampling
            x = torch.cat([x, feature], dim=1)  # concat encoder and decoder features
            x = module(x)  # Conv
        x = self.predictor(x)  # last predictor layer
        return x


class YNetTorch(nn.Module):
    def __init__(self, obs_len, pred_len, segmentation_model_fp, use_features_only=False, semantic_classes=6,
                             encoder_channels=[], decoder_channels=[], waypoints=1):
        """
        Complete Y-net Architecture including semantic segmentation backbone, heatmap embedding and ConvPredictor
        :param obs_len: int, observed timesteps
        :param pred_len: int, predicted timesteps
        :param segmentation_model_fp: str, filepath to pretrained segmentation model
        :param use_features_only: bool, if True -> use segmentation features from penultimate layer, if False -> use softmax class predictions
        :param semantic_classes: int, number of semantic classes
        :param encoder_channels: list, encoder channel structure
        :param decoder_channels: list, decoder channel structure
        :param waypoints: int, number of waypoints
        """
        super(YNetTorch, self).__init__()

        if segmentation_model_fp is not None:
            self.semantic_segmentation = torch.load(segmentation_model_fp)
            if use_features_only:
                self.semantic_segmentation.segmentation_head = nn.Identity()
                semantic_classes = 16  # instead of classes use number of feature_dim
        else:
            self.semantic_segmentation = nn.Identity()


        self.encoder = YNetEncoder(in_channels=semantic_classes + obs_len, channels=encoder_channels)

        self.goal_decoder = YNetDecoder(encoder_channels, decoder_channels, output_len=pred_len)
        self.traj_decoder = YNetDecoder(encoder_channels, decoder_channels, output_len=pred_len, traj=waypoints)

        self.softargmax_ = SoftArgmax2D(normalized_coordinates=False)

    def segmentation(self, image):
        return self.semantic_segmentation(image)

    # Forward pass for goal decoder
    def pred_goal(self, features):
        goal = self.goal_decoder(features)
        return goal

    # Forward pass for trajectory decoder
    def pred_traj(self, features):
        traj = self.traj_decoder(features)
        return traj

    # Forward pass for feature encoder, returns list of feature maps
    def pred_features(self, x):
        features = self.encoder(x)
        return features

    # Softmax for Image data as in dim=NxCxHxW, returns softmax image shape=NxCxHxW
    def softmax(self, x):
        return nn.Softmax(2)(x.view(*x.size()[:2], -1)).view_as(x)

    # Softargmax for Image data as in dim=NxCxHxW, returns 2D coordinates=Nx2
    def softargmax(self, output):
        return self.softargmax_(output)

    def sigmoid(self, output):
        return torch.sigmoid(output)

    def softargmax_on_softmax_map(self, x):
        """ Softargmax: As input a batched image where softmax is already performed (not logits) """
        pos_y, pos_x = create_meshgrid(x, normalized_coordinates=False)
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)
        x = x.flatten(2)

        estimated_x = pos_x * x
        estimated_x = torch.sum(estimated_x, dim=-1, keepdim=True)
        estimated_y = pos_y * x
        estimated_y = torch.sum(estimated_y, dim=-1, keepdim=True)
        softargmax_coords = torch.cat([estimated_x, estimated_y], dim=-1)
        return softargmax_coords


class YNet:
    def __init__(self, obs_len, pred_len, segmentation_model_fp,
                 use_features_only, semantic_classes, encoder_channels, decoder_channels, waypoints):
        """
        Ynet class, following a sklearn similar class structure
        :param obs_len: observed timesteps
        :param pred_len: predicted timesteps
        :param params: dictionary with hyperparameters
        """
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.division_factor = 2 ** len(encoder_channels)

        self.model = YNetTorch(obs_len=obs_len,
                                                   pred_len=pred_len,
                                                   segmentation_model_fp=segmentation_model_fp,
                                                   use_features_only=use_features_only,
                                                   semantic_classes=semantic_classes,
                                                   encoder_channels=encoder_channels,
                                                   decoder_channels=decoder_channels,
                                                   waypoints=len(waypoints))

    def train(self, train_data, val_data, params, train_image_path, val_image_path, experiment_name, batch_size=8, num_goals=20, num_traj=1, device=None, dataset_name=None):
        """
        Train function
        :param train_data: pd.df, train data
        :param val_data: pd.df, val data
        :param params: dictionary with training hyperparameters
        :param train_image_path: str, filepath to train images
        :param val_image_path: str, filepath to val images
        :param experiment_name: str, arbitrary name to name weights file
        :param batch_size: int, batch size
        :param num_goals: int, number of goals per trajectory, K_e in paper
        :param num_traj: int, number of trajectory per goal, K_a in paper
        :param device: torch.device, if None -> 'cuda' if torch.cuda.is_available() else 'cpu'
        :return:
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        obs_len = self.obs_len
        pred_len = self.pred_len
        total_len = pred_len + obs_len

        print('Preprocess data')
        dataset_name = dataset_name.lower()
        if dataset_name == 'sdd':
            image_file_name = 'reference.jpg'
        elif dataset_name == 'ind':
            image_file_name = 'reference.png'
        elif dataset_name == 'eth':
            image_file_name = 'oracle.png'
        else:
            raise ValueError(f'{dataset_name} dataset is not supported')

        # ETH/UCY specific: Homography matrix is needed to convert pixel to world coordinates
        if dataset_name == 'eth':
            self.homo_mat = {}
            for scene in ['eth', 'hotel', 'students001', 'students003', 'uni_examples', 'zara1', 'zara2', 'zara3']:
                self.homo_mat[scene] = torch.Tensor(np.loadtxt(f'data/eth_ucy/{scene}_H.txt')).to(device)
            seg_mask = True
        else:
            self.homo_mat = None
            seg_mask = False

        # Load train images and augment train data and images
        df_train, train_images = augment_data(train_data, image_path=train_image_path, image_file=image_file_name,
                                                                                  seg_mask=seg_mask)

        # Load val scene images
        val_images = create_images_dict(val_data, image_path=val_image_path, image_file=image_file_name)

        # Initialize dataloaders
        train_dataset = SceneDataset(df_train, resize=params['resize'], total_len=total_len)
        train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=scene_collate, shuffle=True)

        val_dataset = SceneDataset(val_data, resize=params['resize'], total_len=total_len)
        val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=scene_collate)

        # Preprocess images, in particular resize, pad and normalize as semantic segmentation backbone requires
        resize(train_images, factor=params['resize'], seg_mask=seg_mask)
        pad(train_images, division_factor=self.division_factor)  # make sure that image shape is divisible by 32, for UNet segmentation
        preprocess_image_for_segmentation(train_images, seg_mask=seg_mask)

        resize(val_images, factor=params['resize'], seg_mask=seg_mask)
        pad(val_images, division_factor=self.division_factor)  # make sure that image shape is divisible by 32, for UNet segmentation
        preprocess_image_for_segmentation(val_images, seg_mask=seg_mask)

        model = self.model.to(device)

        # Freeze segmentation model
        for param in model.semantic_segmentation.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        criterion = nn.BCEWithLogitsLoss()

        # Create template
        size = int(4200 * params['resize'])

        input_template = create_dist_mat(size=size)
        input_template = torch.Tensor(input_template).to(device)

        gt_template = create_gaussian_heatmap_template(size=size, kernlen=params['kernlen'], nsig=params['nsig'], normalize=False)
        gt_template = torch.Tensor(gt_template).to(device)

        best_test_ADE = 99999999999999

        self.train_ADE = []
        self.train_FDE = []
        self.val_ADE = []
        self.val_FDE = []

        print('Start training')
        for e in tqdm(range(params['num_epochs']), desc='Epoch'):
            train_ADE, train_FDE, train_loss = train(model, train_loader, train_images, e, obs_len, pred_len,
                                                                                             batch_size, params, gt_template, device,
                                                                                             input_template, optimizer, criterion, dataset_name, self.homo_mat)
            self.train_ADE.append(train_ADE)
            self.train_FDE.append(train_FDE)

            # For faster inference, we don't use TTST and CWS here, only for the test set evaluation
            val_ADE, val_FDE = evaluate(model, val_loader, val_images, num_goals, num_traj,
                                                                    obs_len=obs_len, batch_size=batch_size,
                                                                    device=device, input_template=input_template,
                                                                    waypoints=params['waypoints'], resize=params['resize'],
                                                                    temperature=params['temperature'], use_TTST=False,
                                                                    use_CWS=False, dataset_name=dataset_name,
                                                                    homo_mat=self.homo_mat, mode='val')
            print(f'Epoch {e}: \nVal ADE: {val_ADE} \nVal FDE: {val_FDE}')
            self.val_ADE.append(val_ADE)
            self.val_FDE.append(val_FDE)

            if val_ADE < best_test_ADE:
                print(f'Best Epoch {e}: \nVal ADE: {val_ADE} \nVal FDE: {val_FDE}')
                torch.save(model.state_dict(), 'pretrained_models/' + experiment_name + '_weights.pt')
                best_test_ADE = val_ADE

    def evaluate(self, data, params, image_path, batch_size=8, num_goals=20, num_traj=1, rounds=1, device=None, dataset_name=None):
        """
        Val function
        :param data: pd.df, val data
        :param params: dictionary with training hyperparameters
        :param image_path: str, filepath to val images
        :param batch_size: int, batch size
        :param num_goals: int, number of goals per trajectory, K_e in paper
        :param num_traj: int, number of trajectory per goal, K_a in paper
        :param rounds: int, number of epochs to evaluate
        :param device: torch.device, if None -> 'cuda' if torch.cuda.is_available() else 'cpu'
        :return:
        """

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        obs_len = self.obs_len
        pred_len = self.pred_len
        total_len = pred_len + obs_len

        print('Preprocess data')
        dataset_name = dataset_name.lower()
        if dataset_name == 'sdd':
            image_file_name = 'reference.jpg'
        elif dataset_name == 'ind':
            image_file_name = 'reference.png'
        elif dataset_name == 'eth':
            image_file_name = 'oracle.png'
        else:
            raise ValueError(f'{dataset_name} dataset is not supported')

        # ETH/UCY specific: Homography matrix is needed to convert pixel to world coordinates
        if dataset_name == 'eth':
            self.homo_mat = {}
            for scene in ['eth', 'hotel', 'students001', 'students003', 'uni_examples', 'zara1', 'zara2', 'zara3']:
                self.homo_mat[scene] = torch.Tensor(np.loadtxt(f'data/eth_ucy/{scene}_H.txt')).to(device)
            seg_mask = True
        else:
            self.homo_mat = None
            seg_mask = False

        test_images = create_images_dict(data, image_path=image_path, image_file=image_file_name)

        test_dataset = SceneDataset(data, resize=params['resize'], total_len=total_len)
        test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=scene_collate)

        # Preprocess images, in particular resize, pad and normalize as semantic segmentation backbone requires
        resize(test_images, factor=params['resize'], seg_mask=seg_mask)
        pad(test_images, division_factor=self.division_factor)  # make sure that image shape is divisible by 32, for UNet architecture
        preprocess_image_for_segmentation(test_images, seg_mask=seg_mask)

        model = self.model.to(device)

        # Create template
        size = int(4200 * params['resize'])

        input_template = torch.Tensor(create_dist_mat(size=size)).to(device)

        self.eval_ADE = []
        self.eval_FDE = []

        print('Start testing')
        for e in tqdm(range(rounds), desc='Round'):
            test_ADE, test_FDE = evaluate(model, test_loader, test_images, num_goals, num_traj,
                                                                          obs_len=obs_len, batch_size=batch_size,
                                                                          device=device, input_template=input_template,
                                                                          waypoints=params['waypoints'], resize=params['resize'],
                                                                          temperature=params['temperature'], use_TTST=True,
                                                                          use_CWS=True if len(params['waypoints']) > 1 else False,
                                                                          rel_thresh=params['rel_threshold'], CWS_params=params['CWS_params'],
                                                                          dataset_name=dataset_name, homo_mat=self.homo_mat, mode='test')
            print(f'Round {e}: \nTest ADE: {test_ADE} \nTest FDE: {test_FDE}')

            self.eval_ADE.append(test_ADE)
            self.eval_FDE.append(test_FDE)

        print(f'\n\nAverage performance over {rounds} rounds: \nTest ADE: {sum(self.eval_ADE) / len(self.eval_ADE)} \nTest FDE: {sum(self.eval_FDE) / len(self.eval_FDE)}')


    def load(self, path):
        print(self.model.load_state_dict(torch.load(path)))

    def save(self, path):
        torch.save(self.model.state_dict(), path)
                
                
class Transformer:
    def __init__(self, obs_len,pred_len,enc_inp_size, dec_inp_size, dec_out_size, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1,mean=[0,0],std=[0,0]):

        self.obs_len = obs_len
        self.pred_len = pred_len
#                 self.division_factor = 2 ** len(encoder_channels)
        c = copy.deepcopy
        attn = MultiHeadAttention(h, d_model)
        ff = PointerwiseFeedforward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.mean=np.array(mean)
        self.std=np.array(std)
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn),
                         c(ff), dropout), N),
            nn.Sequential(LinearEmbedding(enc_inp_size,d_model), c(position)),
            nn.Sequential(LinearEmbedding(dec_inp_size,d_model), c(position)),
            Generator(d_model, dec_out_size))

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, *input):
        return self.model.generator(self.model(*input))

    def train(self, train_data, val_data, params, train_image_path, val_image_path, experiment_name, batch_size=8, num_goals=20, num_traj=1, device=None, dataset_name=None):
        """
        Train function
        :param train_data: pd.df, train data
        :param val_data: pd.df, val data
        :param params: dictionary with training hyperparameters
        :param train_image_path: str, filepath to train images
        :param val_image_path: str, filepath to val images
        :param experiment_name: str, arbitrary name to name weights file
        :param batch_size: int, batch size
        :param num_goals: int, number of goals per trajectory, K_e in paper
        :param num_traj: int, number of trajectory per goal, K_a in paper
        :param device: torch.device, if None -> 'cuda' if torch.cuda.is_available() else 'cpu'
        :return:
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        obs_len = self.obs_len
        pred_len = self.pred_len
        total_len = pred_len + obs_len

        print('Preprocess data')
        dataset_name = dataset_name.lower()
        if dataset_name == 'sdd':
            image_file_name = 'reference.jpg'
        elif dataset_name == 'ind':
            image_file_name = 'reference.png'
        elif dataset_name == 'eth':
            image_file_name = 'oracle.png'
        else:
            raise ValueError(f'{dataset_name} dataset is not supported')

        # ETH/UCY specific: Homography matrix is needed to convert pixel to world coordinates
        if dataset_name == 'eth':
            self.homo_mat = {}
            for scene in ['eth', 'hotel', 'students001', 'students003', 'uni_examples', 'zara1', 'zara2', 'zara3']:
                    self.homo_mat[scene] = torch.Tensor(np.loadtxt(f'data/eth_ucy/{scene}_H.txt')).to(device)
            seg_mask = True
        else:
            self.homo_mat = None
            seg_mask = False

        # Load train images and augment train data and images
        df_train, train_images = augment_data(train_data, image_path=train_image_path, image_file=image_file_name,
                                                                                  seg_mask=seg_mask)

        # Load val scene images
        val_images = create_images_dict(val_data, image_path=val_image_path, image_file=image_file_name)

        # Initialize dataloaders
        train_dataset = SceneDataset(df_train, resize=params['resize'], total_len=total_len)
        train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=scene_collate, shuffle=True)

        val_dataset = SceneDataset(val_data, resize=params['resize'], total_len=total_len)
        val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=scene_collate)

        # Preprocess images, in particular resize, pad and normalize as semantic segmentation backbone requires
        resize(train_images, factor=params['resize'], seg_mask=seg_mask)
        pad(train_images, division_factor=self.division_factor)  # make sure that image shape is divisible by 32, for UNet segmentation
        preprocess_image_for_segmentation(train_images, seg_mask=seg_mask)

        resize(val_images, factor=params['resize'], seg_mask=seg_mask)
        pad(val_images, division_factor=self.division_factor)  # make sure that image shape is divisible by 32, for UNet segmentation
        preprocess_image_for_segmentation(val_images, seg_mask=seg_mask)

        #optim = SGD(list(a.parameters())+list(model.parameters())+list(generator.parameters()),lr=0.01)
        #sched=torch.optim.lr_scheduler.StepLR(optim,0.0005)
        optim = NoamOpt(params.emb_size, params.factor, len(train_loader)*params.warmup,
                torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        #optim=Adagrad(list(a.parameters())+list(model.parameters())+list(generator.parameters()),lr=0.01,lr_decay=0.001)
        epoch=0
        mean=train_dataset[:][0][:,:,2:4].mean((0,1))
        std=train_dataset[:][0][:,:,2:4].std((0,1))
        print("mean:{} std:{}".format(mean,std))

# outer loop, for loop over each scene as scenes have different image size and to calculate segmentation only once

        while epoch<params.num_epochs:
            epoch_loss=0
            self.model.train()
#             train_loss = 0
#             train_ADE = []
#             train_FDE = []
#             counter = 0

            for id_b,(trajectory, meta, scene) in enumerate(train_loader):
#                                 trajectory[:, :obs_len, :]
                for i in range(0, len(trajectory), batch_size):
                    optim.optimizer.zero_grad()
                    inp=(trajectory[i:i+batch_size, :obs_len, :].to(device)-mean.to(device))/std.to(device)
                    target=(trajectory[i:i+batch_size, obs_len:, :].to(device)-mean.to(device))/std.to(device)
                    target_c=torch.zeros((target.shape[0],target.shape[1],1)).to(device)
                    target=torch.cat((target,target_c),-1)
                    start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(target.shape[0],1,1).to(device)

                    dec_inp = torch.cat((start_of_seq, target), 1)

                    src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
                    trg_att=subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0],1,1).to(device)

                    pred=self.forward(inp, dec_inp, src_att, trg_att)

                    loss = F.pairwise_distance(pred[:, :,0:2].contiguous().view(-1, 2),
                                                     ((trajectory[i:i+batch_size, obs_len:, :].to(device)-mean.to(device))/std.to(device)).contiguous().view(-1, 2).to(device)).mean() + torch.mean(torch.abs(pred[:,:,2]))
                    loss.backward()
                    optim.step()
                    print("train epoch %03i/%03i  batch %04i traj %04i/%04i / %04i loss: %7.4f" % (epoch, params.num_epochs, id_b,i,len(trajectory), len(train_loader), loss.item()))
                    epoch_loss += loss.item()
#sched.step()
                    log.add_scalar('Loss/train', epoch_loss / len(train_loader), epoch)
            with torch.no_grad():
                model.eval()

                val_loss=0
                step=0
                gt = []
                pr = []
#                 inp_ = []
#                 peds = []
#                 frames = []
#                 dt = []

                for id_b, (trajectory, meta, scene) in enumerate(val_loader):
#                     inp_.append(batch['src'])
                    gt.append(trajectory[:, obs_len:, :])
#                     frames.append(batch['frames'])
#                     peds.append(batch['peds'])
#                     dt.append(batch['dataset'])

                    inp = (trajectory[:, :obs_len, :].to(device) - mean.to(device)) / std.to(device)
                    src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
                    start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp.shape[0], 1, 1).to(device)
                    dec_inp = start_of_seq

                    for i in range(self):
                        trg_att = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(device)
                        out = self.forward(inp, dec_inp, src_att, trg_att)
                        dec_inp = torch.cat((dec_inp, out[:, -1:, :]), 1)
                        preds_tr_b = (dec_inp[:, :, 0:2] * std.to(device) + mean.to(device)).cpu().numpy().cumsum(1) + trajectory[:, obs_len-1:obs_len, :].cpu().numpy()
                        pr.append(preds_tr_b)
                        print("val epoch %03i/%03i  batch %04i / %04i" % (
                        epoch, params.num_epochs, id_b, len(val_loader)))
#                 peds = np.concatenate(peds, 0)
#                 frames = np.concatenate(frames, 0)
#                     dt = np.concatenate(dt, 0)
                gt = np.concatenate(gt, 0)
#                     dt_names = test_dataset.data['dataset_name']
                pr = np.concatenate(pr, 0)
                mad, fad, errs = baselineUtils.distance_metrics(gt, pr)
                log.add_scalar('validation/MAD', mad, epoch)
                log.add_scalar('validation/FAD', fad, epoch)
                print("epoch:{} validation MAD:{} FAD:{}".format(epoch,mad,fad))

                if params.evaluate:
                    model.eval()
                    gt = []
                    pr = []
#                     inp_ = []
#                     peds = []
#                     frames = []
#                     dt = []

                    for id_b,(trajectory, meta, scene) in enumerate(test_loader):
#                         inp_.append(batch['src'])
                        gt.append(trajectory[:, obs_len:, :])
#                         frames.append(batch['frames'])
#                         peds.append(batch['peds'])
#                         dt.append(batch['dataset'])

                        inp = (trajectory[:, :obs_len, :].to(device) - mean.to(device)) / std.to(device)
                        src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
                        start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp.shape[0], 1, 1).to(device)
                        dec_inp=start_of_seq

                        for i in range(pred_len):
                            trg_att = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(device)
                            out = model(inp, dec_inp, src_att, trg_att)
                            dec_inp=torch.cat((dec_inp,out[:,-1:,:]),1)


                        preds_tr_b=(dec_inp[:,:,0:2]*std.to(device)+mean.to(device)).cpu().numpy().cumsum(1)+trajectory[:, obs_len-1:obs_len, :].cpu().numpy()
                        pr.append(preds_tr_b)
                        print("test epoch %03i/%03i  batch %04i / %04i" % (
                                epoch, params.num_epochs, id_b, len(test_dl)))

#                     peds = np.concatenate(peds, 0)
#                     frames = np.concatenate(frames, 0)
#                     dt = np.concatenate(dt, 0)
                    gt = np.concatenate(gt, 0)
#                     dt_names = test_dataset.data['dataset_name']
                    pr = np.concatenate(pr, 0)
                    mad, fad, errs = baselineUtils.distance_metrics(gt, pr)


                    log.add_scalar('eval/DET_mad', mad, epoch)
                    log.add_scalar('eval/DET_fad', fad, epoch)
                    print("test MAD:{} FAD:{}".format(mad,fad))


    # log.add_scalar('eval/DET_mad', mad, epoch)
    # log.add_scalar('eval/DET_fad', fad, epoch)

#                 scipy.io.savemat(f"output/Individual/{args.name}/det_{epoch}.mat",
#                                  {'input': inp, 'gt': gt, 'pr': pr, 'peds': peds, 'frames': frames, 'dt': dt,
#                                   'dt_names': dt_names})

            if epoch%params.save_step==0:
                if os.path.exists('trained_models/Transformer/'):
                    os.system('mkdir -p {}'.format('trained_models/Transformer/'))
                torch.save(model.state_dict(),'trained_models/Transformer/{epoch:05d}.pth')
            epoch += 1

    def evaluate(self, data, params, image_path, batch_size=8, num_goals=20, num_traj=1, rounds=1, device=None, dataset_name=None):
        """
        Val function
        :param data: pd.df, val data
        :param params: dictionary with training hyperparameters
        :param image_path: str, filepath to val images
        :param batch_size: int, batch size
        :param num_goals: int, number of goals per trajectory, K_e in paper
        :param num_traj: int, number of trajectory per goal, K_a in paper
        :param rounds: int, number of epochs to evaluate
        :param device: torch.device, if None -> 'cuda' if torch.cuda.is_available() else 'cpu'
        :return:
        """

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        obs_len = self.obs_len
        pred_len = self.pred_len
        total_len = pred_len + obs_len

        print('Preprocess data')
        dataset_name = dataset_name.lower()
        if dataset_name == 'sdd':
            image_file_name = 'reference.jpg'
        elif dataset_name == 'ind':
            image_file_name = 'reference.png'
        elif dataset_name == 'eth':
            image_file_name = 'oracle.png'
        else:
            raise ValueError(f'{dataset_name} dataset is not supported')

        # ETH/UCY specific: Homography matrix is needed to convert pixel to world coordinates
        if dataset_name == 'eth':
            self.homo_mat = {}
            for scene in ['eth', 'hotel', 'students001', 'students003', 'uni_examples', 'zara1', 'zara2', 'zara3']:
                    self.homo_mat[scene] = torch.Tensor(np.loadtxt(f'data/eth_ucy/{scene}_H.txt')).to(device)
            seg_mask = True
        else:
            self.homo_mat = None
            seg_mask = False

        test_images = create_images_dict(data, image_path=image_path, image_file=image_file_name)

        test_dataset = SceneDataset(data, resize=params['resize'], total_len=total_len)
        test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=scene_collate)

        # Preprocess images, in particular resize, pad and normalize as semantic segmentation backbone requires
        resize(test_images, factor=params['resize'], seg_mask=seg_mask)
        pad(test_images, division_factor=self.division_factor)  # make sure that image shape is divisible by 32, for UNet architecture
        preprocess_image_for_segmentation(test_images, seg_mask=seg_mask)

        model = self.model.to(device)

        # Create template
        size = int(4200 * params['resize'])

        input_template = torch.Tensor(create_dist_mat(size=size)).to(device)

        self.eval_ADE = []
        self.eval_FDE = []

        print('Start testing')
        for e in tqdm(range(rounds), desc='Round'):
            test_ADE, test_FDE = evaluate(model, test_loader, test_images, num_goals, num_traj,
                                                                          obs_len=obs_len, batch_size=batch_size,
                                                                          device=device, input_template=input_template,
                                                                          waypoints=params['waypoints'], resize=params['resize'],
                                                                          temperature=params['temperature'], use_TTST=True,
                                                                          use_CWS=True if len(params['waypoints']) > 1 else False,
                                                                          rel_thresh=params['rel_threshold'], CWS_params=params['CWS_params'],
                                                                          dataset_name=dataset_name, homo_mat=self.homo_mat, mode='test')
            print(f'Round {e}: \nTest ADE: {test_ADE} \nTest FDE: {test_FDE}')

            self.eval_ADE.append(test_ADE)
            self.eval_FDE.append(test_FDE)

        print(f'\n\nAverage performance over {rounds} rounds: \nTest ADE: {sum(self.eval_ADE) / len(self.eval_ADE)} \nTest FDE: {sum(self.eval_FDE) / len(self.eval_FDE)}')


    def load(self, path):
        print(self.model.load_state_dict(torch.load(path)))

    def save(self, path):
        torch.save(self.model.state_dict(), path)
                
                



























