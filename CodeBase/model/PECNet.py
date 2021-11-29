import torch
from torch import nn
from .backbone.Linear import MLP


class PECNet(nn.Module):
    def __init__(self, 
                enc_past_size, 
                enc_dest_size, 
                enc_latent_size, 
                dec_size, 
                predictor_size, 
                fdim, 
                zdim, 
                sigma, 
                past_length, 
                future_length, 
                verbose):
        '''
        Args:
            size parameters: Dimension sizes
            sigma: Standard deviation used for sampling N(0, sigma)
            past_length: Length of past history (number of timesteps)
            future_length: Length of future trajectory to be predicted
        '''
        super(PECNet, self).__init__()
        self.zdim = zdim
        self.sigma = sigma

        # takes in the past
        self.encoder_past = MLP(input_dim = past_length*2, output_dim = fdim, hidden_size=enc_past_size)

        self.encoder_dest = MLP(input_dim = 2, output_dim = fdim, hidden_size=enc_dest_size)

        self.encoder_latent = MLP(input_dim = 2*fdim, output_dim = 2*zdim, hidden_size=enc_latent_size)

        self.decoder = MLP(input_dim = fdim + zdim, output_dim = 2, hidden_size=dec_size)

        self.predictor = MLP(input_dim = 2*fdim, output_dim = 2*(future_length-1), hidden_size=predictor_size)

        architecture = lambda net: [l.in_features for l in net.layers] + [net.layers[-1].out_features]

        if verbose:
            print("Past Encoder architecture : {}".format(architecture(self.encoder_past)))
            print("Dest Encoder architecture : {}".format(architecture(self.encoder_dest)))
            print("Latent Encoder architecture : {}".format(architecture(self.encoder_latent)))
            print("Decoder architecture : {}".format(architecture(self.decoder)))
            print("Predictor architecture : {}".format(architecture(self.predictor)))

    # def forward(self, x, dest = None, device=torch.device('cpu')):
    def forward(self, obs, otherInp=None, extraInp=None, params=None):

        # provide destination iff training
        # assert model.training
        # obs: (b,s,2)
        # pred: (b, pred, 2)
        obsRelate = (obs - obs[:,:1,:])
        if self.training:
            pred = otherInp[0]
            pred = (pred - obs[:,:1,:])
            dest = pred[:,-1,:]
        obsRelate = obsRelate.view(obsRelate.shape[0], -1)
        # pred = pred.view(pred.shape[0], -1)
        
        # future = pred[:,:-1,:]
        device = params.device


        # assert self.training ^ (dest is None)
    
        # encode
        ftraj = self.encoder_past(obsRelate)

        if not self.training:
            z = torch.Tensor(params.dataset.num_traj, obs.size(0), self.zdim)
            z.normal_(0, self.sigma)

        else:
            # during training, use the destination to produce generated_dest and use it again to predict final future points

            # CVAE code
            dest_features = self.encoder_dest(dest)
            features = torch.cat((ftraj, dest_features), dim = 1)
            latent =  self.encoder_latent(features)

            mu = latent[:, 0:self.zdim] # 2-d array
            logvar = latent[:, self.zdim:] # 2-d array

            var = logvar.mul(0.5).exp_()
            eps = torch.FloatTensor(var.size()).normal_()
            eps = eps.to(device)
            z = eps.mul(var).add_(mu)

        z = z.float().to(device)
        

        if self.training:
            decoder_input = torch.cat((ftraj, z), dim = 1)
            generated_dest = self.decoder(decoder_input)
            generated_dest_features = self.encoder_dest(generated_dest)

            prediction_features = torch.cat((ftraj, generated_dest_features), dim = 1)

            pred_future = self.predictor(prediction_features)

            predTraj = pred_future.view(-1,params.dataset.pred_len-1, 2)
            predTraj = torch.cat((predTraj,generated_dest.unsqueeze(1)),dim=1)
            # print(predTraj.shape)
            # print(obs.shape)
            # t=input()
            predTraj = predTraj + obs[:,:1,:]
            return predTraj, {
                'goal': generated_dest, 
                'mean': mu, 
                'var': logvar, 
                'futureTraj': pred_future}
        else:
            # print(shape)
            ftraj = ftraj.unsqueeze(0).repeat((params.dataset.num_traj, 1, 1))
            decoder_input = torch.cat((ftraj, z), dim=2)
            generated_dest = self.decoder(decoder_input)
            generated_dest_features = self.encoder_dest(generated_dest)
            # print(generated_dest_features.shape)
            # t=input()
            prediction_features = torch.cat((ftraj, generated_dest_features), dim=2)
            pred_future = self.predictor(prediction_features)
            predTraj = pred_future.view(params.dataset.num_traj,-1,params.dataset.pred_len-1,2)
            predTraj = torch.cat((predTraj,generated_dest.unsqueeze(2)),dim=2)

            predTraj = predTraj + obs.unsqueeze(0)[:,:,:1,:]
            # print(decoder_input.shape)
            # print(predTraj.shape)
            # t=input()
        return predTraj
