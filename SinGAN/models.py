import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
   
class WDiscriminator(nn.Module):

    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1)
        self.body = nn.Sequential()

        # In the SinGAN papers it is said:
        # "We start with 32 kernels per block at the coarsest scale and increase
        # this number by a factor of 2 every 4 scales."
        # This is what this cycle exactly does, because our number of layers is four, we only have
        # 3 convblock in the body o the model, and thus all the scales have 32 filters in and out.
        # Ohterwise, if we would had more layers, the coarsest scales would have received and in-out channel conf of 32
        # but the first convolution, which acts on the finest scale, would had 64 channels in and 32 out ;)
        # Same holds for the generator net
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Conv2d(max(N,opt.min_nfc),1,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)

    def forward(self,x):
        #print("x shape:========", x.shape)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class GeneratorConcatSkip2CleanAdd(nn.Module):


    def __init__(self, opt):

        '''
        From the SinGAN paper:
        "Each generator Gn is responsible
        of producing realistic image samples w.r.t. the patch distribution
        in the corresponding image xn. This is achieved
        through adversarial training, where Gn learns to fool an associated
        discriminator Dn, which attempts to distinguish
        patches in the generated samples from patches in xn".
        Parameters
        ----------
        opt
        '''

        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1) #GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N,opt.min_nfc),opt.nc_im,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self, spatial_noise, prev_upsamp_patch):

        spatial_noise = self.head(spatial_noise)
        spatial_noise = self.body(spatial_noise)
        spatial_noise = self.tail(spatial_noise)

        ind = int((prev_upsamp_patch.shape[2] - spatial_noise.shape[2]) / 2)

        prev_upsamp_patch = prev_upsamp_patch[:, :, ind:(prev_upsamp_patch.shape[2] - ind), ind:(prev_upsamp_patch.shape[3] - ind)]

        # Fromt the SinGAN paper: The role of the convonlutional
        # layers is to generate the missing details in $( \tilde{x}_{n+1}) \uparrow^r $
        # (residual learning [22, 57]). Namely, Gn performs the operation
        # (and then puts exactly the equivalent of the following):
        return spatial_noise + prev_upsamp_patch
