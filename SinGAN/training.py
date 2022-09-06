import SinGAN.functions as functions
import SinGAN.models as models
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import matplotlib.pyplot as plt
from SinGAN.imresize import imresize

def train(options, generator_list, noise_maps_list, real_patch_list, NoiseAmp):

    real_ = functions.read_image(options)
    #print("real_ ====", real_.shape)
    in_s = 0
    scale_num = 0
    real = imresize(real_, options.scale1, options)
    #print("real 1 ===", real.shape)

    real_patch_list = functions.creat_reals_pyramid(real, real_patch_list, options)

    nfc_prev = 0

    while scale_num<options.stop_scale+1:

        options.nfc = min(options.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        options.min_nfc = min(options.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        options.out_ = functions.generate_dir2save(options)
        options.outf = '%s/%d' % (options.out_, scale_num)
        try:
            os.makedirs(options.outf)
        except OSError:
                pass

        #plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
        #plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
        plt.imsave('%s/real_scale.png' % (options.outf), functions.convert_image_np(real_patch_list[scale_num]), vmin=0, vmax=1)

        current_discriminator,current_generator = init_models(options)

        if (nfc_prev == options.nfc):

            current_generator.load_state_dict(torch.load('%s/%d/netG.pth' % (options.out_, scale_num - 1)))
            current_discriminator.load_state_dict(torch.load('%s/%d/netD.pth' % (options.out_, scale_num - 1)))

        z_curr, in_s, current_generator = train_single_scale(current_discriminator, current_generator, real_patch_list, generator_list, noise_maps_list, in_s, NoiseAmp, options)

        current_generator = functions.reset_grads(current_generator,False)
        current_generator.eval()
        current_discriminator = functions.reset_grads(current_discriminator,False)
        current_discriminator.eval()


        generator_list.append(current_generator)

        # We append the current noise map to the noise map list
        noise_maps_list.append(z_curr)
        NoiseAmp.append(options.noise_amp)

        torch.save(noise_maps_list, '%s/Zs.pth' % (options.out_))
        torch.save(generator_list, '%s/Gs.pth' % (options.out_))
        torch.save(real_patch_list, '%s/reals.pth' % (options.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (options.out_))

        scale_num+=1
        nfc_prev = options.nfc
        del current_discriminator,current_generator
    return



def train_single_scale(netD, netG, real_patch_pyramid, curr_generator_list, noise_patch_list, in_s, NoiseAmp, opt, centers=None):
    '''

    Parameters
    ----------
    netD
    netG
    real_patch_pyramid
    curr_generator_list
    noise_patch_list
    in_s
    NoiseAmp
    opt
    centers

    Returns
    -------

    '''
    # We take one specific path of the pyramid.
    real = real_patch_pyramid[len(curr_generator_list)]
    #print("real shape===", real.shape)
    # opt.ker_size -> number of kernels (globally fixed)
    # opt.num_layer -> number of layers (globally fixed)

    # These two fellas are the ones that change when we change the learning scale
    # They indicate the hight and width of the current scale path.
    opt.nzx = real.shape[2]#+(opt.ker_size-1)*(opt.num_layer)
    opt.nzy = real.shape[3]#+(opt.ker_size-1)*(opt.num_layer)

    # receptive field = fixed_kernel_size + (( fixed_kernel_size -1 ) * fixed_number_of_layers * fixed_stride )
    # The receptive field si fixed for all scales... same for the paddings for noise and images!!
    opt.receptive_field = opt.ker_size + ((opt.ker_size-1)*(opt.num_layer-1))*opt.stride
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)

    if opt.mode == 'animation_train':
        opt.nzx = real.shape[2]+(opt.ker_size-1)*(opt.num_layer)
        opt.nzy = real.shape[3]+(opt.ker_size-1)*(opt.num_layer)
        pad_noise = 0


    # These torchobjects serve as pad adders for whatever tensor they are feed with.
    noise_padder_layer = nn.ZeroPad2d(int(pad_noise))
    image_padder_layer = nn.ZeroPad2d(int(pad_image))

    alpha = opt.alpha

    # The following noise vector are varying for each scale training, because they are noise maps that share
    # the dimension with each scale-specific image-patch. Notice that, in the case of the coarses scale,
    # this variable (z_opt) is going to be overwritten by a noise map that repeats the noise signal over 4 channels...
    # Maybe the purpose is that, in the generative fashion, you give this kind of repeated noise sandwiches to create and
    # image segmentation sample.
    fixed_noise = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy],device=opt.device)
    z_opt = torch.full(fixed_noise.shape, 0, device=opt.device, dtype=torch.bool)
    z_opt = noise_padder_layer(z_opt)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[1600],gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)

    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    z_opt2plot = []

    for epoch in range(opt.niter):

        '''
        THIS BLOCK GENERATES A NOISE TENSOR that is useful for two purposes:
         - in the case of the first scale, it is going to be useful for reconstruction purposes.
         - in the rest of the scales, it is going to be used to feed the generator and train the discriminator, 
          (it is going to be refined later in the code, though, before being fed to the generator...)
        '''
        # guess SR_train stands for Super-resolution training, need to check SinGAN paper...
        if (curr_generator_list == []) & (opt.mode != 'SR_train'):
            # If we are in the first scale, and we are not in the SR_train setting, then we create 2 noise maps.
            # The first, z_opt, is the one that is going to be saved in memory for reconstruction purposes as stated in
            # the SinGAN paper. Though, they call it z_star in the paper!!!
            z_opt = functions.generate_noise([1, opt.nzx, opt.nzy], device=opt.device)
            # The expand operation is kind of a shape trasnformation but some broadcast is done to fill the expanded dimensions.
            # At this poit z_opt is a (1,1,x,y) tensor, with the expand operation we turn it to a (1,4,x,y) tensor, so we
            # repeat the whole noise map for 4 channels.
            expanded_z_opt = z_opt.expand(1, opt.nc_z, opt.nzx, opt.nzy) # changed second parameter from 3 to opt.nc_z - Vajira
            z_opt = noise_padder_layer(expanded_z_opt)

            # Now the noise_ tensor is going to be the normal noise map for learning purposes... We only need this one when we are
            # not at the coarsest scale
            noise_ = functions.generate_noise([1, opt.nzx, opt.nzy], device=opt.device)
            noise_ = noise_padder_layer(noise_.expand(1, opt.nc_z, opt.nzx, opt.nzy)) # changed second paramter from 3 to opt.nc_z - Vajira
        else:
            noise_ = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
            noise_ = noise_padder_layer(noise_)
        '''
        END OF BLOCK (EOB) 
        '''



        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        # We repeat this process Dsteps times... (Dsteps is the discriminator inner steps)
        for j in range(opt.Dsteps):

            # train with real
            netD.zero_grad()
            # pass the real patch trough the discriminator.
            output = netD(real).to(opt.device)

            # We want to maximize this output, or, equivalently,
            # minimize the negative of this output ;)
            # The torch optimizer (Adam) will do a gradient descent step, so we
            # can compute the gradient w.r.t. a loss function to MINIMIZE
            errD_real = -output.mean()
            errD_real.backward(retain_graph=True)

            # Now, this maximization True positive score, is saved just for plotting purposes ;)
            D_x = -errD_real.item()

            # train with fake
            # the first thing to do is generate the fake sample.
            # The fake sample is generated feeding the generator with a noise tensor,
            # and an upsampled version of the previous fake sample

            '''
            The following block generates:
             
             - The upsampled version of the previous fake patch and calls is "prev"
             
             - the "spatial noise" tensor as referred in the SinGAN paper
            calls it "noise". (it refines the noise previously generated.
            
            '''

            # First we generate the up-sampled version of the previous fake patch and calls is "prev"
            # We have to distinguish two cases: case A and case B.
            # case A
            if (j==0) & (epoch == 0):
                # THE FOLLOWING is done ONLY IN THE FIRST INNER STEP
                # OF THE FIRST EPOCH OF EVERY SCALE-specific training.

                # case A.1
                if (curr_generator_list == []) & (opt.mode != 'SR_train'):
                    # If we are in the first scale and NOT in super-resolution downstream task.
                    # In this case, our "previous"  scale  real-patch is a zeros tensor: (in this code, the authors of
                    # SinGAN-Seg have transformed it to a "falses" tensor, with the boolean type specification.
                    # Notice that the dimensions of this tensor match with the dimension of the current patches,
                    # So if it has to do -for now only in theory- with the "previous patch", then it should be used
                    # to represent an up-sampled version of it, in the first scale, the upsampling is done creating directly this
                    # zero boolean tensor
                    prev = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device, dtype=torch.bool)
                    # up to this point, in_s was a 0 integer :/,
                    in_s = prev
                    prev = image_padder_layer(prev)

                    # Same story holds for noise, the "previous noise" patch is a "falses" tensor
                    z_prev = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device, dtype=torch.bool)
                    z_prev = noise_padder_layer(z_prev)

                    opt.noise_amp = 1

                # case A.2
                elif opt.mode == 'SR_train':
                    # if we are in the SR task (irrespective of the scale), need further study:
                    z_prev = in_s
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real, z_prev))
                    opt.noise_amp = opt.noise_amp_init * RMSE
                    z_prev = image_padder_layer(z_prev)
                    prev = z_prev

                # case A.3
                else:
                    # not first scale, nor SR task: (but remember, we are in the first epoch and in the first inner step
                    # of the current scale training, i.e., we are in case A)
                    # note that now in_s contains the previous scale image patch
                    # because we have passed through case A.1 necessarily (and also through case B)
                    mode_param = 'rand'
                    #TODO explain this function
                    prev = draw_concat(curr_generator_list, noise_patch_list, real_patch_pyramid, NoiseAmp, in_s, mode_param, noise_padder_layer, image_padder_layer, opt)
                    prev = image_padder_layer(prev)

                    mode_param = 'rec'
                    z_prev = draw_concat(curr_generator_list, noise_patch_list, real_patch_pyramid, NoiseAmp, in_s, mode_param, noise_padder_layer, image_padder_layer, opt)

                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real, z_prev))
                    opt.noise_amp = opt.noise_amp_init*RMSE

                    z_prev = image_padder_layer(z_prev)

            # case B
            else:
                # the following is done in the second and third inner steps of the first epoch and
                # in every inner step of the rest of the 1999 epochs
                # notice it doesn't maters which scale we are in.
                # We hace already passed through case A.1 necessarily. so in_s contains a tensor of the same shape of
                # the current image patch. If we are in the coarsest scale, we know is a "falses" tensor (or a zeros boolean tensor)
                # but we could be in other scales, in such a case it is the previous fake image patch. $\hat{x}_{n+1}$ in the paper
                # The following function creates the upsampled version of this previous fake image patch.
                prev = draw_concat(curr_generator_list, noise_patch_list, real_patch_pyramid, NoiseAmp, in_s, 'rand', noise_padder_layer, image_padder_layer, opt)
                prev = image_padder_layer(prev)


            # don't know what this is... will study some day...
            if opt.mode == 'paint_train':
                prev = functions.quant2centers(prev,centers)
                plt.imsave('%s/prev.png' % (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)


            # Now we refine the noise tensor previously generated (i.e. noise_) and call it "noise":
            if (curr_generator_list == []) & (opt.mode != 'SR_train'):
                noise = noise_
            else:
                # if we are not in the first scale,
                # or if we are in the SR downstream task, we could also be in the first scale...

                # "Specifically, the noise zn is added to the [upsampled] image [path]
                # prior to being fed into a sequence of convolutional layers. This ensures that the GAN does not disregard
                # the noise, as often happens in conditional schemes involving randomness" (from the SinGAN paper)
                # That explains the following line of code:
                noise = opt.noise_amp * noise_ + prev

            '''
            EOB
            '''


            # Now we generate our fake sample :)
            fake = netG(noise.detach(), prev)

            # We want to minimize the ouput of the discriminator when feeded with a fake sample...
            output = netD(fake.detach())
            errD_fake = output.mean()
            errD_fake.backward(retain_graph=True)

            D_G_z = output.mean().item()

            # This kind of regularization term is what completes the formation of the WGAN-GP loss that we are using.
            # as mentioned in the SinGAN paper. Very elegant ;)
            gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device)
            gradient_penalty.backward()

            # Notice this stuff is only for reporting and does not interfere with the gradient descent.
            errD = errD_real + errD_fake + gradient_penalty

            # We have already computed all the gradients that we are interested in so we now just perform the gradient descent.
            # NOTICE we have not gave weights to each one of the loss terms... maybe we can experiment with it
            # (don't remember if the paper mentions something about that, neither I have studied completely the WGAN-GP paper...
            optimizerD.step()

        # house-made reporting... what about using W&B??
        errD2plot.append(errD.detach())

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        for j in range(opt.Gsteps):
            netG.zero_grad()
            output = netD(fake)
            #D_fake_map = output.detach()
            errG = -output.mean()
            errG.backward(retain_graph=True)
            if alpha!=0:
                loss = nn.MSELoss()
                if opt.mode == 'paint_train':
                    z_prev = functions.quant2centers(z_prev, centers)
                    plt.imsave('%s/z_prev.png' % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)
                Z_opt = opt.noise_amp*z_opt+z_prev
                rec_loss = alpha*loss(netG(Z_opt.detach(),z_prev),real)
                rec_loss.backward(retain_graph=True)
                rec_loss = rec_loss.detach()
            else:
                Z_opt = z_opt
                rec_loss = 0
            #print("Error is here...!")
            #errG.backward(retain_graph=True)
        optimizerG.step()

        errG2plot.append(errG.detach()+rec_loss)
        D_real2plot.append(D_x)
        D_fake2plot.append(D_G_z)
        z_opt2plot.append(rec_loss)

        if epoch % 25 == 0 or epoch == (opt.niter-1):
            print('scale %d:[%d/%d]' % (len(curr_generator_list), epoch, opt.niter))

        if epoch % 500 == 0 or epoch == (opt.niter-1):
            plt.imsave('%s/fake_sample.png' %  (opt.outf), functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
            plt.imsave('%s/G(z_opt).png'    % (opt.outf),  functions.convert_image_np(netG(Z_opt.detach(), z_prev).detach()), vmin=0, vmax=1)
            #plt.imsave('%s/D_fake.png'   % (opt.outf), functions.convert_image_np(D_fake_map))
            #plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
            #plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
            #plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
            #plt.imsave('%s/noise.png'    %  (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
            #plt.imsave('%s/z_prev.png'   % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)


            torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))

        schedulerD.step()
        schedulerG.step()

    functions.save_networks(netG,netD,z_opt,opt)

    return z_opt, in_s, netG



def draw_concat(list_of_generators, Zs, real_patches_pyramid, NoiseAmp, in_s, mode, noise_padder_layer, image_padder_layer, opt):
    '''


    Parameters
    ----------
    list_of_generators
    Zs: This fella is only used in this function to determine some shape of the noise tensors that are generated...
    real_patches_pyramid
    NoiseAmp
    in_s:
    mode
    noise_padder_layer
    image_padder_layer
    opt

    Returns
    -------

    '''

    G_z = in_s

    if len(list_of_generators) > 0:

        if mode == 'rand':
            count = 0
            # the same exact value of pad_noise variable defined in the train_single_scale function,
            # notice these fellas are scale independent.
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)

            if opt.mode == 'animation_train':
                pad_noise = 0

            for G, Z_opt, real_curr, real_next, noise_amp in zip(list_of_generators, Zs, real_patches_pyramid, real_patches_pyramid[1:], NoiseAmp):

                # In this inner block, Z_opt is one of the elements of the Zs list.

                if count == 0:
                    #  count==0 means we are considering the coarsest scale generator, noise and image patch...

                    # we now compute the width and height of the current scale patches  like this: (manually removing the padding overhead from the current noise patch, but the real patch could had been used also...)
                    noise_width = Z_opt.shape[2] - 2 * pad_noise
                    noise_height = Z_opt.shape[3] - 2 * pad_noise

                    # In the coarsest scale, our noise map must be a repetition of a 2-d noise map over the four channels:
                    z = functions.generate_noise([1, noise_width, noise_height], device=opt.device)
                    z = z.expand(1, opt.nc_z, z.shape[2], z.shape[3]) # changed the second parameter from 3 to opt.nc_z
                else:
                    # If we are not in the coarsest scale, we generate a 3d noise map, each channel independent.
                    z = functions.generate_noise([opt.nc_z, noise_width, noise_height], device=opt.device)

                z = noise_padder_layer(z)

                # G_z must be G(z), i.e. a fake batch, in particular, the previous fake batch (with respect to the corresponding scale training.
                # Now, in this inner "for" we are again looping through all scales (real_curr), having this G_z fixed. So it could be big, (as big as the semi-last finer scale), and thus, real_curr could be smaller. That is why this line of code takes only a part of G_z of the same dimension of real_curr
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                # we have an image patch, we just need to add the padding nos
                G_z = image_padder_layer(G_z)
                # we do the adding strategy not to induce the generator disregarding noise...
                z_in = noise_amp*z+G_z
                # and now we feed our generator with this "spatial noise" and the previous patch (remember, when we say "previous" er refer to the OUTER training loop).
                G_z = G(z_in.detach(),G_z)
                # we have generated a fake batch using the previous patch. NOW WE DO THE UPSAMPLING:
                # we are resizing to a scale wich is > 1, thus, up-sampling... good old-fashioned programming ;)
                # notice we overwrite G_z, which means that, from the second iteration on, G_z is modified also in this inner loop of scales 0.o
                # No worries, we are iterating over the generators we have until now, and each time, we take a piece of G_z as shown before for feeding the generators.
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                # Once we have done the up-sampling, we ensure that our generated patch has the dimensions of the generated pyramid. (the upsampling function uses a
                # real scale factor that could produce some excess of pixels in the resulting image...
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1

        if mode == 'rec':
            count = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(list_of_generators, Zs, real_patches_pyramid, real_patches_pyramid[1:], NoiseAmp):

                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = image_padder_layer(G_z)
                z_in = noise_amp*Z_opt+G_z
                G_z = G(z_in.detach(),G_z)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                #if count != (len(Gs)-1):
                #    G_z = m_image(G_z)
                count += 1

    return G_z



def train_paint(opt,Gs,Zs,reals,NoiseAmp,centers,paint_inject_scale):
    in_s = torch.full(reals[0].shape, 0, device=opt.device, dtype=torch.bool)
    scale_num = 0
    nfc_prev = 0

    while scale_num<opt.stop_scale+1:
        if scale_num!=paint_inject_scale:
            scale_num += 1
            nfc_prev = opt.nfc
            continue
        else:
            opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
            opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

            opt.out_ = functions.generate_dir2save(opt)
            opt.outf = '%s/%d' % (opt.out_,scale_num)
            try:
                os.makedirs(opt.outf)
            except OSError:
                    pass

            #plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
            #plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
            plt.imsave('%s/in_scale.png' %  (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

            D_curr,G_curr = init_models(opt)

            z_curr,in_s,G_curr = train_single_scale(D_curr,G_curr,reals[:scale_num+1],Gs[:scale_num],Zs[:scale_num],in_s,NoiseAmp[:scale_num],opt,centers=centers)

            G_curr = functions.reset_grads(G_curr,False)
            G_curr.eval()
            D_curr = functions.reset_grads(D_curr,False)
            D_curr.eval()

            Gs[scale_num] = G_curr
            Zs[scale_num] = z_curr
            NoiseAmp[scale_num] = opt.noise_amp

            torch.save(Zs, '%s/Zs.pth' % (opt.out_))
            torch.save(Gs, '%s/Gs.pth' % (opt.out_))
            torch.save(reals, '%s/reals.pth' % (opt.out_))
            torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

            scale_num+=1
            nfc_prev = opt.nfc
        del D_curr,G_curr
    return


def init_models(opt):

    #generator initialization:
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    #discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    return netD, netG
