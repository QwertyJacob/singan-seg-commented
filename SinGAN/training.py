import SinGAN.functions as functions
import SinGAN.models as models
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import matplotlib.pyplot as plt
from SinGAN.imresize import imresize


def train(options, generator_list, noise_maps_list, real_patch_list, noise_amps_list):
    real_ = functions.read_image(options)
    # print("real_ ====", real_.shape)
    in_s = 0
    scale_num = 0
    real = imresize(real_, options.scale1, options)
    # print("real 1 ===", real.shape)

    real_patch_list = functions.creat_reals_pyramid(real, real_patch_list, options)

    nfc_prev = 0

    while scale_num < options.stop_scale + 1:

        options.nfc = min(options.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        options.min_nfc = min(options.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        options.out_ = functions.generate_dir2save(options)
        options.outf = '%s/%d' % (options.out_, scale_num)
        try:
            os.makedirs(options.outf)
        except OSError:
            pass

        # plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
        # plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
        plt.imsave('%s/real_scale.png' % options.outf, functions.convert_image_np(real_patch_list[scale_num]), vmin=0,
                   vmax=1)

        current_discriminator, current_generator = init_models(options)

        if nfc_prev == options.nfc:
            current_generator.load_state_dict(torch.load('%s/%d/netG.pth' % (options.out_, scale_num - 1)))
            current_discriminator.load_state_dict(torch.load('%s/%d/netD.pth' % (options.out_, scale_num - 1)))

        z_curr, in_s, current_generator = train_single_scale(current_discriminator,
                                                             current_generator,
                                                             real_patch_list,
                                                             generator_list,
                                                             noise_maps_list,
                                                             in_s,
                                                             noise_amps_list,
                                                             options)

        current_generator = functions.reset_grads(current_generator, False)
        current_generator.eval()
        current_discriminator = functions.reset_grads(current_discriminator, False)
        current_discriminator.eval()

        generator_list.append(current_generator)

        # We append the current noise map to the noise map list
        noise_maps_list.append(z_curr)
        noise_amps_list.append(options.noise_amp)

        torch.save(noise_maps_list, '%s/Zs.pth' % (options.out_))
        torch.save(generator_list, '%s/Gs.pth' % (options.out_))
        torch.save(real_patch_list, '%s/reals.pth' % (options.out_))
        torch.save(noise_amps_list, '%s/NoiseAmp.pth' % (options.out_))

        scale_num += 1
        nfc_prev = options.nfc
        del current_discriminator, current_generator
    return


def train_single_scale(curr_discriminator, curr_generator, real_patch_pyramid, curr_generator_list, noise_patch_list,
                       in_s, noise_amps_list, opt, centers=None):
    '''

    Parameters
    ----------
    curr_discriminator
    curr_generator
    real_patch_pyramid
    curr_generator_list
    noise_patch_list
    in_s
    noise_amps_list
    opt
    centers

    Returns
    -------

    '''

    # We take one specific path of the pyramid.
    current_real_patch = real_patch_pyramid[len(curr_generator_list)]
    # print("real shape===", real.shape)
    # opt.ker_size -> number of kernels (globally fixed)
    # opt.num_layer -> number of layers (globally fixed)

    # These two fellas are the ones that change when we change the learning scale
    # They indicate the height and width of the current scale path.
    opt.nzx = current_real_patch.shape[2]  # +(opt.ker_size-1)*(opt.num_layer)
    opt.nzy = current_real_patch.shape[3]  # +(opt.ker_size-1)*(opt.num_layer)

    # receptive field = fixed_kernel_size + (( fixed_kernel_size -1 ) * fixed_number_of_layers * fixed_stride )
    # The receptive field si fixed for all scales... the paddings of noise and image patches also!!
    opt.receptive_field = opt.ker_size + ((opt.ker_size - 1) * (opt.num_layer - 1)) * opt.stride
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)

    if opt.mode == 'animation_train':
        opt.nzx = current_real_patch.shape[2] + (opt.ker_size - 1) * (opt.num_layer)
        opt.nzy = current_real_patch.shape[3] + (opt.ker_size - 1) * (opt.num_layer)
        pad_noise = 0

    # These torch objects serve as pad adders for whatever tensor they are feed with.
    noise_padder_layer = nn.ZeroPad2d(int(pad_noise))
    image_padder_layer = nn.ZeroPad2d(int(pad_image))

    alpha = opt.alpha

    # The following noise vector are varying for each scale training, because they are noise maps that share
    # the dimension with each scale-specific image-patch.
    fixed_noise = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
    z_opt = torch.full(fixed_noise.shape, 0, device=opt.device, dtype=torch.bool)
    z_opt = noise_padder_layer(z_opt)

    # setup optimizers
    optimizerD = optim.Adam(curr_discriminator.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(curr_generator.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[1600], gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[1600], gamma=opt.gamma)

    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    z_opt2plot = []

    for epoch in range(opt.niter):

        '''
        THIS BLOCK GENERATES A NOISE TENSOR that is going to be used to feed the generator and train the discriminator, 
          (however, it is going to be refined later in the code before being fed to the generator...)
        '''
        # guess SR_train stands for Super-resolution training, need to check SinGAN paper...
        if (curr_generator_list == []) & (opt.mode != 'SR_train'):
            # If we are in the first scale, and we are not in the SR_train setting, then we create 2 noise maps.

            # Notice that, if we are in the case of the coarsest scale,
            # z_opt is overwritten by a noise map that repeats the noise signal over 4 channels...
            # By doing so, SinGAN-seg authors condition the generative machinery on a 4-channel repeated noise patch
            z_opt = functions.generate_noise([1, opt.nzx, opt.nzy], device=opt.device)
            # The expand operation is a kind of shape transformation, where a broadcast operation
            # is done to fill the expanded dimensions.
            # At this point z_opt is a (1,1,x,y) tensor, and
            # with the expand operation we turn it to a (1,4,x,y) tensor, so we
            # repeat the whole noise map for 4 channels.
            expanded_z_opt = z_opt.expand(1, opt.nc_z, opt.nzx, opt.nzy)
            # Notice that SinGAN-seg changes the second parameter from 3 to opt.nc_z
            # in the previous line (w.r.t the SinGAN paper)
            z_opt = noise_padder_layer(expanded_z_opt)

            # Now the noise_ tensor is going to be the noise map for learning purposes...
            # We only need this one when we are not at the coarsest scale
            noise_ = functions.generate_noise([1, opt.nzx, opt.nzy], device=opt.device)
            noise_ = noise_padder_layer(noise_.expand(1, opt.nc_z, opt.nzx, opt.nzy))
            # Notice that SinGAN-seg changes the second parameter from 3 to opt.nc_z
            # in the previous line (w.r.t the SinGAN paper)
        else:
            noise_ = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
            noise_ = noise_padder_layer(noise_)
        '''
        END OF BLOCK (EOB) 
        '''

        ##
        # (1) Update D network:
        # maximize D(x) - D(G(z))
        # Or, equivalently:
        # minimize D(G(z)) - D(x)
        ##

        # We repeat this process Dsteps times... (Dsteps is the discriminator inner steps)
        for j in range(opt.Dsteps):

            # train with real
            curr_discriminator.zero_grad()
            # pass the real patch through the discriminator.
            output = curr_discriminator(current_real_patch).to(opt.device)

            # We want to maximize this output, or, equivalently,
            # minimize the negative of this output ;)
            # The torch optimizer (Adam) will do a gradient descent step, so we
            # can compute the gradient w.r.t. a loss function to MINIMIZE
            err_discriminator_real = - output.mean()
            err_discriminator_real.backward(retain_graph=True)

            # Now, this maximization True positive score, is saved just for plotting purposes ;)
            discr_output_real = - err_discriminator_real.item()

            # train with fake
            # the first thing to do is generate the fake sample.
            # The fake sample is generated feeding the generator with a noise tensor,
            # and an up-sampled version of the previous fake sample

            '''
            The following block generates:
             - The up-sampled version of the previous fake patch and calls is "prev_random_patch"
             - the "spatial noise" tensor as referred in the SinGAN paper
            calls it "noise". (it refines the noise_ tensor previously generated).
            '''

            # First we generate the up-sampled version of the previous fake patch and calls is "prev_random_patch"
            # We have to distinguish two cases: case A and case B.
            # case A
            if (j == 0) & (epoch == 0):
                # THE FOLLOWING is done ONLY IN THE FIRST INNER STEP
                # OF THE FIRST EPOCH OF EVERY SCALE-specific training.

                # case A.1
                if (curr_generator_list == []) & (opt.mode != 'SR_train'):
                    # If we are in the first scale and NOT in super-resolution downstream task.
                    # In this case, our "previous" scale  real-patch is a zeros tensor: (in this code, the authors of
                    # SinGAN-Seg have transformed it to a "falses" tensor, with the boolean-type specification.

                    # Notice that the dimensions of this tensor match with the dimension of the current patches,
                    # so if it has to do (for now, only in theory) with the "previous patch", then it should be used
                    # to represent an up-sampled version of it.

                    # in the first scale, the up-sampling is done creating directly this zero boolean tensor
                    prev_random_patch = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device, dtype=torch.bool)
                    # up to this point, in_s was a 0 integer :/,
                    in_s = prev_random_patch
                    prev_random_patch = image_padder_layer(prev_random_patch)

                    # Same story holds for the prev_reconstructed_patch (used for the reconstruction error)
                    prev_reconstructed_patch = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device, dtype=torch.bool)
                    prev_reconstructed_patch = noise_padder_layer(prev_reconstructed_patch)

                    opt.noise_amp = 1

                # case A.2
                elif opt.mode == 'SR_train':
                    # if we are in the SR task (irrespective of the scale), need further study:  (but remember,
                    # we are in the first epoch and in the first inner step
                    # of the current scale training, i.e., we are in case A)
                    prev_reconstructed_patch = in_s
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(current_real_patch, prev_reconstructed_patch))
                    opt.noise_amp = opt.noise_amp_init * RMSE
                    prev_reconstructed_patch = image_padder_layer(prev_reconstructed_patch)
                    prev_random_patch = prev_reconstructed_patch

                # case A.3
                else:
                    # not first scale, nor SR task: (but remember, we are in the first epoch and in the first inner step
                    # of the current scale training, i.e., we are in case A) note that now in_s contains the previous
                    # scale image patch because we have passed through case A.1 necessarily
                    # (and also through case B at least 2 times)
                    mode_param = 'rand'
                    # setting the mode_param variable to "rand" we are now generating the up-sampled version of the
                    # previous fake patch using new random noise vectors.
                    prev_random_patch = draw_concat(curr_generator_list, noise_patch_list, real_patch_pyramid, noise_amps_list,
                                       in_s, mode_param, noise_padder_layer, image_padder_layer, opt)
                    prev_random_patch = image_padder_layer(prev_random_patch)

                    mode_param = 'rec'
                    # In the paper $\tilde{x}^{rec}_{n+1}$ is "the generated image at the nth scale when using these
                    # noise maps." When it says "these noise maps, it refers to a fixed set of noise maps, here
                    # referenced by noise_patch_list. We use them to generate $\tilde{x}^{rec}_{n+1}$ when we invoke
                    # draw_concat with "rec" as the value of the mode parameter
                    prev_reconstructed_patch = draw_concat(curr_generator_list, noise_patch_list, real_patch_pyramid,
                                                      noise_amps_list, in_s, mode_param, noise_padder_layer,
                                                      image_padder_layer, opt)

                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(current_real_patch, prev_reconstructed_patch))
                    opt.noise_amp = opt.noise_amp_init * RMSE

                    prev_reconstructed_patch = image_padder_layer(prev_reconstructed_patch)

            # case B
            else:
                # the following is done in the second and third inner steps of the first epoch and in every inner step
                # of the rest of the 1999 epochs notice it doesn't maters which scale we are in.
                # We have already passed through case A.1 necessarily, so in_s contains a tensor of the same shape of
                # the current image patch. If we are in the coarsest scale, we know is a "falses" tensor (a zeros
                # boolean tensor) but we could be in other scales, in such a case it is the previous fake image patch.
                # ($\hat{x}_{n+1}$ in the paper).

                # The following function creates the up-sampled version of this previous fake image patch:
                # Notice that, at every iteration, we inject different spatial noise in the process of up-sampling
                # through the mode parameter, which is, in this case, set to "rand". see the following function.
                prev_random_patch = draw_concat(curr_generator_list, noise_patch_list, real_patch_pyramid, noise_amps_list, in_s,
                                   'rand', noise_padder_layer, image_padder_layer, opt)
                prev_random_patch = image_padder_layer(prev_random_patch)

            ##
            # don't know what this is... will study some day...
            ##
            if opt.mode == 'paint_train':
                prev_random_patch = functions.quant2centers(prev_random_patch, centers)
                plt.imsave('%s/prev_random_patch.png' % (opt.outf), functions.convert_image_np(prev_random_patch), vmin=0, vmax=1)

            ##
            # FINAL STEP TO GENERATE THE NOISE:
            ##
            # Now we refine the noise tensor previously generated (i.e. noise_) and call it "noise":
            if (curr_generator_list == []) & (opt.mode != 'SR_train'):
                noise = noise_
            else:
                # if we are not in the first scale,
                # or if we are in the SR downstream task, we could also be in the first scale...

                # "Specifically, the noise zn is added to the [upsampled] image [path]
                # prior to being fed into a sequence of convolutional layers. This ensures that the GAN does not
                # disregard the noise, as often happens in conditional schemes involving randomness"
                # (from the SinGAN paper). That explains the following line of code:
                noise = opt.noise_amp * noise_ + prev_random_patch

            '''
            EOB
            '''

            ##
            # Now we generate our fake sample :)
            fake = curr_generator(noise.detach(), prev_random_patch)

            # We want to minimize the output of the discriminator when fed with a fake sample...
            output = curr_discriminator(fake.detach())

            err_discriminator_fake = output.mean()
            err_discriminator_fake.backward(retain_graph=True)

            discr_output_fake = output.mean().item()

            # This kind of regularization term is what completes the formation of the WGAN-GP loss that we are using.
            # as mentioned in the SinGAN paper. Very elegant ;)
            gradient_penalty = functions.calc_gradient_penalty(curr_discriminator, current_real_patch, fake, opt.lambda_grad,
                                                               opt.device)
            gradient_penalty.backward()

            # Notice this stuff is only for reporting and does not interfere with the gradient descent.
            discriminator_error = err_discriminator_real + err_discriminator_fake + gradient_penalty

            # We have already computed all the gradients that we are interested in, so we now just perform the gradient
            # descent. We have not given weights to each one of the loss terms... maybe we can experiment on that
            # I don't remember if the paper mentions something about that,
            # neither I have studied completely the WGAN-GP paper...
            optimizerD.step()

        # house-made reporting... what about using W&B??
        errD2plot.append(discriminator_error.detach())


        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        for j in range(opt.Gsteps):
            curr_generator.zero_grad()
            output = curr_discriminator(fake)
            # we want to maximize this output (this time, using the generator's parameters).
            # it is equivalent to minimize its negative:
            # (our optimizers minimizes by default, so we do this second thing).
            generator_error = -output.mean()
            generator_error.backward(retain_graph=True)

            if alpha != 0:
                # this means the reconstruction loss weight is not zero.
                # so we need to compute the reconstruction loss.
                loss = nn.MSELoss()
                if opt.mode == 'paint_train':
                    prev_reconstructed_patch = functions.quant2centers(prev_reconstructed_patch, centers)
                    plt.imsave('%s/prev_reconstructed_patch.png' % (opt.outf), functions.convert_image_np(prev_reconstructed_patch), vmin=0, vmax=1)
                final_noise_map = opt.noise_amp * z_opt + prev_reconstructed_patch
                reconstructed_image = curr_generator(final_noise_map.detach(), prev_reconstructed_patch)
                reconstruction_loss = alpha * loss(reconstructed_image, current_real_patch)
                reconstruction_loss.backward(retain_graph=True)
                reconstruction_loss = reconstruction_loss.detach()
            else:
                final_noise_map = z_opt
                reconstruction_loss = 0
            # print("Error is here...!")
            # generator_error.backward(retain_graph=True)
        optimizerG.step()

        errG2plot.append(generator_error.detach() + reconstruction_loss)
        D_real2plot.append(discr_output_real)
        D_fake2plot.append(discr_output_fake)
        z_opt2plot.append(reconstruction_loss)

        if epoch % 25 == 0 or epoch == (opt.niter - 1):
            print('scale %d:[%d/%d]' % (len(curr_generator_list), epoch, opt.niter))

        if epoch % 500 == 0 or epoch == (opt.niter - 1):
            plt.imsave('%s/fake_sample.png' % (opt.outf), functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
            plt.imsave('%s/G(z_opt).png' % (opt.outf),
                       functions.convert_image_np(curr_generator(final_noise_map.detach(),
                                                                 prev_reconstructed_patch).detach()),
                       vmin=0, vmax=1)
            # plt.imsave('%s/D_fake.png'   % (opt.outf), functions.convert_image_np(D_fake_map))
            # plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
            # plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
            # plt.imsave('%s/prev_random_patch.png'     %  (opt.outf), functions.convert_image_np(prev_random_patch), vmin=0, vmax=1)
            # plt.imsave('%s/noise.png'    %  (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
            # plt.imsave('%s/prev_reconstructed_patch.png'   % (opt.outf), functions.convert_image_np(prev_reconstructed_patch), vmin=0, vmax=1)

            torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))

        schedulerD.step()
        schedulerG.step()

    functions.save_networks(curr_generator, curr_discriminator, z_opt, opt)

    return z_opt, in_s, curr_generator


def draw_concat(list_of_generators, noise_patch_list, real_patches_pyramid,
                noise_amps_list, in_s, mode, noise_padder_layer, image_padder_layer, opt):
    '''
    Generates an up-sampled version of a previous lowest scale image patch THROUGH A GENERATIVE PROCESS.
    To do that, it uses the generator network and the set of possible lowest scale patches
    of the current scale patch. This is the reason whe it receives in input a list of generator models.

    Parameters
    ----------
    list_of_generators
    noise_patch_list: if mode is set to "rand", then this fella is only used in this function to determine
    the shape of the current scale noise vectors. if mode is set to "rec" instead, the whole list of saved noise
    maps is used to feed the generators alongside with patches of the provided image to generate an up-sampled version of
    the current patch using the generators.

    tensors that are generated...
    real_patches_pyramid
    noise_amps_list
    in_s:
    mode: if "rand", then the image is generated with the usage of new pseudo random noise maps and patches of the ù
    provided image. If set to "rec", instead, the up-sampling process will use the provided list of saved noise maps to
    produce the up-sampled output.
    noise_padder_layer
    image_padder_layer
    opt

    Returns The generated up-sampled image patch.
    -------

    '''

    fake_image_patch = in_s

    if len(list_of_generators) > 0:

        count = 0
        z = 0
        # the following pad_noise variable is defined in the train_single_scale function, with the same exact value.
        # Notice it is scale independent.
        pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)

        if opt.mode == 'animation_train':
            pad_noise = 0

        # we now iterate inside the provided lists of generators and noise maps:
        for current_generator, curr_noise_patch, real_curr, real_next, noise_amp \
                in zip(list_of_generators, noise_patch_list, real_patches_pyramid, real_patches_pyramid[1:],
                       noise_amps_list):

            if mode == 'rand':
                # We are generating new noise patches for each scale in this inner loop
                # we now manually remove the padding overhead from the current noise patch
                noise_width = curr_noise_patch.shape[2] - 2 * pad_noise
                noise_height = curr_noise_patch.shape[3] - 2 * pad_noise

                if count == 0:
                    #  count==0 means we are considering the coarsest scale generator, noise and image patch...
                    # In the coarsest scale, our noise map must be a repetition of a 2-d noise map over the four
                    # channels:
                    z = functions.generate_noise([1, noise_width, noise_height], device=opt.device)
                    z = z.expand(1, opt.nc_z, z.shape[2], z.shape[3])
                    # Notice that SinGAN-seg changes the second parameter from 3 to opt.nc_z
                    # in the previous line (w.r.t the SinGAN paper)
                else:
                    # If we are not in the coarsest scale, we generate a 4d noise map, each channel independent.
                    z = functions.generate_noise([opt.nc_z, noise_width, noise_height], device=opt.device)

                z = noise_padder_layer(z)

            if mode == 'rec':
                # 'rec' stands for reconstruction. So we are not generating noise but using the saved
                # noise maps to produce the output we want to reconstruct.
                z = curr_noise_patch

            # fake_image_patch is G(z), i.e. a fake batch, in particular, the previous fake batch
            # (with respect to the corresponding scale training.)
            # Now, in this inner "for" we are again looping through all scales (real_curr),
            # having this fake_image_patch fixed. So it could be big, (as big as the semi-last finer scale),
            # and thus, real_curr could be smaller. That is why this line of code takes only a part of
            # fake_image_patch of the same dimension of real_curr.
            fake_image_patch = fake_image_patch[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
            # we have an image patch, we just need to add the padding
            fake_image_patch = image_padder_layer(fake_image_patch)
            # we do the adding strategy not to induce the generator disregarding noise...
            z_in = noise_amp * z + fake_image_patch
            # and now we feed our generator with this "spatial noise" and the previous patch
            fake_image_patch = current_generator(z_in.detach(), fake_image_patch)
            # we have generated a fake batch using the previous patch. NOW WE DO THE UP-SAMPLING:
            # we are resizing to a scale which is > 1, thus, up-sampling...
            # notice we overwrite fake_image_patch
            fake_image_patch = imresize(fake_image_patch, 1 / opt.scale_factor, opt)
            # Once we have done the up-sampling, we ensure that our generated patch has the dimensions
            # of the generated pyramid. (the upsampling function uses a real scale factor that could produce
            # some excess of pixels in the resulting image...
            fake_image_patch = fake_image_patch[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
            count += 1

    return fake_image_patch


def train_paint(opt, Gs, Zs, reals, noise_amps_list, centers, paint_inject_scale):
    in_s = torch.full(reals[0].shape, 0, device=opt.device, dtype=torch.bool)
    scale_num = 0
    nfc_prev = 0

    while scale_num < opt.stop_scale + 1:
        if scale_num != paint_inject_scale:
            scale_num += 1
            nfc_prev = opt.nfc
            continue
        else:
            opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
            opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

            opt.out_ = functions.generate_dir2save(opt)
            opt.outf = '%s/%d' % (opt.out_, scale_num)
            try:
                os.makedirs(opt.outf)
            except OSError:
                pass

            # plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
            # plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
            plt.imsave('%s/in_scale.png' % (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

            D_curr, G_curr = init_models(opt)

            z_curr, in_s, G_curr = train_single_scale(D_curr, G_curr, reals[:scale_num + 1], Gs[:scale_num],
                                                      Zs[:scale_num], in_s, noise_amps_list[:scale_num], opt,
                                                      centers=centers)

            G_curr = functions.reset_grads(G_curr, False)
            G_curr.eval()
            D_curr = functions.reset_grads(D_curr, False)
            D_curr.eval()

            Gs[scale_num] = G_curr
            Zs[scale_num] = z_curr
            noise_amps_list[scale_num] = opt.noise_amp

            torch.save(Zs, '%s/Zs.pth' % (opt.out_))
            torch.save(Gs, '%s/Gs.pth' % (opt.out_))
            torch.save(reals, '%s/reals.pth' % (opt.out_))
            torch.save(noise_amps_list, '%s/NoiseAmp.pth' % (opt.out_))

            scale_num += 1
            nfc_prev = opt.nfc
        del D_curr, G_curr
    return


def init_models(opt):
    # generator initialization:
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    # discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    return netD, netG
