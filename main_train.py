from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions
import torch



if __name__ == '__main__':

    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    parser.add_argument('--gpu_id', help='GPU ID to train')
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:{}".format(opt.gpu_id))

    generator_list = []
    noise_map_list = []
    real_patch_list = []
    noise_amps_list = []
    dir2save = functions.generate_dir2save(opt)

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass

        real = functions.read_image(opt)

        functions.adjust_scales2image(real, opt)

        train(opt, generator_list, noise_map_list, real_patch_list, noise_amps_list)

        SinGAN_generate(generator_list, noise_map_list, real_patch_list, noise_amps_list, opt)
