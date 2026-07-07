# import argparse
# import logging
# import os

# import numpy as np
# import torch
# import torch.nn.functional as F
# from PIL import Image
# from torchvision import transforms

# from utils.data_loading import BasicDataset
# from unet import UNet
# from utils.utils import plot_img_and_mask

# def predict_img(net,
#                 full_img,
#                 device,
#                 scale_factor=1,
#                 out_threshold=0.5):
#     net.eval()
#     img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
#     img = img.unsqueeze(0)
#     img = img.to(device=device, dtype=torch.float32)

#     with torch.no_grad():
#         output = net(img).cpu()
#         output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
#         if net.n_classes > 1:
#             mask = output.argmax(dim=1)
#         else:
#             mask = torch.sigmoid(output) > out_threshold

#     return mask[0].long().squeeze().numpy()


# def get_args():
#     parser = argparse.ArgumentParser(description='Predict masks from input images')
#     parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
#                         help='Specify the file in which the model is stored')
#     parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
#     parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
#     parser.add_argument('--viz', '-v', action='store_true',
#                         help='Visualize the images as they are processed')
#     parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
#     parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
#                         help='Minimum probability value to consider a mask pixel white')
#     parser.add_argument('--scale', '-s', type=float, default=0.5,
#                         help='Scale factor for the input images')
#     parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
#     parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
#     return parser.parse_args()


# def get_output_filenames(args):
#     def _generate_name(fn):
#         return f'{os.path.splitext(fn)[0]}_OUT.png'

#     return args.output or list(map(_generate_name, args.input))


# def mask_to_image(mask: np.ndarray, mask_values):
#     if isinstance(mask_values[0], list):
#         out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
#     elif mask_values == [0, 1]:
#         out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
#     else:
#         out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

#     if mask.ndim == 3:
#         mask = np.argmax(mask, axis=0)

#     for i, v in enumerate(mask_values):
#         out[mask == i] = v

#     return Image.fromarray(out)


# if __name__ == '__main__':
#     args = get_args()
#     logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#     in_files = args.input
#     out_files = get_output_filenames(args)

#     net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logging.info(f'Loading model {args.model}')
#     logging.info(f'Using device {device}')

#     net.to(device=device)
#     state_dict = torch.load(args.model, map_location=device)
#     mask_values = state_dict.pop('mask_values', [0, 1])
#     net.load_state_dict(state_dict)

#     logging.info('Model loaded!')

#     for i, filename in enumerate(in_files):
#         logging.info(f'Predicting image {filename} ...')
#         img = Image.open(filename)

#         mask = predict_img(net=net,
#                            full_img=img,
#                            scale_factor=args.scale,
#                            out_threshold=args.mask_threshold,
#                            device=device)

#         if not args.no_save:
#             out_filename = out_files[i]
#             result = mask_to_image(mask, mask_values)
#             result.save(out_filename)
#             logging.info(f'Mask saved to {out_filename}')

#         if args.viz:
#             logging.info(f'Visualizing results for image {filename}, close to continue...')
#             plot_img_and_mask(img, mask)




import argparse
import logging
import os

import torch
import numpy as np
from PIL import Image

from models.baf_unet import BAFUNet


def predict_img(net, full_img, device, out_threshold=0.5):
    net.eval()
    
    # Preprocess
    img = full_img.resize((512, 512))
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = net(img_tensor)
        output = torch.sigmoid(output)
        mask = (output > out_threshold).float()

    # Return as numpy (for saving)
    return mask.squeeze().cpu().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images using BAF-UNet')
    parser.add_argument('--model', '-m', default='checkpoints/best_model.pth', metavar='FILE',
                        help='Path to the trained model')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', required=True,
                        help='Filenames of input images')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+',
                        help='Filenames of output masks (optional)')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images')
    parser.add_argument('--no-save', '-n', action='store_true',
                        help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability to consider as lesion')

    return parser.parse_args()


def get_output_filenames(args):
    if args.output:
        return args.output
    return [f'{os.path.splitext(fn)[0]}_mask.png' for fn in args.input]


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    net = BAFUNet(n_channels=3, n_classes=1).to(device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    logging.info(f'Model loaded from {args.model} on {device}')

    in_files = args.input
    out_files = get_output_filenames(args)

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting {filename} ...')
        
        img = Image.open(filename).convert('RGB')
        mask = predict_img(net, img, device, args.mask_threshold)

        # Save mask
        if not args.no_save:
            out_filename = out_files[i]
            mask_img = (mask * 255).astype(np.uint8)
            Image.fromarray(mask_img).save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(img)
            ax[0].set_title('Input Image')
            ax[1].imshow(mask, cmap='gray')
            ax[1].set_title('Predicted Mask')
            plt.show()

    logging.info('Done!')