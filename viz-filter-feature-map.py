import torch
import models.common
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import cv2 as cv
from torchvision import transforms
import math
import argparse
import os
import yaml

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='path to model')
    ap.add_argument('--cfg', required=True, help='path to cfg(.yaml) file of model')
    ap.add_argument('--image', required=True, help='path to image')
    ap.add_argument('--name', required=True, help='name of output folder')
    args = vars(ap.parse_args())
    print('\n')
    print(args)

    # Model load
    model = torch.load(args['model'])

    # Entire Model View -> Save to txt file
    f = open('./model-log.txt', 'w')
    print(model, file=f)
    f.close()
    
    conv_layers_count = []
    weights = []

    layers = model['model'].model


    """
    Not Completed Coded. Be Careful when using.
    # Visualize Filter ======================================================

    # Parse conv and Save conv's layer value
    for i in range(len(model['model'].model)):
        if type(model['model'].model[i]) == models.common.Conv:
            weights.append(model['model'].model[i].conv)
            conv_layers_count.append(i)

    for i in range(len(weights)):

        filters = weights[i].weight

        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)

        #print(filters.shape)
        n_filters = filters.shape[0]
        n_input_filters = filters.shape[1]
        #print('layer'+str(conv_layers_count[i])+' filters\n', n_filters)

        plt.figure(figsize=(4, 4))
        for j in range(n_filters):

            for k in range(n_input_filters):
                ax = plt.subplot(n_filters, n_input_filters, (n_input_filters*j)+k+1)
                plt.imshow(filters[j,k,:,:].cpu().detach(), cmap='gray')
                plt.axis('off')

        ax.set(xlabel = 'Input Channel Shape', ylabel = 'Output Channel Shape')

        fname = 'conv_layer_{}_Filter.jpg'.format(i)
        plt.savefig(os.getcwd() + '/visualize-filter/filters/' + args['name'] + '/' + fname, dpi=200)
    # ==========================================================================
    """

    # Visualize Feature Map =========================================================

    # Load Img and Pre-Processing ====================
    img = cv.imread(args['image'])
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((224, 224)), # Can change
                                transforms.ToTensor()])

    im = np.array(img)
    im = transform(im)
    im = im.unsqueeze(0)
    
    im = torch.as_tensor(im, dtype=torch.half, device=0)
    # =================================================

    # Getting Each Layer's Result =====================
    conv_out = [layers[0](im)]
    conv_layers_index = []

    # Getting Input Layer Infomation =====
    with open(args['cfg']) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

        for i in range(len(data['backbone'])):
            if type(data['backbone'][i][0]) == int:
                conv_layers_index.append([data['backbone'][i][0]])
            else:
                conv_layers_index.append(data['backbone'][i][0])

        for i in range(len(data['head']) - 1):
            if type(data['head'][i][0]) == int:
                conv_layers_index.append([data['head'][i][0]])
            else:
                conv_layers_index.append(data['head'][i][0])
    # ====================================
    
    # Apply Img to Model and Save Result =====
    for i in range(1, len(layers)-1):

        index = conv_layers_index[i]

        if len(index) == 1:
            src = conv_out[index[0]]
        else:
            src = [conv_out[j] for j in index]

        conv_out.append(layers[i](src))
        # Due to Error {AttributeError: 'Upsample' object has no attribute 'recompute_scale_factor'},
        # I edited "anaconda3/lib/python3.9/site-packages/torch/nn/modules/upsampling.py", line 157.
        # Just comment out "recompute_scale_factor=self.recompute_scale_factor"
        """
        Like this -> return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners,
                         # recompute_scale_factor=self.recompute_scale_factor
                         )
        """
    # =========================================

    results = conv_out
    # =================================================

    """
    for i in range(len(results)):
        
        print("i: ", i, results[i].shape)
    """

    # Make Output Folder
    print('\nCreate Output Folder to ' + os.getcwd() + '/visualize-filter/feature-map' +  '...\n')
    os.makedirs(os.getcwd() + '/visualize-filter/feature-map/' + args['name'], exist_ok=True)

    # Edit & Plot Result ==============================
    for i in range(len(results)):

        print("layer (" + str(i) + "/" + str(len(results)-1) + ") Feature Map Processing... ")
        
        conv_layer_vis = results[i][0, :, :, :]
        conv_layer_vis = conv_layer_vis.data

        n_feature_maps = conv_layer_vis.size(0)
        #print('n_feature_maps : ', n_feature_maps)
        if math.log2(n_feature_maps) % 2 == 0:
            x = (int)(math.log2(n_feature_maps) / 2)
            y = x
        else:
            x = (int)(math.log2(n_feature_maps) // 2)
            y = x + 1

        #print('x :', x, 'y : ', y)

        plt.figure(figsize=(6, 6))
        for j in range(n_feature_maps):

            #print('j : ', j)
            plt.subplot(2**x, 2**y, j+1)
            plt.imshow(conv_layer_vis[j, :, :].cpu().detach(), cmap='gray')
            plt.axis('off')
    
        fname = 'conv_layer_{}_Feature_Map.jpg'.format(i)
        plt.savefig(os.getcwd() + '/visualize-filter/feature-map/' + args['name'] + '/' + fname, dpi=200)
    # ==================================================
    # ==============================================================================

    print('Program is terminated.\n')