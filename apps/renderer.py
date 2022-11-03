from configparser import Interpolation
from re import T
from tkinter import image_names
import numpy as np
import torch
import cv2 as cv
import glob
import os
import argparse


import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model import load_model
from utils.config import load_cfg
from utils.utils import get_mano_path, imgUtils
from dataset.dataset_utils import IMG_SIZE
from core.test_utils import InterRender

import open3d as o3d
from tqdm import tqdm
import json
from utils.vis_utils import ours_two_hand_renderer
import pickle

# def depth2world(depth, depth_camera_intrinsic):
import scipy
import skimage
import numpy as np
#from pypardiso import spsolve
from scipy.sparse.linalg import spsolve
from PIL import Image


# fill_depth_colorization.m
# Preprocesses the kinect depth image using a gray scale version of the
# RGB image as a weighting for the smoothing. This code is a slight
# adaptation of Anat Levin's colorization code:
#
# See: www.cs.huji.ac.il/~yweiss/Colorization/

def fill_depth_colorization(imgRgb=None, imgDepthInput=None, alpha=1):
    '''
    :param imgRgb: - HxWx3 matrix, the rgb image for the current frame. This must be between 0 and 1.
    :param imgDepthInput:  HxW matrix, the depth image for the current frame in absolute (meters) space.
    :param alpha: a penalty value between 0 and 1 for the current depth values.
    :return: Filled depth
    '''
    imgIsNoise = imgDepthInput == 0
    maxImgAbsDepth = np.max(imgDepthInput)
    imgDepth = imgDepthInput / maxImgAbsDepth
    imgDepth[imgDepth > 1] = 1
    (H, W) = imgDepth.shape
    numPix = H * W
    indsM = np.arange(numPix).reshape((W, H)).transpose()
    knownValMask = (imgIsNoise == False).astype(int) # valid values regions
    grayImg = skimage.color.rgb2gray(imgRgb)
    winRad = 1
    len_ = 0
    absImgNdx = 0
    len_window = (2 * winRad + 1) ** 2
    len_zeros = numPix * len_window

    cols = np.zeros(len_zeros) - 1
    rows = np.zeros(len_zeros) - 1
    vals = np.zeros(len_zeros) - 1
    gvals = np.zeros(len_window) - 1

    for j in range(W):
        for i in range(H):
            nWin = 0
            for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
                for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
                    if ii == i and jj == j:
                        continue

                    rows[len_] = absImgNdx
                    cols[len_] = indsM[ii, jj]
                    gvals[nWin] = grayImg[ii, jj]

                    len_ = len_ + 1
                    nWin = nWin + 1

            curVal = grayImg[i, j]
            gvals[nWin] = curVal
            c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin + 1])) ** 2)

            csig = c_var * 0.6
            mgv = np.min((gvals[:nWin] - curVal) ** 2)
            if csig < -mgv / np.log(0.01):
                csig = -mgv / np.log(0.01)

            if csig < 2e-06:
                csig = 2e-06

            gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
            gvals[:nWin] = gvals[:nWin] / sum(gvals[:nWin])
            vals[len_ - nWin:len_] = -gvals[:nWin]

            # Now the self-reference (along the diagonal).
            rows[len_] = absImgNdx
            cols[len_] = absImgNdx
            vals[len_] = 1  # sum(gvals(1:nWin))

            len_ = len_ + 1
            absImgNdx = absImgNdx + 1

    vals = vals[:len_]
    cols = cols[:len_]
    rows = rows[:len_]
    A = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

    rows = np.arange(0, numPix)
    cols = np.arange(0, numPix)
    vals = (knownValMask * alpha).transpose().reshape(numPix)
    G = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

    A = A + G
    b = np.multiply(vals.reshape(numPix), imgDepth.flatten('F'))

    # print ('Solving system..')

    new_vals = spsolve(A, b)
    new_vals = np.reshape(new_vals, (H, W), 'F')

    # print ('Done.')

    denoisedDepthImg = new_vals * maxImgAbsDepth

    output = denoisedDepthImg.reshape((H, W)).astype('float32')

    output = np.multiply(output, (1 - knownValMask)) + imgDepthInput

    return output

def depth_to_img(color_camera_intrinsic, depth_camera_intrinsic, extrinsic, depth_img, color):
    fx_d = depth_camera_intrinsic[0]
    fy_d = depth_camera_intrinsic[1]
    cx_d = depth_camera_intrinsic[2]
    cy_d = depth_camera_intrinsic[3]

    fx_rgb = color_camera_intrinsic[0]
    fy_rgb = color_camera_intrinsic[1]
    cx_rgb = color_camera_intrinsic[2]
    cy_rgb = color_camera_intrinsic[3]

    height = depth_img.shape[0]
    width = depth_img.shape[1]
    aligned = np.zeros((height, width, 6))

    depth_scale = 1.
    for v in range(height):
        for u in range(width):
            d = depth_img[v,u] * depth_scale
            x_over_z = ((u - cx_d)) / fx_d
            y_over_z = ((v - cy_d)) / fy_d

            z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)

            x = x_over_z * z
            y = y_over_z * z

            transformed_x = x - extrinsic[0]
            transformed_y = y - extrinsic[1]
            transformed_z = z - extrinsic[2]

            aligned[v,u,0] = transformed_x
            aligned[v,u,1] = transformed_y
            aligned[v,u,2] = transformed_z
    
    aligned_depth = np.zeros((height, width))
    fill = np.zeros((height, width))
    count = np.zeros((height, width))

    linear_expand_x = [0,0,0,0,0,0,0,0,  -1,-1,-1,-1,-1,-1,-1,-1, 1,1,1,1,1,1,1,1, -2,-2,-2,-2,-2,-2,-2,-2, 2,2,2,2,2,2,2,2]
    linear_expand_y = [-3,-4,-5,-6,-7,-8,-9,-10, -3,-4,-5,-6,-7,-8,-9,-10, -3,-4,-5,-6,-7,-8,-9,-10,-3,-4,-5,-6,-7,-8,-9,-10,-3,-4,-5,-6,-7,-8,-9,-10]

    dx = [1,0,-1,0,1,1,-1,-1,-2,-2,-2,-2,-2,-1,-1,0,0,1,1,2,2,2,2,2] + linear_expand_x + linear_expand_x  + (-1 * linear_expand_y)
    dx = np.array(dx)
    dy = [0,1,0,-1,-1,1,-1,1,-2,-1,0,1,2,-2,2,-2,2,-2,2,-2,-1,0,2,2] + linear_expand_y + (-1 * linear_expand_y) + linear_expand_x
    dy = np.array(dy)
    for v in range(height):
        for u in range(width):
            x = aligned[v,u,0]
            y = aligned[v,u,1]
            z = aligned[v,u,2]
            
            if z != 0:
                x_over_z = x / z    
                y_over_z = y / z 
                d = z * np.sqrt(1. + x_over_z**2 + y_over_z**2)
                new_u = x_over_z * fx_rgb + cx_rgb
                new_v = y_over_z * fy_rgb + cy_rgb
                if new_u > width-1 or new_v > height-1 or new_u < 0 or new_v < 0:
                    pass
                else:
                    rounded_v = int(round(new_v))
                    rounded_u = int(round(new_u))

                    # 넣을 때 주위 픽셀에 함께 넣어주어서 빈공간 채우기
                    if (aligned_depth[rounded_v][rounded_u] == 0):
                        aligned_depth[rounded_v][rounded_u] = d
                        expand_v = np.array([rounded_v] * dx.shape[0])
                        expand_u = np.array([rounded_u] * dx.shape[0])

                        fill_v = expand_v + dx
                        fill_u = expand_u + dy

                        v_mask = (fill_v <= height-1) & (fill_v >= 0) 
                        u_mask = (fill_u <= width-1) & (fill_u >= 0) 
                        mask = v_mask & u_mask

                        fill_v = fill_v[mask]
                        fill_u = fill_u[mask]
                        
                        fill[fill_v,fill_u] += d
                        count[fill_v,fill_u] += 1

                        # for d_x, d_y in zip(dx,dy):
                        #     if rounded_v + d_x > height -1 or rounded_v + d_x < 0 or rounded_u + d_y >width or rounded_u + d_y < 0:
                        #         pass
                        #     else:
                        #         fill[rounded_v + d_x][rounded_u + d_y] += d
                        #         count[rounded_v + d_x][rounded_u + d_y] += 1

                    elif (aligned_depth[rounded_v][rounded_u] > d):
                        aligned_depth[rounded_v][rounded_u] = d
                        expand_v = np.array([rounded_v] * dx.shape[0])
                        expand_u = np.array([rounded_u] * dx.shape[0])

                        fill_v = expand_v + dx
                        fill_u = expand_u + dy

                        v_mask = (fill_v <= height-1) & (fill_v >= 0) 
                        u_mask = (fill_u <= width-1) & (fill_u >= 0) 
                        mask = v_mask & u_mask

                        fill_v = fill_v[mask]
                        fill_u = fill_u[mask]
                        fill[fill_v,fill_u] = d
                        count[fill_v,fill_u] = 1
                        # for d_x, d_y in zip(dx,dy):
                        #     if rounded_v + d_x > height -1 or rounded_v + d_x < 0 or rounded_u + d_y >width or rounded_u + d_y < 0:
                        #         pass
                        #     else:
                        #         fill[rounded_v + d_x][rounded_u + d_y] = d
                        #         count[rounded_v + d_x][rounded_u + d_y] = 1
    mask = (aligned_depth == 0) & (count != 0)
    aligned_depth[mask] = fill[mask] / count[mask]

    mask =  aligned_depth * count > fill 
    aligned_depth[mask] = fill[mask] / count[mask]
    
    # for v in range(height):
    #     for u in range(width):
    #         if aligned_depth[v][u] == 0 and count[v][u] != 0.:
    #             aligned_depth[v][u] = fill[v][u] / count[v][u]

        
    return aligned_depth

def img_preprocessing(dataset):
    '''
        annotation 맨 왼쪽위 맨 오른쪽밑 좌표 뽑아서 padding: 50으로 하고 뽑뽑
    '''
    os.makedirs('./masked_test/', exist_ok=True)
    print('Start preprocessing')
    if dataset == 'RGB2Hands':
        root_path = './RGB2HANDS_Benchmark/'
        # folder_list = ['seq01_crossed/', 'seq02_occlusion/', 'seq03_shuffle/', 'seq04_scratch/']
        folder_list = ['seq04_scratch/']
        
        total_img_list_out = []
        total_camera_param = []

        # fx = focal length axis-x
        # fy = focal length axis-y
        # dx = principal point axis-x
        # dy = principal point axis-y
        with open(root_path + 'color_intrinsics.txt', "r") as f:
            color_camera_intrinsics = list(map(float, f.readline().split()))
        with open(root_path + 'depth_intrinsics.txt', "r") as f:
            depth_camera_intrinsics = list(map(float, f.readline().split()))
        with open(root_path + 'extrinsics.txt', "r") as f:
            extrinsics = list(map(float, f.readline().split()))
        idx = 1332
        for folder in folder_list:
            depth_path = f'./RGB2HANDS_Benchmark/{folder}/depth/'
            img_path = f'./RGB2HANDS_Benchmark/{folder}/color/'
            anno_path = f'./RGB2HANDS_Benchmark/{folder}/annotation/annot2D_color/'
            anno_path_list = glob.glob(os.path.join(anno_path, '*.txt'))
            img_path_list = glob.glob(os.path.join(img_path, '*.jpg')) + glob.glob(os.path.join(img_path, '*.png'))
            depth_path_list = glob.glob(os.path.join(depth_path, '*.jpg')) + glob.glob(os.path.join(depth_path, '*.png'))
            img_path_list.sort()
            depth_path_list.sort()
            os.makedirs(f'./masked_test/{folder}', exist_ok=True) 
            total_joints = []
            for anno_path in tqdm(anno_path_list):
                with open(anno_path, "r") as f:
                    for line in f:
                        joint = list(map(float, line.split()))
                        if joint[0] == 0.0 and joint[1] ==0.0:
                            continue
                        total_joints.append(joint[:2])
            total_joints = np.array(total_joints)
            Min = np.min(total_joints, axis=0)
            Max = np.max(total_joints, axis=0)

            mid = (Min + Max) / 2
            L = np.max(Max - Min) / 2 / 0.8
            # import pdb; pdb.set_trace()
            # img_list_out = []
            # for img_path in tqdm(img_path_list):
            #     img = cv.imread(img_path)
            #     cropped_img = img[int(mid[1]-L):int(mid[1] + L), int(mid[0]-L):int(mid[0] + L)]
            #     img_list_out.append(cropped_img)
            # total_img_list_out.append(img_list_out)


            M = 256 / 2 / L * np.array([[1, 0, L - mid[0]],
                                        [0, 1, L - mid[1]]])

            img_list_out = []
            for img_path, depth_path in tqdm(zip(img_path_list,depth_path_list)):
                depth_img = cv.imread(depth_path, cv.IMREAD_UNCHANGED)
                depth_img = np.array(depth_img).astype(np.float64)
                depth_img[(depth_img > 1000)] = 0
                
                img = cv.imread(img_path).astype(np.float64)
                
                aligned_depth = depth_to_img(color_camera_intrinsics, depth_camera_intrinsics, extrinsics, depth_img, img / 255.0)

                mask = np.zeros_like(aligned_depth)
                mask[(aligned_depth > 350) &(aligned_depth < 600)] = 1.0
                
                for h in range(mask.shape[0]):
                    for i in range(1,10):
                        one_mask = mask[mask.shape[0] -h -i,:] > 0
                        mask[(mask.shape[0] - h - 1), one_mask] = mask[(mask.shape[0] -h -i),one_mask]
                    for i in range(1,5):
                        if h+i > mask.shape[0] - 1:
                            break
                        one_mask = mask[h+i, :] > 0
                        mask[h, one_mask] = mask[h + i, one_mask]
        
                
                mask = np.expand_dims(mask, axis=-1)
                cv.imwrite(f'./masked_test/aligned_mask_{idx}.jpg', mask * 255.0)

                aligned_depth /= aligned_depth.max()
                aligned_depth *= 255.0
                aligned_depth = aligned_depth.astype(np.uint8)
                colormap = cv.applyColorMap(aligned_depth, cv.COLORMAP_INFERNO)
                cv.imwrite(f'./masked_test/aligned_depth_{idx}.jpg', colormap)
                
                img = img * mask
                
                cv.imwrite(f'./masked_test/masked_output_{idx}.jpg', img)
                idx += 1
                cropped_img = cv.warpAffine(img, M, dsize=(256, 256))

                cv.imwrite(f'./masked_test/masked_cropped_output_{idx}.jpg', cropped_img)
                
            
            refined_camera_intrinsics = [color_camera_intrinsics[0] *  M[0, 0],
                                         color_camera_intrinsics[1] * M[1,1],
                                         color_camera_intrinsics[2] * M[0,0] + M[0,2],
                                         color_camera_intrinsics[3] * M[1,1] + M[1,2]]

            total_camera_param.append(refined_camera_intrinsics)
        import pickle
        ## Save pickle
        with open(f"./masked_test/{folder}/rgb2_hands_total_camera_param.pickle","wb") as fw:
            pickle.dump(total_camera_param, fw)

        
        print('End preprocessing')
        return total_img_list_out, total_camera_param

    elif dataset == 'EgoHands':
        raise NotImplementedError
    else:
        raise NotImplementedError

    return 


def cut_img(img, bbox):
    cut = img[max(int(bbox[2]), 0):min(int(bbox[3]), img.shape[0]),
              max(int(bbox[0]), 0):min(int(bbox[1]), img.shape[1])]
    cut = cv.copyMakeBorder(cut,
                            max(int(-bbox[2]), 0),
                            max(int(bbox[3] - img.shape[0]), 0),
                            max(int(-bbox[0]), 0),
                            max(int(bbox[1] - img.shape[1]), 0),
                            borderType=cv.BORDER_CONSTANT,
                            value=(0, 0, 0))
    return cut

# TODO: Need to implement
def ours_renderer(opt, output_path, img_path=None, img_list=None, camera_param_list=None):

    if opt.obj_path is not None:
    
        file_dict = {} # interhand index : fild index
        with open(os.path.join(f"{opt.file_dict_path}file_dict.txt"), "r") as f:
            for line in f:
                chunks = line.split()
                file_dict[chunks[2]] = chunks[0] 

        for interhand_idx in tqdm(file_dict.keys()):
            
            file_idx = file_dict[interhand_idx]
            
            img = cv.imread(f"{img_path}{int(file_idx)}.jpg")
            obj_paths = {
                "left": f'{opt.obj_path}{file_idx}_left.obj',
                "right": f'{opt.obj_path}{file_idx}_right.obj',
            }
            with open(f"anno/{int(file_idx)}.pkl","rb") as fr:
                data_info = pickle.load(fr)
            
            renderer = ours_two_hand_renderer(img_size=256, device='cuda')

            img_out, mask_out = renderer.render(img, obj_paths, data_info['camera'])

            img_out = img_out[0].detach().cpu().numpy() * 255
            mask_out = mask_out[0].detach().cpu().numpy()[..., np.newaxis]

            bg_img = cv.resize(img, (256, 256))

            img_out = img_out * mask_out + bg_img * (1 - mask_out)
            img_out = img_out.astype(np.uint8)
            
            concated_image = cv.hconcat([img, img_out])
            cv.imwrite(os.path.join(output_path, file_idx + '_output.jpg'), concated_image)

def intaghand_renderer(opt, output_path, img_path=None, img_list=None, camera_param_list=None, subset=1000):
    # with predictions
    if opt.obj_path is not None:
        img_path_list = glob.glob(os.path.join(img_path, '*.jpg')) + glob.glob(os.path.join(img_path, '*.png')).sort()

        predictions_path_list =  glob.glob(os.path.join(opt.obj_path, '*.pkl')).sort()

        for img_path, predictions_path in zip(img_path_list, predictions_path_list):
            img_name = os.path.basename(img_path)
            if img_name.find('output.jpg') != -1:
                continue
            img_name = img_name[:img_name.find('.')]
            img = cv.imread(img_path)

            with open(predictions_path,"rb") as fr:
                params = pickle.load(fr)

            img_overlap = model.render(params, bg_img=img)
            concated_image = cv.hconcat([img, img_overlap])
            cv.imwrite(os.path.join(output_path, img_name + '_output.jpg'), concated_image)

    # RGB2Hands
    elif img_list is not None:
        #import pdb; pdb.set_trace()
        model = InterRender(cfg_path=opt.cfg,
                            model_path=opt.model,
                            render_size=opt.render_size)
        img_name = 0
        for imgs in tqdm(img_list):
            for img in tqdm(imgs):
                #cv.imwrite(os.path.join(output_path, str(img_name) + '_in.jpg'), img)
                params = model.run_model(img)
                img_overlap = model.render(params, bg_img=img)
                concated_image = cv.hconcat([img, img_overlap])
                cv.imwrite(os.path.join(output_path, str(img_name) + '_output.jpg'), concated_image)
                img_name += 1

    # No predictions
    else:
        model = InterRender(cfg_path=opt.cfg,
                            model_path=opt.model,
                            render_size=opt.render_size)
        
        img_path_list = glob.glob(os.path.join(img_path, '*.jpg')) + glob.glob(os.path.join(img_path, '*.png'))
        subset_idx = list(range(0, len(img_path_list), subset))
        # import pdb; pdb.set_trace()
        for i in tqdm(subset_idx):
            img_path = img_path_list[i]
            img_name = os.path.basename(img_path)
            if img_name.find('output.jpg') != -1:
                continue
            img_name = img_name[:img_name.find('.')]
            img = cv.imread(img_path)
            h,w,c = img.shape

            if h > opt.render_size and w > opt.render_size:
                crop_width =  opt.render_size
                crop_height =  opt.render_size
                mid_x, mid_y = w//2, h//2
                offset_x, offset_y = crop_width//2, crop_height//2
                img = img[mid_y - offset_y:mid_y + offset_y, mid_x - offset_x:mid_x + offset_x]

            params = model.run_model(img)
            img_overlap = model.render(params, bg_img=img)
            img_other_view_1 = model.render_other_view(params, theta=60)
            img_other_view_2 = model.render_other_view(params, theta=120)

            cv.imwrite(os.path.join(output_path, img_name + '_output.jpg'), img_overlap)
            concated_image = cv.hconcat([img, img_overlap, img_other_view_1, img_other_view_2])
            cv.imwrite(os.path.join(output_path, img_name + '_output.jpg'), concated_image)

def intaghand_and_ours_renderer_interhand(opt, output_path, img_path=None, root_path=None, high_resolution=False):
    if high_resolution == False:
        img_size = 256
    else:
        img_size = 1024
    model = InterRender(cfg_path=opt.cfg,
                        model_path=opt.model,
                        render_size=img_size)
    renderer = ours_two_hand_renderer(img_size=img_size, device='cuda')

    # file_dict = {} # interhand index : fild index
    # with open(os.path.join(f"{opt.file_dict_path}file_dict.txt"), "r") as f:
    #     for line in f:
    #         chunks = line.split()
    #         file_dict[chunks[2]] = chunks[0] 
    if root_path == None:
        root_path = './honggyu/'
    else:
        pass
    ours_path = root_path + 'honggyu_vis/'
    halo_baseline_path = root_path + 'halo_baseline/'

    obj_list = glob.glob(os.path.join(ours_path, '*.obj'))
    file_idx_list = list(set([i.split('/')[3].split('_')[0] for i in obj_list]))
    # file_idx_list = list(set([i.split('_')[0].split('/')[3] for i in obj_list]))
    file_idx_list.sort()
    #import pdb; pdb.set_trace()

    #import pdb; pdb.set_trace()
    # 10 단위
    for file_idx in tqdm(file_idx_list):
        print(file_idx)
        img = cv.imread(f"{img_path}{int(file_idx)}.jpg")
        ours_obj_paths = {
            "left": f'{ours_path}{file_idx}_left.obj',
            "right": f'{ours_path}{file_idx}_right.obj',
        }
        halo_baseline_obj_path = {
            "left": f'{halo_baseline_path}{file_idx}_left.obj',
            "right": f'{halo_baseline_path}{file_idx}_right.obj',
        }
        if high_resolution == False:
            up_img = img
        else:
            up_img = cv.resize(img, dsize=(0,0), fx=4, fy=4, interpolation=cv.INTER_LANCZOS4)
        with open(f"anno/{int(file_idx)}.pkl","rb") as fr:
            data_info = pickle.load(fr)
        #import pdb;pdb.set_trace()
        # Ours 
        obj_paths = [halo_baseline_obj_path, ours_obj_paths]

        total_img = [] 
        for obj_path in obj_paths:
            img_out = renderer.render(up_img, obj_path, data_info['camera'])
            img_out_1 = renderer.render_other_view_ours(up_img, obj_path, data_info['camera'], theta=90)
            img_out_2= renderer.render_other_view_ours(up_img, obj_path, data_info['camera'], theta=180)
            
            total_img.append([img_out, img_out_1, img_out_2])

        # Intaghand
        params = model.run_model(img)
        img_overlap = model.render(params, bg_img=up_img)
        img_other_view_1 = model.render_other_view(params, theta=90)
        img_other_view_2 = model.render_other_view(params, theta=180)

        concated_image = cv.hconcat([up_img, 
                                    total_img[0][0], # halo_baseline
                                    total_img[0][1], 
                                    total_img[0][2], 
                                    up_img, 
                                    img_overlap, #intaghand
                                    img_other_view_1,
                                    img_other_view_2,
                                    up_img, 
                                    total_img[1][0], # ours
                                    total_img[1][1], 
                                    total_img[1][2], 
                                    ])
        cv.imwrite(os.path.join(output_path, file_idx + '_output.jpg'), concated_image)
        break

def intaghand_and_ours_renderer_rgb2hands(opt, output_path, img_path='./digit/imgs/', obj_path='./digit/meshs/', high_resolution=False):
    if high_resolution == False:
        img_size = 256
    else:
        img_size = 1024

    model = InterRender(cfg_path=opt.cfg,
                        model_path=opt.model,
                        render_size=img_size)
    renderer = ours_two_hand_renderer(img_size=img_size, device='cuda')

    # file_dict = {} # interhand index : fild index
    # with open(os.path.join(f"{opt.file_dict_path}file_dict.txt"), "r") as f:
    #     for line in f:
    #         chunks = line.split()
    #         file_dict[chunks[2]] = chunks[0] 
    if root_path == None:
        root_path = '../'
    else:
        pass

    obj_list = glob.glob(os.path.join(obj_path, '*.obj'))
    file_idx_list = list(set([i.split('/')[-1].split('_')[0] for i in obj_list]))
    file_idx_list.sort()

    #img_list = glob.glob(os.path.join(img_path, '*.png')) + glob.glob(os.path.join(img_path, '*.jpg'))
    #import pdb; pdb.set_trace()

    #import pdb; pdb.set_trace()
    # 10 단위
    for obj_idx in file_idx_list:
        #import pdb; pdb.set_trace()
        ours_obj_paths = {
            "left": f'{obj_path}{obj_idx}_left.obj',
            "right": f'{obj_path}{obj_idx}_right.obj',
        }
        img = cv.imread(f'{img_path}{obj_idx}_output.jpg')
        if high_resolution == False:
            up_img = img
        else:
            up_img = cv.resize(img, dsize=(0,0), fx=4, fy=4, interpolation=cv.INTER_LANCZOS4)
        # Ours 
        img_out = renderer.render_rgb2hands(up_img, ours_obj_paths)
            
        # Intaghand
        params = model.run_model(img)
        img_overlap = model.render(params, bg_img=up_img)

        concated_image = cv.hconcat([
                                    up_img, 
                                    img_overlap, #intaghand
                                    up_img, 
                                    img_out, # ours
                                    ])
        cv.imwrite(os.path.join(output_path, obj_idx + '_output.jpg'), concated_image)

           



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default='misc/model/config.yaml')
    parser.add_argument("--model", type=str, default='misc/model/interhand.pth') # 'misc/model/wild_demo.pth'
    parser.add_argument("--img_path", type=str, default=None)
    parser.add_argument("--obj_path", type=str, default=None)
    parser.add_argument("--file_dict_path", type=str, default=None)
    parser.add_argument("--render_size", type=int, default=256)
    parser.add_argument("--method", type=str, default='intaghand') # 'inataghand', 'ours'
    parser.add_argument("--dataset", type=str, default='InterHand') # 'InterHand' , 'RGB2Hands', 'EgoHands'
    parser.add_argument("--render_both", action='store_true', default=False)
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--high_resolution", action='store_true', default=False)

    opt = parser.parse_args()
    #img_preprocessing('RGB2Hands')
    #raise NotImplementedError

    if opt.dataset == 'InterHand':
        if opt.img_path == None:
            img_path = './interhand_test_image/img/'
        else:
            img_path = opt.img_path
        if opt.out_path is not None:
            output_path = opt.out_path
        else:
            output_path = 'qualitative_outputs/'

        # Render all(baseline + Intaghand + ours)
        if opt.render_both == True:
            output_path = output_path + 'both/'
            os.makedirs(output_path, exist_ok=True)
            intaghand_and_ours_renderer_interhand(opt, output_path, img_path, opt.root_path, opt.high_resolution)
          
        else:
            if opt.method == 'intaghand':
                output_path = output_path + 'intaghand/'
                os.makedirs(output_path, exist_ok=True)
                intaghand_renderer(opt, output_path, img_path)
            elif opt.method == 'ours':
                output_path = output_path + 'ours/'
                os.makedirs(output_path, exist_ok=True)
                ours_renderer(opt, output_path, img_path) 
            else:
                raise NotImplementedError

    elif opt.dataset == 'RGB2Hands':
        if opt.img_path == None:
            img_path = './RGB2Hands_test_image/color/'
        else:
            img_path = opt.img_path

        if opt.out_path is not None:
            output_path = opt.out_path
        else:
            output_path = './RGB2Hands_outputs/'

        os.makedirs(output_path, exist_ok=True)

        #img_list, camera_param_list = img_preprocessing(opt.dataset)

        if opt.render_both == True:
            output_path = output_path + 'both/'
            os.makedirs(output_path, exist_ok=True)
            intaghand_and_ours_renderer_rgb2hands(opt, output_path, opt.img_path, opt.obj_path, opt.high_resolution)

        # else:
        #     if opt.method == 'intaghand':
        #         output_path = output_path + 'intaghand/'
        #         os.makedirs(output_path, exist_ok=True)
        #         intaghand_renderer(opt, output_path,  img_list=img_list)
        #     elif opt.method == 'ours':
        #         ours_renderer(opt, img_list, camera_param_list, output_path  + 'ours/') 
        #     else:
        #         raise NotImplementedError

    # elif opt.dataset == 'EgoHands':
    #     if opt.img_path == None:
    #         img_path = './EgoHands_test_image/'
    #     else:
    #         img_path = opt.img_path
    #     output_path = './EgoHands_outputs/'

    #     os.makedirs(output_path, exist_ok=True)

    # elif opt.dataset == 'test':
    #     if opt.img_path == None:
    #         img_path = './test_data/'
    #     else:
    #         img_path = opt.img_path
    #     output_path = './test_data_outputs/'

    #     os.makedirs(output_path, exist_ok=True)

    else:
        raise NotImplementedError

'''
# for rendering
python apps/renderer.py --dataset InterHand --render_both --root_path 'path to data_dir_root'
python apps/renderer.py --dataset InterHand --render_both --root_path ../render/

python apps/renderer.py --dataset InterHand --render_both --root_path ../render/ --out_path ./qualitative_outputs_high/ --high_resolution

python apps/renderer.py --dataset RGB2Hands --render_both --root_path ../ --out_path ./rgb2hand_qualitative_high/


# subdir: ours, halo_baseline

python apps/renderer.py --dataset InterHand --method ours --obj_path 'path to obj folder' --img_path 'path to image folder' --file_dict_path 'path to file_dict folder'
python apps/renderer.py --dataset InterHand --method ours --obj_path './sample_2/' --img_path './test_data/' --file_dict_path './test_data/'

python apps/renderer.py --dataset InterHand --render_both  --obj_path './sample_2/' --img_path './test_data/' --file_dict_path './test_data/'

python apps/renderer.py --dataset RGB2Hands --method intaghand
python apps/renderer.py --dataset InterHand --method intaghand
python apps/renderer.py --dataset EgoHands --method intaghand

python apps/renderer.py --dataset test --method ours
'''