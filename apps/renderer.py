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


def img_preprocessing(dataset):
    '''
        annotation 맨 왼쪽위 맨 오른쪽밑 좌표 뽑아서 padding: 50으로 하고 뽑뽑
    '''
    print('Start preprocessing')
    if dataset == 'RGB2Hands':
        root_path = './RGB2HANDS_Benchmark/'
        folder_list = ['seq01_crossed/', 'seq02_occlusion/', 'seq03_shuffle/', 'seq04_scratch/']
        
        total_img_list_out = []
        total_camera_param = []

        # fx = focal length axis-x
        # fy = focal length axis-y
        # dx = principal point axis-x
        # dy = principal point axis-y
        with open(root_path + 'color_intrinsics.txt', "r") as f:
            camera_intrinsics = list(map(float, f.readline().split()))
        # with open(root_path + 'extrinsics.txt', "r") as f:
        #     extrinsics = list(map(float, f.readline().split()))
        for folder in folder_list:
            
            img_path = f'./RGB2HANDS_Benchmark/{folder}/color/'
            anno_path = f'./RGB2HANDS_Benchmark/{folder}/annotation/annot2D_color/'
            anno_path_list = glob.glob(os.path.join(anno_path, '*.txt'))
            img_path_list = glob.glob(os.path.join(img_path, '*.jpg')) + glob.glob(os.path.join(img_path, '*.png'))


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
            for img_path in tqdm(img_path_list):
                img = cv.imread(img_path)
                img_list_out.append(cv.warpAffine(img, M, dsize=(256, 256)))
            total_img_list_out.append(img_list_out)
            refined_camera_intrinsics = [camera_intrinsics[0] *  M[0, 0],
                                         camera_intrinsics[1] * M[1,1],
                                         camera_intrinsics[2] * M[0,0] + M[0,2],
                                         camera_intrinsics[3] * M[1,1] + M[1,2]]

            total_camera_param.append(refined_camera_intrinsics)
        
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

def intaghand_and_ours_renderer_interhand(opt, output_path, img_path=None, root_path=None):
    model = InterRender(cfg_path=opt.cfg,
                        model_path=opt.model,
                        render_size=1024)
    renderer = ours_two_hand_renderer(img_size=1024, device='cuda')

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

def intaghand_and_ours_renderer_rgb2hands(opt, output_path, img_lists, camera_param_list, img_path=None, root_path=None):
    model = InterRender(cfg_path=opt.cfg,
                        model_path=opt.model,
                        render_size=1024)
    renderer = ours_two_hand_renderer(img_size=1024, device='cuda')

    # file_dict = {} # interhand index : fild index
    # with open(os.path.join(f"{opt.file_dict_path}file_dict.txt"), "r") as f:
    #     for line in f:
    #         chunks = line.split()
    #         file_dict[chunks[2]] = chunks[0] 
    if root_path == None:
        root_path = '../'
    else:
        pass
    ours_path = root_path + 'rgb2hands_results/'

    obj_list = glob.glob(os.path.join(ours_path, '*.obj'))
    file_idx_list = list(set([i.split('/')[2].split('_')[0] for i in obj_list]))
    # file_idx_list = list(set([i.split('_')[0].split('/')[3] for i in obj_list]))
    file_idx_list.sort()
    #import pdb; pdb.set_trace()

    #import pdb; pdb.set_trace()
    # 10 단위
    idx = 0 
    for i in range(len(img_lists)):
        camera = camera_param_list[i]
        camera = torch.Tensor(camera)
        img_list = img_lists[i]
        for img in img_list:
            #import pdb; pdb.set_trace()
            file_idx = file_idx_list[idx]
            print(file_idx)
            ours_obj_paths = {
                "left": f'{ours_path}{file_idx}_left.obj',
                "right": f'{ours_path}{file_idx}_right.obj',
            }
            up_img = cv.resize(img, dsize=(0,0), fx=4, fy=4, interpolation=cv.INTER_LANCZOS4)
            # Ours 
            img_out = renderer.render_rgb2hands(up_img, ours_obj_paths, camera)
                
            # Intaghand
            params = model.run_model(img)
            img_overlap = model.render(params, bg_img=up_img)

            concated_image = cv.hconcat([
                                        up_img, 
                                        img_overlap, #intaghand
                                        up_img, 
                                        img_out, # ours
                                        ])
            cv.imwrite(os.path.join(output_path, file_idx + '_output.jpg'), concated_image)
            idx+=1
           



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

    opt = parser.parse_args()

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
            intaghand_and_ours_renderer_interhand(opt, output_path, img_path, opt.root_path)
          
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

        img_list, camera_param_list = img_preprocessing(opt.dataset)

        if opt.render_both == True:
            output_path = output_path + 'both/'
            os.makedirs(output_path, exist_ok=True)
            intaghand_and_ours_renderer_rgb2hands(opt, output_path, img_list, camera_param_list, img_path, opt.root_path)

        else:
            if opt.method == 'intaghand':
                output_path = output_path + 'intaghand/'
                os.makedirs(output_path, exist_ok=True)
                intaghand_renderer(opt, output_path,  img_list=img_list)
            elif opt.method == 'ours':
                ours_renderer(opt, img_list, camera_param_list, output_path  + 'ours/') 
            else:
                raise NotImplementedError

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

python apps/renderer.py --dataset InterHand --render_both --root_path ../render/ --out_path ./qualitative_outputs_high/

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