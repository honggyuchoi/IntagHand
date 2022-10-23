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
def ours_renderer(opt, img_path, output_path, test=True):

    if test:
        img_path_list = glob.glob(os.path.join(img_path, '*.jpg')) + glob.glob(os.path.join(img_path, '*.png'))
        
        # with open(f'./test_data/InterHand2.6M_test_camera.json') as f:
        #     camera_info = json.load(f)
        # with open(f'./test_data/InterHand2.6M_test_data.json') as f:
        #     data_info = json.load(f)

        file_dict = {} # interhand index : fild index
        with open(os.path.join("./test_data/file_dict.txt"), "r") as f:
            for line in f:
                chunks = line.split()
                file_dict[chunks[2]] = chunks[0] 


        # import pickle
        # datas = []

        # for interhand_idx in tqdm(file_dict.keys()):

        #     camera_id = str(data_info['images'][int(interhand_idx)]['camera'])
        #     capture_id = str(data_info['images'][int(interhand_idx)]['capture'])
        #     width = int(data_info['images'][int(interhand_idx)]['width'])
        #     height = int(data_info['images'][int(interhand_idx)]['height'])

        #     data = {
        #         'campos': camera_info[capture_id]['campos'][camera_id],
        #         'camrot': camera_info[capture_id]['camrot'][camera_id],
        #         'focal': camera_info[capture_id]['focal'][camera_id],
        #         'princpt': camera_info[capture_id]['princpt'][camera_id],
        #         'width': width,
        #         'height': height,
        #     }
        #     datas.append(data)

        # with open('camera_info.pickle', 'wb') as f:
        #     pickle.dump(datas, f)

        with open("camera_info.pickle","rb") as fr:
            camera_info = pickle.load(fr)


        for i, interhand_idx in tqdm(enumerate(file_dict.keys())):
            #metadata = loader.load_metadata(int(interhand_idx))
            
            file_idx = file_dict[interhand_idx]
            
            # camera_id = str(data_info['images'][int(interhand_idx)]['camera'])
            # capture_id = str(data_info['images'][int(interhand_idx)]['capture'])

            # cam_param = {
            #     'campos': camera_info[capture_id]['campos'][camera_id],
            #     'camrot': camera_info[capture_id]['camrot'][camera_id],
            #     'focal': camera_info[capture_id]['focal'][camera_id],
            #     'princpt': camera_info[capture_id]['princpt'][camera_id],
            # }
            #cam_param = camera_info[i]

            img = cv.imread(f"./test_data/{int(file_idx)}.jpg")
            obj_paths = {
                "left": f'./sample_2/{file_idx}_left.obj',
                "right": f'./sample_2/{file_idx}_right.obj',
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

    # if opt.predictions_path is not None:
    #     raise NotImplementedError
    
    # else:
    #     model = InterRender(cfg_path=opt.cfg,
    #                         model_path=opt.model,
    #                         render_size=opt.render_size)
        
    #     img_path_list = glob.glob(os.path.join(img_path, '*.jpg')) + glob.glob(os.path.join(img_path, '*.png'))
    #     for img_path in img_path_list:
    #         img_name = os.path.basename(img_path)
    #         if img_name.find('output.jpg') != -1:
    #             continue
    #         img_name = img_name[:img_name.find('.')]
    #         img = cv.imread(img_path)
    #         params = model.run_model(img)
    #         img_overlap = model.render(params, bg_img=img)
    #         concated_image = cv.hconcat([img, img_overlap])
    #         cv.imwrite(os.path.join(output_path, img_name + '_output.jpg'), concated_image)

def intaghand_renderer(opt, img_path, output_path):
    # with predictions
    if opt.predictions_path is not None:
        img_path_list = glob.glob(os.path.join(img_path, '*.jpg')) + glob.glob(os.path.join(img_path, '*.png')).sort()

        predictions_path_list =  glob.glob(os.path.join(opt.predictions_path, '*.pkl')).sort()

        for img_path, predictions_path in zip(img_path_list, predictions_path_list):
            img_name = os.path.basename(img_path)
            if img_name.find('output.jpg') != -1:
                continue
            img_name = img_name[:img_name.find('.')]
            img = cv.imread(img_path)
            params = model.run_model(img)
            img_overlap = model.render(params, bg_img=img)
            concated_image = cv.hconcat([img, img_overlap])
            cv.imwrite(os.path.join(output_path, img_name + '_output.jpg'), concated_image)

    # No predictions
    else:
        model = InterRender(cfg_path=opt.cfg,
                            model_path=opt.model,
                            render_size=opt.render_size)
        
        img_path_list = glob.glob(os.path.join(img_path, '*.jpg')) + glob.glob(os.path.join(img_path, '*.png'))
        for img_path in tqdm(img_path_list):
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
            cv.imwrite(os.path.join(output_path, img_name + '_output.jpg'), img_overlap)
            concated_image = cv.hconcat([img, img_overlap])
            cv.imwrite(os.path.join(output_path, img_name + '_output.jpg'), concated_image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default='misc/model/config.yaml')
    parser.add_argument("--model", type=str, default='misc/model/wild_demo.pth')
    parser.add_argument("--img_path", type=str, default=None)
    parser.add_argument("--predictions_path", type=str, default=None)
    parser.add_argument("--live_demo", action='store_true')
    parser.add_argument("--render_size", type=int, default=256)
    parser.add_argument("--method", type=str, default='intaghand') # 'inataghand', 'ours'
    parser.add_argument("--dataset", type=str, default='InterHand') # 'InterHand' , 'RGB2Hands', 'EgoHands'

    opt = parser.parse_args()

    if opt.dataset == 'InterHand':
        if opt.img_path == None:
            img_path = './interhand_test_image/img/'
        else:
            img_path = opt.img_path
        output_path = 'interhand_outputs'
    elif opt.dataset == 'RGB2Hands':
        if opt.img_path == None:
            img_path = './RGB2Hands_test_image/color/'
        else:
            img_path = opt.img_path
        output_path = './RGB2Hands_outputs/'
    elif opt.dataset == 'EgoHands':
        if opt.img_path == None:
            img_path = './EgoHands_test_image/'
        else:
            img_path = opt.img_path
        output_path = './EgoHands_outputs/'
    elif opt.dataset == 'test':
        if opt.img_path == None:
            img_path = './test_data/'
        else:
            img_path = opt.img_path
        output_path = './test_data_outputs/'

    else:
        raise NotImplementedError
    
    os.makedirs(output_path, exist_ok=True)
    
    if opt.method == 'intaghand':
        intaghand_renderer(opt, img_path, output_path)
    elif opt.method == 'ours':
        ours_renderer(opt, img_path, output_path) 
    else:
        raise NotImplementedError

'''
python apps/renderer.py --dataset RGB2Hands --method intaghand
python apps/renderer.py --dataset InterHand --method intaghand
python apps/renderer.py --dataset EgoHands --method intaghand

python apps/renderer.py --dataset test --method ours
'''