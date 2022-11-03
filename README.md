# Rendering

## How to use

### Comparing intaghand and ours

#### Interhand
```
python apps/renderer.py --dataset InterHand --render_both --root_path 'path to data_dir_root' --out_path 'path to output_dir' --high_resolution
```

```
python apps/renderer.py --dataset InterHand --render_both --img_path {path to image/} --obj_path {path to obj file/} --out_path ./qualitative_rgb2hands_outputs_high/ --high_resolution

```

#### RGB2Hands
```
python apps/renderer.py --dataset RGB2Hands --render_both --root_path ../ --out_path ./rgb2hand_qualitative_high/

```


### For our model
```
python apps/renderer.py --dataset InterHand --method ours --obj_path 'path to obj folder' --img_path 'path to image folder' --file_dict_path 'path to file_dict folder'
'''

'''
python apps/renderer.py --dataset InterHand --method ours --obj_path './sample_2/' --img_path './test_data/' --file_dict_path './test_data/'
```

### For Intaghand
```
python apps/renderer.py --dataset RGB2Hands --method intaghand
python apps/renderer.py --dataset InterHand --method intaghand
python apps/renderer.py --dataset EgoHands --method intaghand
```