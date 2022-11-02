# Rendering

## How to use

### Comparing intaghand and ours

```
python apps/renderer.py --dataset InterHand --render_both --root_path 'path to data_dir_root' --out_path 'path to output_dir' --high_resolution
```

```
python apps/renderer.py --dataset InterHand --render_both --root_path ../render/ --out_path ./qualitative_outputs_high/ --high_resolution

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