# Rendering

## How to use

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