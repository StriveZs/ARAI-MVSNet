import os
import sys
import os.path as osp
import argparse, os

parser = argparse.ArgumentParser(description='Test ARAI-MVSNet.')

parser.add_argument('--filepath', type=str, help='rename point cloud',
                    default=".../ARAI-RMVSNet/outputs/dtu/test/result/result")
parser.add_argument('--out_path', type=str, help='point cloud output path.',
                    default='.../ARAI-RMVSNet/outputs/dtu/test/result/mvsnet')
parser.add_argument('--dataset', type=str, default='dtu', choices=['dtu', 'tnt', 'blendedmvs', 'eth3d'])  # dtu or tnt or blendedmvs or eth3d

args = parser.parse_args()


if args.dataset == 'dtu':
    scene_list = ["scan1", "scan4", "scan9", "scan10", "scan11", "scan12", "scan13", "scan15", "scan23",
                  "scan24", "scan29", "scan32", "scan33", "scan34", "scan48", "scan49", "scan62", "scan75",
                  "scan77", "scan110", "scan114", "scan118"]
elif args.dataset == 'tnt':
    scene_list = ['Family', 'Francis', 'Horse', 'Lighthouse', 'M60', 'Panther', 'Playground', 'Train']
elif args.dataset == 'blendedmvs':
    scene_list = ['5b7a3890fc8fcf6781e2593a', '5c189f2326173c3a09ed7ef3', '5b950c71608de421b1e7318f', '5a6400933d809f1d8200af15', '59d2657f82ca7774b1ec081d', '5ba19a8a360c7c30c1c169df', '59817e4a1bd4b175e7038d19']
elif args.dataset == 'eth3d':
    scene_list = ['lakeside', 'sand_box', 'storage_room', 'storage_room_2', 'tunnel']
# scene_list = ['Francis', 'Lighthouse']

# scene_list = ["scan77"]

if not os.path.exists(args.out_path):
    os.mkdir(args.out_path)

filepath = args.filepath
out_path = args.out_path
del_scene_list = []
for scene in scene_list:
    scan_path = osp.join(filepath, scene, 'points_arainet')
    if not os.path.exists(scan_path):
        continue
    del_scene_list.append(scan_path)
    dirs = os.listdir(scan_path)
    filename = None
    for item in dirs:
        if item[:16] == 'consistencyCheck':
            filename = item
        else:
            # 删除没用文件夹
            rm_cmd = 'rm -rf ' + scan_path + '/' + item
            os.system(rm_cmd)
    if filename is None:
        continue
    rename_path = scan_path + '/' + filename + '/' + 'final3d_model.ply'
    rename_file = 'unknow_dataset'
    if args.dataset == 'dtu':
        rename_file = out_path + '/' + 'mvsnet{:0>3}_l3.ply'.format(scene[4:])
    elif args.dataset == 'tnt':
        rename_file = out_path + '/' + '{}.ply'.format(scene)
    elif args.dataset == 'blendedmvs':
        rename_file = out_path + '/' + '{}.ply'.format(scene)
    rename_cmd = "cp -r " + rename_path + " " + rename_file
    os.system(rename_cmd)


for item in del_scene_list:
    del_cmd = 'rm -rf ' + item
    os.system(del_cmd)

print('rename over and delete over')