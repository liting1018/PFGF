import cv2
import mmcv
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:, 0, :, :] * 0
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap += feature_map[:, c, :, :]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    # heatmap = 1 - heatmap
    heatmaps.append(heatmap)

    return heatmaps

def draw_feature_map(features, img_path, save_dir='feature_map_tirgraphmamba_neck_llvip'):
    img = mmcv.imread(img_path)
    img_name = os.path.splitext(os.path.basename(img_path))[0]  # 获取图片的文件名（不带扩展名）
    
    if isinstance(features, torch.Tensor):
        for i, heat_maps in enumerate(features):
            heat_maps = heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            for heatmap in heatmaps:
                heatmap = cv2.resize(heatmap, (256, 256))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap
                plt.imshow(superimposed_img, cmap='gray')
                plt.show()
    else:
        for i, featuremap in enumerate(features):
            heatmaps = featuremap_2_heatmap(featuremap)
            for j, heatmap in enumerate(heatmaps):
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap * 0.6 + img * 0.4  # 这里的0.4是热力图强度因子
                # superimposed_img = heatmap
                plt.imshow(superimposed_img)
                plt.show()
                # plt.savefig(superimposed_img)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                heatmap_filename = f"{img_name}_heatmap_{i}.png"
                output_path = os.path.join(save_dir, heatmap_filename)
                print(output_path)
                cv2.imwrite(output_path, superimposed_img)
                cv2.destroyAllWindows()
