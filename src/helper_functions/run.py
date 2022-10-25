from src.helper_functions.generate_mask import multi2single
import mmcv
import numpy as np

if __name__ == '__main__':
    ann_file = "/home/pengpeng/ASL/appendix/VOCdevkit/longtail2012/img_id.txt"
    class_freq_file = "/home/pengpeng/ASL/appendix/VOCdevkit/longtail2012/class_freq.pkl"

    data = mmcv.load(class_freq_file)
    print(data['class_freq'])
    from temp import VOC_SPLIT_GROUP
    for group in VOC_SPLIT_GROUP:
        print(data['class_freq'][group])
    print("before m2s,the sample number:", data['class_freq'][0] + data['neg_class_freq'][0])
    # single_ids, masks = multi2single(ann_file, class_freq_file)
    # print("after m2s, the sample number:", len(single_ids))
    #
    # np.save("/home/pengpeng/ASL/appendix/VOCdevkit/lt2012/single_img_ids.npy",single_ids)
    # np.save("/home/pengpeng/ASL/appendix/VOCdevkit/lt2012/masks.npy",masks)