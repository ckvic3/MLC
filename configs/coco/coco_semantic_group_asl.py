# 使用的是由kmeans生成的语义长尾分组
kd_three_group = [
    [0, 2, 3, 5, 6, 15, 16, 22, 30, 35, 38, 50, 53, 57, 59, 60, 61, 62, 63, 64, 66, 69, 71, 74], 
    [1, 7, 9, 13, 17, 20, 25, 26, 27, 32, 34, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 55, 56, 58, 65, 68, 72, 73, 75, 77], 
[4, 8, 10, 11, 12, 14, 18, 19, 21, 23, 24, 28, 29, 31, 33, 36, 51, 52, 54, 67, 70, 76, 78, 79]]

useCopyDecoupling = False
split_group = [[0, 2, 24, 26, 39, 41, 42, 43, 44, 45, 56, 57, 58, 60, 62, 63, 67, 69, 71, 72, 73, 75],
               [1, 3, 5, 7, 8, 9, 13, 15, 16, 25, 27, 28, 32, 34, 35, 38, 40, 46, 47, 48, 49, 50, 51,
                53, 55, 59, 61, 64, 65, 66, 68, 74, 77],
               [4, 6, 10, 11, 12, 14, 17, 18, 19, 20, 21, 22, 23, 29, 30, 31, 33, 36, 37, 52, 54, 70, 76, 78, 79]]

model = dict(
    name="group",
    param = dict( 
    mode="local",
    label_groups=kd_three_group,
    num_classes=80,
    freeze_max_layer=0),
    weight_path = None
)

dataset = dict(
    name = "coco",
    root = "/data/coco/",
    useCopyDecoupling = useCopyDecoupling,
    sampler=None,
    clsDataListFile = './appendix/coco/longtail2017/class_freq.pkl',
    imageSize=224,
)

loss = dict(
    name="asl",
    param = dict(
    gamma_pos=0,
    gamma_neg=4,
    disable_torch_grad_focal_loss=True,
    useCopyDecoupling = useCopyDecoupling
    )
)

milestones = [50, 75]
warm_up = dict(
    total_iters = 500,
)

epochs = 80
lr = 2e-2

output_path = "/data/pengpeng/logs/semantic_{}_{}_{}_NoPFC".format(dataset['name'],loss['name'],model['name'])
