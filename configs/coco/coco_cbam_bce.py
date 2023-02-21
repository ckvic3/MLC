useCopyDecoupling = False
model = dict(
    name="cbam",
    param = dict(num_classes=80)
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
    name="bce",
    param = dict(
    useCopyDecoupling = useCopyDecoupling
    )
)

milestones = [50, 75]
warm_up = dict(
    total_iters = 500,
)

split_group = [[0, 2, 24, 26, 39, 41, 42, 43, 44, 45, 56, 57, 58, 60, 62, 63, 67, 69, 71, 72, 73, 75],
               [1, 3, 5, 7, 8, 9, 13, 15, 16, 25, 27, 28, 32, 34, 35, 38, 40, 46, 47, 48, 49, 50, 51,
                53, 55, 59, 61, 64, 65, 66, 68, 74, 77],
               [4, 6, 10, 11, 12, 14, 17, 18, 19, 20, 21, 22, 23, 29, 30, 31, 33, 36, 37, 52, 54, 70, 76, 78, 79]]

epochs = 80
lr = 2e-2

output_path = "/data/pengpeng/logs/{}_{}_{}_NoPFC".format(dataset['name'],loss['name'],model['name'])