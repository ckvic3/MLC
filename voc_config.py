voc_one_group = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]


voc_kn_two_group = [[8, 14], [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]]

voc_kn_three_group = [[0, 1, 2, 3, 5, 7, 9, 11, 12, 13, 15, 16, 17, 18, 19], [8, 14], [4, 6, 10]]

voc_kn_four_group = [[0, 1, 2, 3, 5, 7, 9, 11, 12, 13, 15, 16, 17, 18, 19], [14], [8], [4, 6, 10]]

VOC_K_SAMPLE_NUMBER_GROUPS = [voc_one_group, voc_kn_two_group, voc_kn_three_group,voc_kn_four_group]




#  需要指定VOC的 GROUP
voc_kd_two_group = [[0, 1, 2, 3, 5, 6, 7, 10, 11, 12, 13, 14, 17, 18, 19], [4, 8, 9, 15, 16]]

voc_kd_three_group = [[4, 9, 16], [0, 1, 3, 6, 7, 10, 11, 13, 14, 18, 19], [2, 5, 8, 12, 15, 17]]

voc_kd_four_group = [[2, 5, 8, 12, 15, 17], [9], [0, 1, 3, 6, 7, 10, 11, 13, 14, 18, 19], [4, 16]]


VOC_K_DIFFICULTY_GROUPS = [voc_one_group,voc_kd_two_group,voc_kd_three_group,voc_kd_four_group]