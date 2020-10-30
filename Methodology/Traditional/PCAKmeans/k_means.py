import time
import numpy as np


def k_means_cluster(vectors, cluster_num):
    tic = time.time()
    vec_dim, vec_n = vectors.shape

    vec_class = np.zeros([1, vec_n])
    # randomly select vectors as initial cluster center
    cluster_center = np.zeros([vec_dim, cluster_num])
    for i in range(cluster_num):
        init_id = np.random.randint(0, vec_n)
        cluster_center[:, i] = vectors[:, init_id]

    cluster_changed = True
    iter_count = 0
    while cluster_changed:
        cluster_changed = False
        iter_count += 1
        # assign every vectors' class according to their distance with center
        for i in range(vec_n):
            # min_index = -1
            # min_diff = np.inf
            temp_dist = np.linalg.norm(vectors[:, i].reshape(-1, 1) - cluster_center, axis=0)
            min_index = np.argmin(temp_dist)
            # for j in range(cluster_num):
            #     temp_dist = np.linalg.norm(vectors[:, i] - cluster_center[:, j])
            #     if temp_dist < min_diff:
            #         min_diff = temp_dist
            #         min_index = j
            if vec_class[0, i] != min_index:
                cluster_changed = True
                vec_class[0, i] = min_index

        # calculate center co-ordinates based on assignment results
        for i in range(cluster_num):
            cluster_center[:, i] = np.mean(vectors[:, vec_class[0, :] == i], axis=1)
    toc = time.time()
    print("K-means has been completed, the total iter count is %d and cost time is %.2f s" % (iter_count, toc - tic))
    return cluster_center
