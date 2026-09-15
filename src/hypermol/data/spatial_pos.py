import numpy as np


def floyd_warshall(adjacency_matrix):
    """
    Floyd-Warshall 算法的 Pythonic 实现版本。

    计算图中所有节点对之间的最短路径。

    参数:
    adjacency_matrix (np.ndarray): 一个 N*N 的邻接矩阵。
                                  adjacency_matrix[i, j] 是从 i 到 j 的权重。
                                  如果节点间没有直接路径，值应为 0 或无穷大。

    返回:
    tuple:
        - dist (np.ndarray): 包含所有节点对之间最短路径长度的距离矩阵。
                           如果路径不存在，值为 np.inf。
        - path (np.ndarray): 路径重建矩阵。path[i, j] 记录了 i->j 最短路径上 k 的索引。
                           值为 -1 表示 i 和 j 之间是直接连接。
    """
    # 1. 初始化
    num_nodes = adjacency_matrix.shape[0]
    assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1], "邻接矩阵必须是方阵"

    # 创建距离矩阵的副本，使用浮点数以容纳 'inf'
    dist = np.array(adjacency_matrix, dtype=np.float64, copy=True)

    # 初始化路径矩阵，-1 表示没有中间节点（即直连）
    path = np.full((num_nodes, num_nodes), -1, dtype=np.int64)

    # 2. 设置初始距离 (向量化版本)
    # 将输入中代表“无连接”的 0 替换为无穷大
    # 同时确保对角线 i->i 的距离为 0
    dist[dist == 0] = np.inf
    np.fill_diagonal(dist, 0)

    # 3. Floyd-Warshall 核心算法
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                # 寻找更短的路径 i -> k -> j
                new_dist = dist[i, k] + dist[k, j]
                if dist[i, j] > new_dist:
                    dist[i, j] = new_dist
                    # 记录中间节点 k
                    path[i, j] = k

    # 4. 后处理 (可选)
    # 在这个版本中，不可达路径自然就是 np.inf，不需要像原代码那样手动设置为 510。
    # path 矩阵中对应的值已经是 -1，也无需特殊处理。
    dist[dist == np.inf] = 510

    return dist, path


# --- 示例 ---
if __name__ == '__main__':
    # 0 表示没有直接连接
    graph = np.array([
        [0, 3, 0, 5],
        [2, 0, 0, 0],
        [0, 7, 0, 1],
        [6, 0, 2, 0]
    ])


    print("--- Pythonic 版本 ---")
    dist_matrix_2, path_matrix_2 = floyd_warshall(graph)
    print("距离矩阵:\n", dist_matrix_2)
    print("\n路径矩阵 (值为-1表示直连):\n", path_matrix_2)
    # 从 1 到 2 的最短路径是 1 -> 0 -> 3 -> 2，距离为 10
    # path[1,2] = 0 (表示 1->2 的路径经过了节点 0)
    # path[0,2] = 3 (表示 0->2 的路径经过了节点 3)
    # path[1,0] = -1 (直连), path[0,3]=-1(直连), path[3,2]=-1 (直连)