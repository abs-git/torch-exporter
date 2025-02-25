import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn_extra.cluster import KMedoids


def align_centers_to_data(data, centers):
    # 각 초기 중심값에 대해 가장 가까운 데이터 포인트를 선택
    aligned_centers = []
    for center in centers:
        closest_point_idx = np.argmin(cdist([center], data, metric="cosine"))
        aligned_centers.append(data[closest_point_idx])
    return np.array(aligned_centers)

def affi(features):
    all_features = [] # 프레임 별로 추출된 feature가 저장되어 있는 리스트.

    if isinstance(features, torch.Tensor):
        all_features.append(features.detach().cpu().numpy())
    elif isinstance(features, np.ndarray):
        all_features.append(features)

    clustering_engine = None
    for features in all_features:
        if clustering_engine == None:
            clustering_engine = KMedoids(n_clusters=2, random_state=42, max_iter=10, metric="cosine", init="k-medoids++")
            clustering_engine.fit(features)
            pred_cluster = clustering_engine.predict(features).tolist()
        else:
            prev_centers = clustering_engine.cluster_centers_
            prev_centers_aligned = align_centers_to_data(features, prev_centers)

            clustering_engine = KMedoids(n_clusters=2, random_state=42, max_iter=10, metric="cosine", init=prev_centers_aligned)
            clustering_engine.fit(features)
            pred_cluster = clustering_engine.predict(features).tolist()

    return pred_cluster
