import numpy as np
from data_loader import DataLoader
from utils import compute_iou_wh


class KMeans():

    def __init__(self, n_clusters, img_size=416):
        self._n_clusters = n_clusters
        self.centers = np.random.rand(self._n_clusters,2)
        self.img_size = img_size


    def fit(self, boxes):

        boxes /= np.max(boxes,axis=0)

        centers = self.centers

        last_labels = np.zeros(boxes.shape[0])
        current_labels = np.ones_like(last_labels)

        while not (last_labels == current_labels).all():
            last_labels = current_labels

            d = 1 - compute_iou_wh(boxes, centers)
            current_labels = np.argmin(d, -1)

            for i in range(self._n_clusters):
                if len(boxes[current_labels==i]) != 0:
                    centers[i] = np.median(boxes[current_labels==i], out=centers[i], axis=0)
                else:
                    centers[i] = np.random.rand(2,)
            print('acc:',np.sum(last_labels == current_labels)/last_labels.size)

        return np.int32(centers*self.img_size)


if __name__=='__main__':
    data_loader = DataLoader()
    bboxes, labels = data_loader.generate_all()
    km = KMeans(9)
    centers = km.fit(bboxes[:,2:])
    print(centers)
        # np.savetxt('anchors.txt', np.int32(centers), delimiter=',')