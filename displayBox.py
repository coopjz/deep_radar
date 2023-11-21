
import cv2
import numpy as np


def draw_bounding_box(image_path):
    label_path = image_path.replace('/images/', '/labels/')
    label_path = label_path.replace('.jpg', '.txt')

    image = cv2.imread(image_path)
    H, W, _ = image.shape
    with open(label_path, 'r') as f:
        for line in f.readlines():
            label, x, y, w, h = line.split(' ')
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)
            x1 = (x - w/2) * W
            y1 = (y - h/2) * H
            x2 = (x + w/2) * W
            y2 = (y + h/2) * H
            cv2.rectangle(image, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 255, 0), 2)

    # adjust the window size
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # show the image
    cv2.imshow('image', image)
    # wait for any key to exit
    cv2.waitKey()
    # destroy the program by ctrl+c
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image_path = '/home/cooper/Documents/dataset/armor_dataset_v4/images/000201.jpg'
    try:
        draw_bounding_box(image_path)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
