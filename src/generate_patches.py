# @copy: Minivision_AI
"""
Create patch from original input image by using bbox coordinate
"""
import os

import cv2
import numpy as np


class CropImage:
    @staticmethod
    def _get_new_box(src_w, src_h, bbox, scale):
        x = bbox[0]
        y = bbox[1]
        box_w = bbox[2]
        box_h = bbox[3]

        scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))

        new_width = box_w * scale
        new_height = box_h * scale
        center_x, center_y = box_w / 2 + x, box_h / 2 + y

        left_top_x = center_x - new_width / 2
        left_top_y = center_y - new_height / 2
        right_bottom_x = center_x + new_width / 2
        right_bottom_y = center_y + new_height / 2

        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0

        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0

        if right_bottom_x > src_w - 1:
            left_top_x -= right_bottom_x - src_w + 1
            right_bottom_x = src_w - 1

        if right_bottom_y > src_h - 1:
            left_top_y -= right_bottom_y - src_h + 1
            right_bottom_y = src_h - 1

        return (
            int(left_top_x),
            int(left_top_y),
            int(right_bottom_x),
            int(right_bottom_y),
        )

    def crop(self, org_img, bbox, scale, out_w, out_h, crop=True):

        if not crop:
            dst_img = cv2.resize(org_img, (out_w, out_h))
        else:
            src_h, src_w, _ = np.shape(org_img)
            left_top_x, left_top_y, right_bottom_x, right_bottom_y = self._get_new_box(
                src_w, src_h, bbox, scale
            )

            img = org_img[
                left_top_y : right_bottom_y + 1, left_top_x : right_bottom_x + 1
            ]
            dst_img = cv2.resize(img, (out_w, out_h))
        return dst_img


def process_zalo(db_dir, save_dir, scale, crop_sz):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_list = open(save_dir + "/file_list.txt", "w")
    for file in open(db_dir + "/file_label.txt", "r"):
        file_info = file.strip("\n").split(" ")
        file_name = file_info[0]
        bboxs = file_info[1:5]
        bboxs = [int(bboxs[0]), int(bboxs[1]), int(bboxs[2]), int(bboxs[3])]
        bboxs = [bboxs[0], bboxs[1], bboxs[2] - bboxs[0], bboxs[3] - bboxs[1]]
        label = file_info[5]
        cur_save_dir = os.path.join(save_dir, *file_name.split("/")[-4:-1])

        if not os.path.exists(cur_save_dir):
            os.makedirs(cur_save_dir)

        frame = cv2.imread(file_name)
        croper = CropImage()
        croped = croper.crop(
            frame, bbox=bboxs, scale=scale, out_w=crop_sz, out_h=crop_sz, crop=True
        )
        print(croped)
        save_fname = os.path.join(cur_save_dir, file_name.split("/")[-1])
        file_list.writelines("%s %s\n" % (save_fname, label))
        cv2.imwrite(save_fname, croped, [cv2.IMWRITE_JPEG_QUALITY, 100])


if __name__ == "__main__":
    db_dir = "G:/zalo_challenge/liveness_face/antispoofing_zalo/datasets/images_train/datasets/images/"
    save_dir = "F:/zalo-challenge/datasets/"
    scales = [1.0, 4.0]
    crop_sz = 80
    for scale in scales:
        cur_save_dir = save_dir + "/scale_" + str(scale)
        process_zalo(db_dir, cur_save_dir, scale, crop_sz)
