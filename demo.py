# -*- coding: utf-8 -*-
from Image import Image

if __name__ == "__main__":
    img = Image("img.jpg")
    main_colors_octree, percents_octree = img.extract_main_colors_octree(max_colors=10)
    print("主色彩RGB值：", main_colors_octree)
    # img.show_image()
    # img.show_gray_hist()
    # img.show_rgb_gray_hist()
    # img.show_hs_hist()
    # img.show_color_moment()
    # img.show_ccv()
    # img.show_color_correlogram()
