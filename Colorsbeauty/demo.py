# -*- coding: gbk -*-
import sys
import os

# 设置控制台编码为GBK
if sys.platform.startswith('win'):
    os.system('chcp 936 > nul')

from Image import Image

if __name__ == "__main__":
    img = Image("D://AIGC/Colorsbeauty/picture/img.jpg")


    # 分析色彩和谐度
    harmony_score = img.get_color_harmony_score()

