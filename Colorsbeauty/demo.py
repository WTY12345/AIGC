import sys
import os

from Image import Image

if __name__ == "__main__":
    img = Image("D://AIGC/Colorsbeauty/picture/img.jpg")
    harmony_score = img.get_color_harmony_score()

