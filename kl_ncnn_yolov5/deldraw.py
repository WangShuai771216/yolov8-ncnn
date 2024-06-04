from glob import glob
from PIL import Image
import os

b = glob("build/examples/ai_result_dydy_062/*_draw.jpg")
for i in b:
    os.remove(i)

b = glob("build/examples/ai_result_dydy_062/*_raw.txt")
for i in b:
    os.remove(i)

# b = glob("build/examples/ai_result_xy_062/*_draw.jpg")
# for i in b:
#     os.remove(i)
#
# b = glob("build/examples/ai_result_xy_062/*_raw.txt")
# for i in b:
#     os.remove(i)

