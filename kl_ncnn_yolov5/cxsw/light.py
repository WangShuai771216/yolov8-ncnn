from PIL import Image
from PIL import ImageEnhance

#原始图像
image = Image.open('lena.jpg')
image.show()

#亮度增强
enh_bri = ImageEnhance.Brightness(image)
brightness = 1.5
image_brightened = enh_bri.enhance(brightness)