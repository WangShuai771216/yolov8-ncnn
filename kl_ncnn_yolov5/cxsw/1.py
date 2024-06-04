import ctypes


cpp = ctypes.CDLL('./DisplayImage.so')

cpp.cm('./demo-ysp.jpg')