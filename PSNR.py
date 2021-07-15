from math import log10, sqrt
import cv2
import numpy as np

def PSNR(original, synthesis):
  mse = np.mean((original - synthesis) ** 2)
  #print(mse)
  max_pixel = 255.0
  psnr = 20 * log10(max_pixel / sqrt(mse))
  return psnr

def main():
  psnr_sum = 0
  for i in range(1,13):
    original = cv2.imread("/content/drive/MyDrive/Neural-Point-Cloud-Rendering/data/ScanNet/scene0010_00/color/"+str(i)+"99.jpg")
    original = cv2.resize(original,(640,480))
    original  = np.asarray(original[120:(120+240),160:(160+320),:])
    synthesis = cv2.imread("/content/drive/MyDrive/Neural-Point-Cloud-Rendering/ScanNet_npcr_scene0010_00/TestResultpytorch/"+"{0:0=6d}".format(i)+".png", 1)
    value = PSNR(original, synthesis)
    psnr_sum = psnr_sum + value
    print(f"PSNR value is {value} dB")
  print(f"Averaged PSNR value is {value} dB")
	
if __name__ == "__main__":
  main()
