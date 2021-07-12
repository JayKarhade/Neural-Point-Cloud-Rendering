from math import log10, sqrt
import cv2
import numpy as np

def PSNR(original, synthesis):
	mse = np.mean((original - snythesis) ** 2)
	if(mse == 0): # MSE is zero means no noise is present in the signal .
				# Therefore PSNR have no importance.
		return 100
	max_pixel = 255.0
	psnr = 20 * log10(max_pixel / sqrt(mse))
	return psnr

def main():
	original = cv2.imread("original_image.png")
	synthesis = cv2.imread("synthesized.png", 1)
	value = PSNR(original, synthesis)
	print(f"PSNR value is {value} dB")
	
if __name__ == "__main__":
	main()
