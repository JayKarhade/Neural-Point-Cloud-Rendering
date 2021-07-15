# import the necessary packages
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
def SSIM(imageA, imageB, title):
  m = mse(imageA, imageB)
  s = ssim(imageA, imageB)
  fig = plt.figure(title)
  plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
  # show first image
  ax = fig.add_subplot(1, 2, 1)
  plt.imshow(imageA, cmap = plt.cm.gray)
  plt.axis("off")
  # show the second image
  ax = fig.add_subplot(1, 2, 2)
  plt.imshow(imageB, cmap = plt.cm.gray)
  plt.axis("off")
  # show the images
  plt.show()
  return s

ssim_sum=0
for i in range(1,13):
  original = cv2.imread("/content/drive/MyDrive/Neural-Point-Cloud-Rendering/data/ScanNet/scene0010_00/color/"+str(i)+"99.jpg")
  original = cv2.cvtColor(original,cv2.COLOR_BGR2RGB)
  original = cv2.resize(original,(640,480))
  #print(original.shape)
  original  = np.asarray(original[120:(120+240),160:(160+320),:])
  contrast = cv2.imread("/content/drive/MyDrive/Neural-Point-Cloud-Rendering/ScanNet_npcr_scene0010_00/TestResultpytorch/"+"{0:0=6d}".format(i)+".png")
  original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
  contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)

  # show the figure
  #plt.show()
  # compare the images
  print(i)
  s = SSIM(original, contrast, "Original vs. Contrast")
  ssim_sum = ssim_sum + s

print(ssim_sum/12) 
