import cv2
from cv2 import dnn_superres

# Initialize super resolution object
sr = dnn_superres.DnnSuperResImpl_create()

# Read the model
path = 'EDSR_x4.pb'
sr.readModel(path)

# Set the model and scale
sr.setModel('edsr', 12)

# If you have CUDA support
sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load the image
image = cv2.imread('test.png')

# Upsample the image
upscaled = sr.upsample(image)

# Save the upscaled image
cv2.imwrite('upscaled_test.png', upscaled)

# Traditional method - bicubic
height, width = upscaled.shape[:2]
bicubic = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

# Save the bicubic image
cv2.imwrite('bicubic_test.png', bicubic)
