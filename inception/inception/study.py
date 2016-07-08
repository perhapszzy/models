from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from inception import image_processing
from inception.imagenet_data import ImagenetData

images, labels = image_processing.distorted_inputs(ImagenetData(subset="train"), 32, 8)

print(images)
print(labels)