import matplotlib.image as mpimg
from matplotlib.pyplot import imshow
from glob import glob

results = glob("colab_inputs/output/*_compose.png")
print(results)
if len(results) > 0:
  print(f"Showing {results[0]}")
  testim = mpimg.imread(results[0])
  imshow(testim)