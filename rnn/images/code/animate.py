from PIL import Image as PIL_Image
import glob

images = []
for i in [1,2]:
	for j in range(1,5+1):
		im = PIL_Image.open(f'../batch-{i}-t-{j}.png')
		images.append(im)
	images.append(PIL_Image.open(f'../batch-{i}-output.png'))

images[0].save('/tmp/minibatch-animation.gif',
               save_all=True,
               append_images=images[1:],
               duration=100,
               loop=0)
