import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from mrcnn.model import mold_image
from mrcnn.model import MaskRCNN
from dataset import KangarooDataset
from prediction_config import PredictionConfig

def plot_actual_vs_predicted(dataset, model, cfg, figname, n_images=5):
	for i in range(n_images):
		# load image and mask
		image = dataset.load_image(i)
		mask, _ = dataset.load_mask(i)
		scaled_image = mold_image(image, cfg)

		sample = np.expand_dims(scaled_image, 0)
		yhat = model.detect(sample, verbose=0)[0]
		plt.subplot(n_images, 2, i*2+1)
		plt.axis('off')
		plt.imshow(image)

		if i == 0:
			plt.title('Actual')
		# plot masks
		for j in range(mask.shape[2]):
			plt.imshow(mask[:,:,j], cmap='gray', alpha=0.3)

		# get drawing context
		plt.subplot(n_images, 2, i*2+2)
		plt.axis('off')
		plt.imshow(image)
		if i == 0:
			plt.title('Predicted')
		ax = plt.gca()
		for box in yhat['rois']:
			y1, x1, y2, x2 = box
			width, height = x2 - x1, y2 - y1
			rect = Rectangle((x1, y1), width, height, fill=False, color='red')
			ax.add_patch(rect)
	plt.savefig(figname)

train_set = KangarooDataset()
train_set.load_dataset('kangaroo', is_train=True)
train_set.prepare()
print('Train: {:d}'.format(len(train_set.image_ids)))

test_set = KangarooDataset()
test_set.load_dataset('kangaroo', is_train=False)
test_set.prepare()
print('Test: {:d}'.format(len(test_set.image_ids)))

cfg = PredictionConfig()

model = MaskRCNN(mode='inference', model_dir='./', config=cfg)

model.load_weights('kangaroo_cfg20190918T1342/mask_rcnn_kangaroo_cfg_0005.h5', by_name=True)

plot_actual_vs_predicted(train_set, model, cfg, 'plot_train_set.png')

plot_actual_vs_predicted(test_set, model, cfg, 'plot_test_set.png')