from os import listdir
from xml.etree import ElementTree
import numpy as np
from mrcnn.utils import Dataset
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from parser import extract_boxes
import matplotlib.pyplot as plt

class KangarooDataset(Dataset):
	def load_dataset(self, dataset_dir, is_train=True):
		self.add_class("dataset", 1, "kangaroo")
		images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annots/'

		for filename in listdir(images_dir):
			image_id = filename[:-4]

			if image_id in ['00090']:
				continue

			if is_train and int(image_id) >= 150:
				continue

			if not is_train and int(image_id) < 150:
				continue

			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

	# Load masks from annotation file
	# We don't have masks in this example so load bounding boxes to return as masks
	def load_mask(self, image_id):
		info = self.image_info[image_id]
		path = info['annotation']
		boxes, w, h = extract_boxes(path)

		masks = np.zeros([h, w, len(boxes)], dtype='uint8')
		class_ids = list()
		for i in range(len(boxes)):
		  box = boxes[i]
		  row_s, row_e = box[1], box[3]
		  col_s, col_e = box[0], box[2]
		  masks[row_s:row_e, col_s:col_e, i] = 1
		  class_ids.append(self.class_names.index('kangaroo'))

		return masks, np.asarray(class_ids, dtype='int32')

	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

if __name__ == '__main__':
	train_set = KangarooDataset()
	train_set.load_dataset('kangaroo', is_train=True)
	train_set.prepare()
	print('Train: {:d}'.format(len(train_set.image_ids)))

	# test/val set
	test_set = KangarooDataset()
	test_set.load_dataset('kangaroo', is_train=False)
	test_set.prepare()
	print('Test: {:d}'.format(len(test_set.image_ids)))

	for i in range(9):
		plt.subplot(330 + 1 + i)
		plt.axis('off')
		image = train_set.load_image(i)
		plt.imshow(image)
		mask, _ = train_set.load_mask(i)
		for j in range(mask.shape[2]):
			plt.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
	plt.savefig('test.png')

	for image_id in train_set.image_ids:
		info = train_set.image_info[image_id]
		print(info)

	image_id = 1
	image = train_set.load_image(image_id)
	mask, class_ids = train_set.load_mask(image_id)
	bbox = extract_bboxes(mask)
	display_instances(image, bbox, mask, class_ids, train_set.class_names)