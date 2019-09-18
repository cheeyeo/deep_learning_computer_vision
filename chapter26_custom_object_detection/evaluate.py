from prediction_config import PredictionConfig
from dataset import KangarooDataset
from parser import extract_boxes
import numpy as np
from mrcnn.model import MaskRCNN
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image

def evaluate_model(dataset, model, cfg):
	aps = list()
	for image_id in dataset.image_ids:
		image, _, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)

		# center pixel values
		scaled_image = mold_image(image, cfg)

		# convert image to one sample
		sample = np.expand_dims(scaled_image, 0)

		yhat = model.detect(sample, verbose=0)

		r = yhat[0]

		# calculate stats including AP
		ap, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r["masks"])
		aps.append(ap)
	mAP = np.mean(aps)
	return mAP


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

train_mAP = evaluate_model(train_set, model, cfg)
print("Train mAP: {:.3f}".format(train_mAP))

test_mAP = evaluate_model(test_set, model, cfg)
print("Test mAP: {:.3f}".format(test_mAP))
