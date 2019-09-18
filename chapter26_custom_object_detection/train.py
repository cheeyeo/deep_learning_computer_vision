from dataset import KangarooDataset
from config import KangarooConfig
from mrcnn.model import MaskRCNN

train_set = KangarooDataset()
train_set.load_dataset('kangaroo', is_train=True)
train_set.prepare()
print('Train: {:d}'.format(len(train_set.image_ids)))

test_set = KangarooDataset()
test_set.load_dataset('kangaroo', is_train=False)
test_set.prepare()
print('Test: {:d}'.format(len(test_set.image_ids)))

config = KangarooConfig()
config.display()

# define model
model = MaskRCNN(mode='training', model_dir='./', config=config)

model.load_weights('data/mask_rcnn_coco.h5', by_name=True, exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])

# train output layer
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')