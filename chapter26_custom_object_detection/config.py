from mrcnn.config import Config

class KangarooConfig(Config):
	# name of configuration
	NAME = 'kangaroo_cfg'
	# num of classes, background + kangaroo
	NUM_CLASSES = 2
	STEPS_PER_EPOCH = 131
