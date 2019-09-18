from mrcnn.config import Config

class PredictionConfig(Config):
	NAME = "kangaroo_cfg"
	NUM_CLASSES = 2
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1