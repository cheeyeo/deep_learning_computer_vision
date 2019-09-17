from mrcnn.config import Config

class TestConfig(Config):
	NAME = "test"
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	NUM_CLASSES = 81