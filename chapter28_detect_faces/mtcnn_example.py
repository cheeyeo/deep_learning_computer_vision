# Using the mtcnn lib to perform face detection
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--filename', required=True, help='Input file')
ap.add_argument('-o', '--outfile', required=True, help='Output file of detection')
ap.add_argument('-s', '--save-faces', action='store_true', help='Save detected faces.')
ap.add_argument('-d', '--debug', action='store_true', help='Print MTCNN predictions')
args = vars(ap.parse_args())

print(args)

def draw_image_with_boxes(filename, outfile, result_list):
	data = plt.imread(filename)
	plt.imshow(data)

	ax = plt.gca()

	for result in result_list:
		x, y, width, height = result['box']
		rect = Rectangle((x, y), width, height, color='red', fill=False)
		ax.add_patch(rect)

		for _, value in result['keypoints'].items():
			dot = Circle(value, radius=2, color='red')
			ax.add_patch(dot)

	plt.savefig(outfile)


def draw_faces(filename, outfile, result_list):
	data = plt.imread(filename)
	for i in range(len(result_list)):
		x1, y1, width, height = result_list[i]['box']
		x2, y2 = x1 + width, y1 + height
		# define subplot
		plt.subplot(1, len(result_list), i+1)
		plt.axis('off')
		plt.imshow(data[y1:y2, x1:x2])
	plt.savefig(outfile)


filename = args['filename']
output = args['outfile']

pixels = plt.imread(filename)

# create detector using default weights
detector = MTCNN()

faces = detector.detect_faces(pixels)
if args['debug']:
	print('[INFO] MTCNN predictions: ', faces)

draw_image_with_boxes(filename, output, faces)

if args['save_faces']:
	fname, ext = output.split('.')
	fname = fname + '_faces'
	output2 = fname + '.' + ext
	print(output2)
	draw_faces(filename, output2, faces)