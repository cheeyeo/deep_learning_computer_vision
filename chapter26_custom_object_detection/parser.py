from xml.etree import ElementTree

def extract_boxes(filename):
	tree = ElementTree.parse(filename)

	root = tree.getroot()

	boxes = list()
	for box in root.findall('.//bndbox'):
		xmin = int(box.find('xmin').text)
		ymin = int(box.find('ymin').text)
		xmax = int(box.find('xmax').text)
		ymax = int(box.find('ymax').text)
		coors = [xmin, ymin, xmax, ymax]
		boxes.append(coors)
	width = int(root.find('.//size/width').text)
	height = int(root.find('.//size/height').text)
	return boxes, width, height

if __name__ == '__main__':
	boxes, w, h = extract_boxes('kangaroo/annots/00001.xml')
	print(boxes, w, h)