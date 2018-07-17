import xml.etree.ElementTree as ET
import os
import cv2

xmlFolder = '/home/ankit/Documents/Ankit-BackUp/Develop/LiverVision/Instrument Detection/dataset/train_annot_folder/'
newxmlFolder = '/home/ankit/Documents/Ankit-BackUp/Develop/LiverVision/Instrument Detection/dataset/train_annot_folder_new/'

for filename in os.listdir(xmlFolder):
    if not filename.endswith('.xml'):
        continue
    fullname = os.path.join(xmlFolder, filename)
    tree = ET.parse(fullname)
    root = tree.getroot()
    for elem in root:
        if elem.tag == 'size':
            for var in elem:
                if var.tag == 'width':
                    var.text = str(512)
                if var.tag == 'height':
                    var.text = str(512)
        if elem.tag == 'object':
            for var in elem:
                if var.tag == 'bndbox':
                    for prop in var:
                        if prop.tag == 'xmin':
                            prop.text = str(int(int(prop.text)*0.53333))
                        if prop.tag == 'xmax':
                            prop.text = str(int(int(prop.text) * 0.53333))
                        if prop.tag == 'ymin':
                            prop.text = str(int(int(prop.text)*0.94814))
                        if prop.tag == 'ymax':
                            prop.text = str(int(int(prop.text) * 0.94814))
    newxmlName = (newxmlFolder+filename)
    tree.write(newxmlName)







# path = '/home/ankit/Documents/Ankit-BackUp/Develop/LiverVision/Instrument Detection/dataset/train_annot_folder'
# xmlsavePath = '/home/ankit/Documents/Ankit-BackUp/Develop/LiverVision/Instrument Detection/dataset/train_annot_folder'
# newPathName = 'C:/Develop/LiverVision/Instrument Detection/dataset/train_image_folder/'
#
# for filename in os.listdir(path):
#     if not filename.endswith('.xml'):
#         continue
#     fullname = os.path.join(path, filename)
#     tree = ET.parse(fullname)
#     root = tree.getroot()
#     for child in root:
#         if child.tag == 'path':
#             child.text = newPathName
#     newxmlName = os.path.join(xmlsavePath, filename)
#     tree.write(newxmlName)