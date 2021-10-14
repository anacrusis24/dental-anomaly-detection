'''
anomaly_matching
function matches the anomaly bounding box to the tooth bounding box
by finding the minimum distance between centers

@input
anomaly_file : bounding box around anomaly as .txt file.
anomaly, x_center, y_center, width, height

segmentation_file : bounding box around tooth as .txt file
tooth number, x_center, y_center, width, height

@output
matching : bounding box and corresponding label as .txt file
anomaly, tooth_number, image_name, x_center, y_center, width, height
'''
def anomaly_matching(anomaly_file, segmentation_file):
    pass
    return matching

'''
extract_image 
function uses tooth bounding box to extract smaller image from main x-ray

@input 
anomaly_matching : .txt from anomaly_matching_function

@output
segmented_tooth : create at least one image per tooth in a folder
'''
def extract_image(anomaly_matching):
    pass
    return segmented_tooth