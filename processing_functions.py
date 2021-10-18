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

def yolo_to_cartesian(yolo_coordinates, image_size, normalized=True):
    """
    This function transforms Yolo output for coordinates into the Cartesian coordinate system.
    Inputs:
        - yolo_coordinates: list of length 4, following the format [x_center, y_center, width, height], normalized
        - image_size: .shape output for the original image
        - normalized: optional binary parameter, if false then coordinates are not normalized by image width
    Outputs:
        - cartesian_coordinates: outputs Cartesian coordinates based on image inputs
    """
    
    x_center, y_center, width, height = yolo_coordinates
    image_height, image_width = image_size
    
    # Back out the true image sizes
    if normalized:
        x_center = x_center * image_width
        y_center = y_center * image_height
        width = width * image_width
        height = height * image_height
                
    x_start = int(x_center - (width) / 2)
    x_end = int(x_start + width)
    
    y_start = int(y_center - (height) / 2)
    y_end = int(y_start + height)
    
    return([x_start, x_end, y_start, y_end])




def extract_image(filename, yolo_coordinates, tooth_number, output_folder="SegmentedTeethImages", print_names=False):
    """
    This function creates separate image files per tooth out of a larger X-ray file.
    Inputs:
        - filename: string of the name of the file containing the X-ray image in question
        - yolo_coordinates: list of length 4, following the format [x_center, y_center, width, height], normalized
        - tooth_number: number of tooth in question
        - output_folder: optional parameter naming the folder name and path where the output images will be saved
        - print_names: optional parameter to print file names of completed teeth
        """
    
    # Read in file
    image = io.imread(filename)
    
    # Translate yolo coordinates 
    cartesian_coords = yolo_to_cartesian(yolo_coordinates, image.shape)
    
    # Crop image
    image_cropped = image[cartesian_coords[2]:cartesian_coords[3], cartesian_coords[0]:cartesian_coords[1]]
    
    # Create a new folder if needed for the output images
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print("New directory created")
    
    # Write the new images to the folder
    filename_new = output_folder + "\\" + os.path.splitext(filename)[0] + "_" + str(tooth_number) + ".jpg"
    io.imsave(filename_new, image_cropped)
    
    if print_names:
        print(filename_new)