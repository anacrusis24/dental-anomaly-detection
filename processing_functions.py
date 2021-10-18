import pandas as pd
import numpy as np

def distance_centers(center_anomaly, center_tooth):
    """
    Function finds the distance between the two bounding box centers
    :param center_anomaly: The center of the bounding box surrounding the anomaly in the form [x_center, y_center]
    :param center_tooth: The center of the bounding box surrounding the tooth in the form [x_center, y_center]
    :return: The distance between the two centers
    """
    center_distance = np.sqrt((float(center_anomaly[0]) - float(center_tooth[0]))**2 +
                              (float(center_anomaly[1]) - float(center_tooth[1]))**2)
    return center_distance


def anomaly_matching(anomaly_file, segmentation_file):
    """
    Function matches the anomaly and tooth number
    :param anomaly_file: Bounding box around anomaly as .txt file
    :param center_tooth: Bounding box around tooth as .txt file
    :return: Dataframe with columns anomaly_category, tooth_number, image_name, x_center, y_center, width, height
    """
    # Read in data and remove duplicates
    anomaly_df = pd.read_csv(anomaly_file, header=None, sep=' ', dtype=np.float64)
    tooth_df = pd.read_csv(segmentation_file, header=None, sep=' ', dtype=np.float64)
    tooth_df = tooth_df.drop_duplicates()

    # Empty tooth and anomaly list
    tooth_list = []
    anomaly_list = []

    for i in range(len(anomaly_df)):
        # For each row in anomaly df find anomaly center
        anomaly_center = [anomaly_df.iloc[i][1], anomaly_df.iloc[i][2]]
        anomaly_code = anomaly_df.iloc[i][0]
        temp_distance_list = []
        temp_tooth_list = []

        for j in range(len(tooth_df)):
            # Calculate distance between anomaly center and each tooth bounding box
            tooth_center = [tooth_df.iloc[j][1], tooth_df.iloc[j][2]]
            distance = distance_centers(anomaly_center, tooth_center)

            # Add distance to temporary list
            temp_distance_list.append(distance)
            temp_tooth_list.append(tooth_df.iloc[j][0])

        # Add anomaly and tooth corresponding to smallest distance
        anomaly_list.append(anomaly_code)
        idx = temp_distance_list.index(min(temp_distance_list))
        tooth_list.append(temp_tooth_list[idx])

    # Add healthy teeth
    list_difference = [ele for ele in tooth_df[0] if ele not in tooth_list]
    tooth_list.extend(list_difference)
    anomaly_list.extend([7.0 for i in range(len(list_difference))])

    # Get rest of the corresponding data
    x_center_list = []
    y_center_list = []
    width_list = []
    height_list = []

    for tooth in tooth_list:
        idx = list(tooth_df[0]).index(tooth)
        x_center_list.append(tooth_df[1][idx])
        y_center_list.append(tooth_df[2][idx])
        width_list.append(tooth_df[3][idx])
        height_list.append(tooth_df[4][idx])

    output_df = pd.DataFrame()
    output_df['anomaly_category'] = anomaly_list
    # Add 1 so we don't have the 0th tooth
    output_df['tooth_number'] = [1 + tooth_num for tooth_num in tooth_list]
    output_df['image_name'] = [segmentation_file for i in range(len(tooth_list))]
    output_df['x_center'] = x_center_list
    output_df['y_center'] = y_center_list
    output_df['width'] = width_list
    output_df['height'] = height_list
    output_df = output_df.sort_values(by='tooth_number', axis=0)

    return output_df

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