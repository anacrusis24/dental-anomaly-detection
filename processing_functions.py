import pandas as pd
import numpy as np
from skimage import io

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

    return ([x_start, x_end, y_start, y_end])

def nested_box_area(cartesian_anomaly, cartesian_tooth):
    """
    Function finds the total area that the anomaly bounding box takes within the tooth bounding box
    :param center_anomaly: The cartesian coordinates of the anomaly
    :param center_tooth: The cartesian coordinates of the tooth
    :return: The area the anomaly takes within the bounding box of the tooth
    """
    # Initialize values for bounding box of anomaly
    x_start_new = 0
    x_end_new = 0
    y_start_new = 0
    y_end_new = 0
    area = 0

    # Conditions to makes sure the area is solely within the tooth bounding box
    # First check if one of the corners is within the tooth bounding box
    corner_inside = (((cartesian_anomaly[0] >= cartesian_tooth[0]) and (cartesian_anomaly[2] >= cartesian_tooth[2])
        and (cartesian_anomaly[0] <= cartesian_tooth[1]) and (cartesian_anomaly[2] <= cartesian_tooth[3])) or
       ((cartesian_anomaly[1] <= cartesian_tooth[1]) and (cartesian_anomaly[2] >= cartesian_tooth[2])
        and (cartesian_anomaly[1] >= cartesian_tooth[0]) and (cartesian_anomaly[2] <= cartesian_tooth[3])) or
       ((cartesian_anomaly[0] >= cartesian_tooth[0]) and (cartesian_anomaly[3] <= cartesian_tooth[3])
        and (cartesian_anomaly[0] <= cartesian_tooth[1]) and (cartesian_anomaly[3] >= cartesian_tooth[2])) or
       ((cartesian_anomaly[1] <= cartesian_tooth[1]) and (cartesian_anomaly[3] <= cartesian_tooth[3])
        and (cartesian_anomaly[1] >= cartesian_tooth[0]) and (cartesian_anomaly[3] >= cartesian_tooth[2])))

    if(corner_inside):
        # If one of the corners is in the box find the limits of the anomaly bounding box
        if(cartesian_anomaly[0] >= cartesian_tooth[0]):
            x_start_new = cartesian_anomaly[0]
        else:
            x_start_new = cartesian_tooth[0]

        if(cartesian_anomaly[1] <= cartesian_tooth[1]):
            x_end_new = cartesian_anomaly[1]
        else:
            x_end_new = cartesian_tooth[1]

        if(cartesian_anomaly[2] >= cartesian_tooth[2]):
            y_start_new = cartesian_anomaly[2]
        else:
            y_start_new = cartesian_tooth[2]

        if(cartesian_anomaly[3] <= cartesian_tooth[3]):
            y_end_new = cartesian_anomaly[3]
        else:
            y_end_new = cartesian_tooth[3]

        area = (x_end_new - x_start_new) * (y_end_new - y_start_new)
    else:
        # No anomaly corners are in the tooth bounding box
        area = 0

    return area


def anomaly_matching(anomaly_file, segmentation_file, image_size, normalized=True):
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
        anomaly_yolo = [anomaly_df.iloc[i][1], anomaly_df.iloc[i][2], anomaly_df.iloc[i][3], anomaly_df.iloc[i][4]]
        anomaly_cartesian = yolo_to_cartesian(anomaly_yolo, image_size, normalized=True)
        anomaly_code = anomaly_df.iloc[i][0]
        temp_area_list = []
        temp_tooth_list = []

        for j in range(len(tooth_df)):
            # Calculate distance area of anomaly within tooth bounding box
            tooth_yolo = [tooth_df.iloc[i][1], tooth_df.iloc[i][2], tooth_df.iloc[i][3], tooth_df.iloc[i][4]]
            tooth_cartesian = yolo_to_cartesian(tooth_yolo, image_size, normalized=True)
            area = nested_box_area(anomaly_cartesian, tooth_cartesian)
            print(anomaly_cartesian)
            print(tooth_cartesian)
            print(area)
            # Add area to temporary list
            temp_area_list.append(area)
            temp_tooth_list.append(tooth_df.iloc[j][0])

        print(temp_area_list)
        # Add anomaly and tooth corresponding to max area
        anomaly_list.append(anomaly_code)
        idx = temp_area_list.index(max(temp_area_list))
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
        if normalized:
            image_height, image_width = image_size
            idx = list(tooth_df[0]).index(tooth)
            x_center_list.append(tooth_df[1][idx] * image_width)
            y_center_list.append(tooth_df[2][idx] * image_height)
            width_list.append(tooth_df[3][idx] * image_width)
            height_list.append(tooth_df[4][idx] * image_height)
        else:
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
    output_df = output_df.reset_index(drop=True)

    return output_df

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