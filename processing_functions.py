import pandas as pd
import numpy as np
from skimage import io as io
from skimage.transform import rotate
from skimage.util import random_noise
from skimage.filters import gaussian
import os
from tqdm import tqdm
import torch
from imblearn.over_sampling import SMOTE


def distance_centers(center_anomaly, center_tooth):
    """
    Function finds the distance between the two bounding box centers
    :param center_anomaly: The center of the bounding box surrounding the anomaly in the form [x_center, y_center]
    :param center_tooth: The center of the bounding box surrounding the tooth in the form [x_center, y_center]
    :return: The distance between the two centers
    """
    center_distance = np.sqrt((float(center_anomaly[0]) - float(center_tooth[0])) ** 2 +
                              (float(center_anomaly[1]) - float(center_tooth[1])) ** 2)
    return center_distance


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
    
    tooth_df[0] = pd.Series(range(1, len(tooth_df) + 1)) # Comment out if teeth number given

    # Un-normalize the centers, widths, and heights
    if normalized:
        image_height, image_width = image_size
        anomaly_df[1] = anomaly_df[1] * image_width
        anomaly_df[2] = anomaly_df[2] * image_height
        anomaly_df[3] = anomaly_df[3] * image_width
        anomaly_df[4] = anomaly_df[4] * image_height

        tooth_df[1] = tooth_df[1] * image_width
        tooth_df[2] = tooth_df[2] * image_height
        tooth_df[3] = tooth_df[3] * image_width
        tooth_df[4] = tooth_df[4] * image_height

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
#     output_df['tooth_number'] = [1 + tooth_num for tooth_num in tooth_list]
    output_df['tooth_number'] = [tooth_num for tooth_num in tooth_list]
    output_df['image_name'] = [os.path.basename(segmentation_file) for i in range(len(tooth_list))]
    output_df['x_center'] = x_center_list
    output_df['y_center'] = y_center_list
    output_df['width'] = width_list
    output_df['height'] = height_list
    output_df = output_df.sort_values(by='tooth_number', axis=0)
    output_df = output_df.reset_index(drop=True)

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

    return ([x_start, x_end, y_start, y_end])


def extract_image(filename, yolo_coordinates, tooth_number, output_folder="SegmentedTeethImages/", print_names=False):
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
    cartesian_coords = yolo_to_cartesian(yolo_coordinates, image.shape, normalized=False)

    # Crop image
    image_cropped = image[cartesian_coords[2]:cartesian_coords[3], cartesian_coords[0]:cartesian_coords[1]]

    # Create a new folder if needed for the output images
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print("New directory created")

    # Write the new images to the folder
    basename = os.path.basename(filename)
    filename_new = output_folder + os.path.splitext(basename)[0] + "_" + str(tooth_number) + ".jpg"
    io.imsave(filename_new, image_cropped)

    if print_names:
        print(filename_new)

    return filename_new


def remove_duplicates(df):
    df['IS_DUPLICATED'] = df.duplicated(subset=['tooth_number'])
    dup_index = df[df['IS_DUPLICATED'] == True].index
    df.iloc[dup_index - 1, 0] = 8
    df.drop(dup_index, axis=0, inplace=True)
    df.drop('IS_DUPLICATED', axis=1, inplace=True)
    return df

def multi_labeling(df):
    df = df.copy()  # Copy not to delete original dataset
    df['anomaly_category'] = df['anomaly_category'].astype(object)  # Change dtype from int to list
    df['anomaly_category_old'] = df['anomaly_category']
    
    for index, row in df.iterrows():  # Updating anomaly category to list of anomalies
        filler_list = [0] * 8
        all_anomalies = df[df['tooth_number'] == row['tooth_number']]['anomaly_category_old'].to_list()
        for anomaly in all_anomalies:
            filler_list[anomaly] = 1
        df.at[index, 'anomaly_category'] = filler_list
        
    deduplicated_df = df[~df.duplicated(subset=['tooth_number'])]
    deduplicated_df = deduplicated_df.drop(['anomaly_category_old'], axis = 1)
    return(deduplicated_df)

def image_rotation(filename, deg, output_folder="SegmentedTeethImages/", print_names=False):
    """
    This function takes an image and rotates it and then saves the new image.  The number of rotations
    and new images made is 360/deg or 360/deg - 1 if 360//deg==0
    Inputs:
        filename: the image to rotate
        deg: the number of degrees to rotate the image by
        output_folder: optional parameter naming the folder name and path where the output images will be saved
        print_names: optional parameter to print file names of completed teeth
    """

    # Read in file
    image = io.imread(filename)

    # Create a new folder if needed for the output images
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print("New directory created")

    # Create the rotated images
    basename = os.path.basename(filename)
    cur_deg = deg
    file_names = []

    while (cur_deg < 360):
        # Make the rotated image
        rotated_image = rotate(image, angle=cur_deg)

        # Write the new images to the folder
        filename_new = output_folder + os.path.splitext(basename)[0] + "_" + "rotated" + str(cur_deg) + ".jpg"
        io.imsave(filename_new, rotated_image)
        file_names.append(filename_new)

        # Print names of files
        if print_names:
            print(filename_new)

        # Increase the number of degrees
        cur_deg += deg

    return file_names


def image_flip(filename, output_folder="SegmentedTeethImages/", print_names=False):
    """
    This function flips the image left-right and up-down and saves the images
    Inputs:
        filename: the image to flip
        output_folder: optional parameter naming the folder name and path where the output images will be saved
        print_names: optional parameter to print file names of completed teeth
    """
    # Read in file
    image = io.imread(filename)

    # Create a new folder if needed for the output images
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print("New directory created")

    # Flip image left and right
    flipLR = np.fliplr(image)

    # Flip image up and down
    flipUD = np.flipud(image)

    # Write the new images to the folder
    basename = os.path.basename(filename)
    filename_new_LR = output_folder + os.path.splitext(basename)[0] + "_" + "LR" + ".jpg"
    filename_new_UD = output_folder + os.path.splitext(basename)[0] + "_" + "UD" + ".jpg"
    io.imsave(filename_new_LR, flipLR)
    io.imsave(filename_new_UD, flipUD)
    file_names = [filename_new_LR, filename_new_UD]

    # Print names of files
    if print_names:
        print(filename_new_LR, filename_new_UD)

    return file_names


def image_noise(filename, sigma=.1, output_folder="SegmentedTeethImages/", print_names=False):
    """
    This function adds random noise to the image and saves it
    Inputs:
        filename: the image to add noise too
        output_folder: optional parameter naming the folder name and path where the output images will be saved
        print_names: optional parameter to print file names of completed teeth
    """
    # Read in file
    image = io.imread(filename)

    # Create a new folder if needed for the output images
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print("New directory created")

    # Add random noise
    noise_rand = random_noise(image, var=sigma ** 2)

    # Write the new images to the folder
    basename = os.path.basename(filename)
    filename_new = output_folder + os.path.splitext(basename)[0] + "_" + "noise" + ".jpg"
    io.imsave(filename_new, noise_rand)

    # Print names of files
    if print_names:
        print(filename_new)

    return filename_new


def image_gauss_blur(filename, sigma=1, output_folder="SegmentedTeethImages/", print_names=False):
    """
    This function adds gaussian blur to the image and saves it
    Inputs:
        filename: the image to add blur too
        output_folder: optional parameter naming the folder name and path where the output images will be saved
        print_names: optional parameter to print file names of completed teeth
    """
    # Read in file
    image = io.imread(filename)

    # Create a new folder if needed for the output images
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print("New directory created")

    # Add random noise
    blurred_image = gaussian(image, sigma=sigma, multichannel=True)

    # Write the new images to the folder
    basename = os.path.basename(filename)
    filename_new = output_folder + os.path.splitext(basename)[0] + "_" + "blur" + ".jpg"
    io.imsave(filename_new, blurred_image)

    # Print names of files
    if print_names:
        print(filename_new)

    return filename_new


def make_data(xray_path, anomaly_path, segmentation_path, output_path, which_anomaly=[0, 1, 2, 3, 4, 5, 6, 7, 8],
              add_rotation=False, rotation_deg=20, add_flip=False, add_noise=False, sigma_noise=0.1,
              add_blur=False, sigma_blur=1):
    """
    Function makes the segmented teeth with whichever data augmentation and then creates a csv that lists
    the file paths and the corresponding label.
    Inputs:
        x_ray_path: path to the x-ray jpegs
        anomaly_path: path to the anomaly bounding box txts
        segmentation_path: path to the teeth bounding box txts
        output_path: path to output the segmented teeth
        which_anomaly: a list of anomalies that we want to have augmented
        add_rotation: if true augment image with rotation
        rotatation_deg: how much the rotation should turn
        add_flip: if true augment image with flips
        add_noise: if true augment image with noise
        sigma_noise: noise parameter
        add_blur: if true augment image with blur
        sigma_blur: blur parameter
    """
    file_paths = []
    anomaly_codes = []
    is_rotated = []
    is_flipped = []
    is_noise = []
    is_blur = []

    xray_filenames = os.listdir(xray_path)
    anomaly_filenames = os.listdir(anomaly_path)
    segmentation_filenames = os.listdir(segmentation_path)
    
    xray_file_stripped = [x[:-4] for x in xray_filenames]
    anomaly_file_stripped = [x[:-4] for x in anomaly_filenames]
    segmentation_file_stripped = [x[:-4] for x in segmentation_filenames]
    final_list = set(xray_file_stripped).intersection(anomaly_file_stripped)
    final_list = set(final_list).intersection(segmentation_file_stripped)
    final_list = list(final_list)
    
    extension_list = []
    
    for file in final_list:
        if(file + '.jpg' in xray_filenames):
            extension_list.append(1)
        else:
            extension_list.append(0)

    for i in range(len(final_list)):
        # Create dataframe with tooth number
        if(extension_list[i] == 1):
            anomalies_df = anomaly_matching(anomaly_path + final_list[i] + '.txt',
                                            segmentation_path + final_list[i] + '.txt',
                                            io.imread(xray_path + final_list[i] + '.jpg').shape,
                                            normalized=True)
            x_ray = xray_path + final_list[i] + '.jpg'
        else:
            anomalies_df = anomaly_matching(anomaly_path + final_list[i] + '.txt',
                                            segmentation_path + final_list[i] + '.txt',
                                            io.imread(xray_path + final_list[i] + '.png').shape,
                                            normalized=True)
            x_ray = xray_path + final_list[i] + '.png'
        
        anomalies_df = remove_duplicates(anomalies_df)

        for index, row in anomalies_df.iterrows():
            yolo_coord = row[['x_center', 'y_center', 'width', 'height']].to_list()
            tooth_file = extract_image(x_ray,
                                       yolo_coord,
                                       int(row['tooth_number']),
                                       output_folder=output_path,
                                       print_names=False)
            
            file_paths.append(tooth_file)
            anomaly_codes.append(row['anomaly_category'])
            is_rotated.append(0)
            is_flipped.append(0)
            is_noise.append(0)
            is_blur.append(0)

            tooth_file_path = tooth_file

            if row['anomaly_category'] in which_anomaly:
                if add_rotation:
                    rotated_images = image_rotation(tooth_file_path, deg=rotation_deg,
                                                    output_folder=output_path, print_names=False)
                    rotated_labels = [row['anomaly_category']] * len(rotated_images)
                    rotated_bool = [1] * len(rotated_images)
                    flipped_bool = [0] * len(rotated_images)
                    noise_bool = [0] * len(rotated_images)
                    blur_bool = [0] * len(rotated_images)
                   
                    file_paths.extend(rotated_images)
                    anomaly_codes.extend(rotated_labels)
                    is_rotated.extend(rotated_bool)
                    is_flipped.extend(flipped_bool)
                    is_noise.extend(noise_bool)
                    is_blur.extend(blur_bool)

                if add_flip:
                    flipped_images = image_flip(tooth_file_path, output_folder=output_path, print_names=False)
                    flipped_labels = [row['anomaly_category']] * len(flipped_images)
                    rotated_bool = [0] * len(flipped_images)
                    flipped_bool = [1] * len(flipped_images)
                    noise_bool = [0] * len(flipped_images)
                    blur_bool = [0] * len(flipped_images)
                   
                    file_paths.extend(flipped_images)
                    anomaly_codes.extend(flipped_labels)
                    is_rotated.extend(rotated_bool)
                    is_flipped.extend(flipped_bool)
                    is_noise.extend(noise_bool)
                    is_blur.extend(blur_bool)

                if add_noise:
                    noise_image = image_noise(tooth_file_path, sigma=sigma_noise,
                                              output_folder=output_path, print_names=False)
                    noise_label = row['anomaly_category']
                    
                    file_paths.append(noise_image)
                    anomaly_codes.append(noise_label)
                    is_rotated.append(0)
                    is_flipped.append(0)
                    is_noise.append(1)
                    is_blur.append(0)
                    

                if add_blur:
                    blur_image = image_gauss_blur(tooth_file_path, sigma=sigma_blur,
                                                  output_folder=output_path, print_names=False)
                    blur_label = row['anomaly_category']
                    
                    file_paths.append(blur_image)
                    anomaly_codes.append(blur_label)
                    is_rotated.append(0)
                    is_flipped.append(0)
                    is_noise.append(0)
                    is_blur.append(1)

    main_df = pd.DataFrame()
    main_df['file_path'] = file_paths
    main_df['anomaly_code'] = anomaly_codes
    main_df['is_rotated'] = is_rotated
    main_df['is_flipped'] = is_flipped
    main_df['is_noise'] = is_noise
    main_df['is_blur'] = is_blur
    main_df.to_csv(f'C:/Documents/Dental_Detection/data_csv/segmented_data.csv')
    print('Made segmented_data.csv')
    

def SMOTE_Balance(train_dataloader):
    sm = SMOTE(random_state=42, k_neighbors=1)
    X_train_img = pd.DataFrame(columns=range(0,128*128))
    y_train_img = list()

    for (x,y) in tqdm(train_dataloader):
        y = y.tolist()
        for i in range(len(x)):
            img = x[i]
            c = y[i]
            img = np.array(img)
            img = np.resize(img, (128*128)).tolist()
            length = len(X_train_img)
            X_train_img.loc[length] = img
            y_train_img.append(c)
    print("Transformed images for SMOTE")       
    
    smote_X_train, smote_ny_train = sm.fit_resample(X_train_img,y_train_img)
    print("Performed SMOTE")
    
    smote_X_train_rs = []
    for i in range(0, len(smote_X_train)):
        smote_X_train_rs.append(np.resize(np.array(smote_X_train.loc[i]), (3,244,244)))
    smote_X_train_rs = np.asarray(smote_X_train_rs)
    smote_X_train_rs = torch.from_numpy(smote_X_train_rs)
    smote_ny_train = torch.FloatTensor(smote_ny_train)
    print("Transformed images for Model")
    
    return smote_X_train_rs, smote_ny_train

