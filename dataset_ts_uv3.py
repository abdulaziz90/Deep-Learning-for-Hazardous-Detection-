import os
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

def gas_release(cropped_img,xs,ys,ts,u,v):
    max_shift = 2
    Dx = 9
    Dy = 4
    M = 50  # mass
    xmin, xmax = -64, 63  # x-axis interval
    ymin, ymax = -64, 63  # y-axis interval
    tt = np.arange(1,41)  # time
    x, y = np.meshgrid(np.arange(xmin, xmax+1), np.arange(ymin, ymax+1))
    # fig = plt.figure(figsize=(12, 5))

    combined = np.empty((len(tt),128,128,3))
    cloud = np.empty((len(tt),128,128))
    for ind, t in enumerate(tt):
        combined[ind,...] =  cropped_img

        if ind >= ts:
                xx = x - xs - u * (t-ts)
                yy = y - ys - v * (t-ts)
                c = (M / (4 * np.pi * np.sqrt(Dx * Dy))) * np.exp(-0.25 * (xx ** 2 / Dx + yy ** 2 / Dy) / (t-ts))
        else:
                c = np.zeros((cropped_img.shape[0],cropped_img.shape[1]))
        c = np.array(c)
        combined[ind,:,:,1] +=  c
        cloud[ind,...] = c

        # ax = fig.add_subplot(1, 2, 1)
        # ax.imshow(combined[ind,...])

        # plt.show(block=False)
        # plt.title(t)

        # ax = fig.add_subplot(1, 2, 2)
        # ax.imshow(cloud[ind,...])

        # plt.show(block=False)
        # plt.title(t)

        # plt.pause(1)

    return combined, cloud

# Define the parameters for the gas release
xs_min, xs_max = -50, 50
ys_min, ys_max = -50, 50
ts_min, ts_max = 0, 19
u_min, u_max = -2, 2 #[1] #np.array([1, -1])
v_min, v_max = -2, 2 #[1] #np.array([1, -1])

# Define the parameters for the smaller images
num_images_per_original = 10
image_size = 128

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define the paths to the original images and the directories to save the datasets
original_images_dir = '/home/aa587/codes/VAE/Python/DSTL/Code/Kompsat-2-Multispectral-L1'
train_dir = '/home/aa587/codes/VAE/Python/DSTL/Code/dataset_ts_uv3/train'
test_dir = '/home/aa587/codes/VAE/Python/DSTL/Code/dataset_ts_uv3/test2'

# Create the directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Load all image paths
image_paths = [os.path.join(original_images_dir, img) for img in os.listdir(original_images_dir)]

# Split the data into train and test
train_paths, test_paths = train_test_split(image_paths, test_size=40, random_state=42)

# Prepare lists to collect parameters
params_x_train, params_y_train, params_t_train, params_u_train, params_v_train = [], [], [], [], []

# Function to process each image and save in the specified directory
def process_and_save_images(paths, dataset_dir, collect_params=False):
    for i, image_path in enumerate(paths):
        # Load the original image and resize it
        image = plt.imread(image_path)
        image = image[..., :3]

        # Normalize the pixel values to be in the range [0, 1]
        image = image.astype('float32') / np.max(image)

        # Loop over the smaller images to create a time-series cube for each one
        for j in range(num_images_per_original):
            # Randomly select the position to crop the smaller image
            x = np.random.randint(0, image.shape[0] - image_size + 1)
            y = np.random.randint(0, image.shape[1] - image_size + 1)
            cropped_img = image[x:x+image_size, y:y+image_size, :]

            # Randomly select the parameters for the gas release
            xs = np.random.randint(xs_min, xs_max+1)
            ys = np.random.randint(ys_min, ys_max+1)
            ts = np.random.randint(ts_min, ts_max+1)
            u = np.random.randint(u_min, u_max+1)
            v = np.random.randint(v_min, v_max+1)

            # Call the gas_release function to create the time-series cube with the gas release added
            # time_series_cube, cloud = gas_release(cropped_img, xs, ys, ts, u, v)

            ts = 0
            time_series_cube, cloud = gas_release(cropped_img,xs,ys,ts,u,v)

            # Collect parameters if processing training images
            if collect_params:
                params_x_train.append(xs)
                params_y_train.append(ys)
                params_t_train.append(ts)
                params_u_train.append(u)
                params_v_train.append(v)

            # Save the time-series cube and the corresponding parameters as a data point in the dataset
            filename = f'image_{i}_smaller_image_{j}'
            np.save(os.path.join(dataset_dir, filename), time_series_cube)
            np.save(os.path.join(dataset_dir, f'{filename}_cloud'), cloud)
            np.save(os.path.join(dataset_dir, f'{filename}_params'), np.array([xs, ys, ts, u, v]))

# Process and save train images, and collect parameters
# process_and_save_images(train_paths, train_dir, collect_params=True)

# Process and save test images
process_and_save_images(test_paths, test_dir)

# Calculate means and standard deviations for training data
mean_x_train = np.mean(params_x_train)
std_x_train = np.std(params_x_train)

mean_y_train = np.mean(params_y_train)
std_y_train = np.std(params_y_train)

mean_t_train = np.mean(params_t_train)
std_t_train = np.std(params_t_train)

mean_u_train = np.mean(params_u_train)
std_u_train = np.std(params_u_train)

mean_v_train = np.mean(params_v_train)
std_v_train = np.std(params_v_train)

# Save the means and variances
mean2 = np.array([[mean_x_train,mean_y_train,mean_t_train,mean_u_train,mean_v_train]])
var2 = np.array([[std_x_train,std_y_train,std_t_train,std_u_train,std_v_train]])**2

np.save(os.path.join(train_dir, 'mean2'), mean2)
np.save(os.path.join(train_dir, 'variance2'), var2)



print('Done!')

