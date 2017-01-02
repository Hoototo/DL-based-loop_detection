#from sklearn.utils import shuffle

from pcanet import PCANet
import tensorflow as tf
import numpy as np

# convert files containing multple images in png format to
# gray_scale images and pack them in
# image batch[batch_num,height,width]


def convert_to_array_queue(files, files_num, height, width, channels):
    filenames = tf.train.match_filenames_once(files)
    filename_queue = tf.train.string_input_producer(filenames)

    # step 3: read, decode and resize images
    reader = tf.WholeFileReader()
    filename, content = reader.read(filename_queue)
    image = tf.image.decode_png(content, channels=channels)
    resized_image = tf.image.resize_images(image,[height, width])
    gray_images = tf.image.rgb_to_grayscale(resized_image)

    # step 4: Batching
    image_batch = tf.train.batch([gray_images], batch_size=1)
    batch_size = 1
    batch_num = int(files_num / batch_size)

    with tf.Session() as sess_1:
        tf.initialize_all_variables().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess_1, coord=coord)
        image_total = []
        for i in range(batch_num):
            image_tensor = image_batch.eval()  # (1,heightheight,width,1)
            image_array = np.asarray(image_tensor[0])  # (height,widthwidth,1)
            image_total.append(image_array)
        # convert list to array
        image_total = np.array(image_total)  # (batch_num,height,width,1))
        image_total=np.reshape(image_total,(batch_num,height,width))
        num_examples = image_total.shape[0]

        coord.request_stop()
        coord.join(threads)
    return image_total,num_examples


height = 60
width = 80
channels = 3
files = "/home/ubuntu/PCANet-python3/rgb_png/*.png"
files_num = 96

images,num_examples = convert_to_array_queue(
    files=files, files_num=files_num, height=height,
    width=width, channels=channels)
#images= shuffle(images, random_state=0)
print(images)
print(num_examples)
print(images.shape)

pcanet = PCANet(
    image_shape=(60,80),
    filter_shape_l1=2, step_shape_l1=1, n_l1_output=4,
    filter_shape_l2=2, step_shape_l2=1, n_l2_output=4,
    block_shape=2
)

print("has excuted PCANET function")

pcanet.validate_structure()

print("has excuted pcanet.validate_structure")

pcanet.fit(images)

print("has excuted pacnet.fit")

X_train = pcanet.transform(images)

print("has excuted pcanet.transform")

print(X_train)
print(X_train.shape)

# Fit any models you like
# add confusion_matrix
X_test = pcanet.transform(images)

def normalize(x): return x / (x)

def build_confusion_matrix(representations):
    n_frames = len(representations)
    confusion_matrix = np.zeros((n_frames, n_frames))

    for i in range(n_frames):
        for j in range(n_frames):   
            confusion_matrix[i][j] = np.dot(representations[i],representations[j]) / (np.linalg.norm(representations[i]) * np.linalg.norm(representations[j]))
            
    return confusion_matrix

confusion_matrix = build_confusion_matrix(X_test)
print (len(representations)) 
print (confusion_matrix)

#load the ground truth
GROUND_TRUTH_PATH = os.path.expanduser(
    '/home/lenovo/TF/test/NewCollegeGroundTruth.mat')

gt_data = sio.loadmat(GROUND_TRUTH_PATH)['truth'][::2, ::2]
gt_data = gt_data.T + gt_data + np.eye(1073) # the number of len(representations)
print(gt_data)

# Set up plotting

default_heatmap_kwargs = dict(
    xticklabels=False,
    yticklabels=False,
    square=True,
    cbar=True,)

fig, ax1 = plt.subplots(ncols=1)
fig, ax2 = plt.subplots(ncols=1)

# Plot ground truth
sns.heatmap(gt_data,
    ax=ax1,
    **default_heatmap_kwargs)
ax1.set_title('Ground truth')

# Only look at the lower triangle
# confusion_matrix = np.tril(confusion_matrix, 0)

sns.heatmap(confusion_matrix,
           ax=ax2,
           **default_heatmap_kwargs)
# ax2.set_title('CNN')
ax2.set_title('PCANet')

# precision recall curve
prec_recall_curve = []

for thresh in np.arange(0, 1, 0.02):
    # precision: fraction of retrieved instances that are relevant
    # recall: fraction of relevant instances that are retrieved
    true_positives = (confusion_matrix > thresh) & (gt_data == 1)
    all_positives = (confusion_matrix > thresh)

    try:
        precision = float(np.sum(true_positives)) / np.sum(all_positives)
        recall = float(np.sum(true_positives)) / np.sum(gt_data == 1)

        prec_recall_curve.append([thresh, precision, recall])
    except:
        break

prec_recall_curve = np.array(prec_recall_curve)

plt.plot(prec_recall_curve[:, 1], prec_recall_curve[:, 2])

for thresh, prec, rec in prec_recall_curve[ 30: : 5]:
    plt.annotate(
        str(thresh),
        xy=(prec, rec),
        xytext=(8, 8),
        textcoords='offset points')

plt.xlabel('Precision', fontsize=14)
plt.ylabel('Recall', fontsize=14)
