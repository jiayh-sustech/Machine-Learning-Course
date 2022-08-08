import cv2
from train import processFiles, trainSVM
from detector import Detector

# Replace these with the directories containing your
# positive and negative sample images, respectively.
pos_dir = "../samples/vehicles"
neg_dir = "../samples/non-vehicles"

# Replace this with the path to your test video file.
video_file = "../videos/project_video.mp4"

def experiment1():
    """
    Train a classifier and run it on a video using default settings
    without saving any files to disk.
    """

    # Extract HOG features from images in the sample directories and return
    # results and parameters in a dict.
    # feature_data = processFiles(pos_dir, neg_dir, recurse=True,
    #     hog_features=True)
    feature_data = processFiles(pos_dir, neg_dir, recurse=True, output_file=False,
        output_filename=None, color_space="rgb", channels=[0, 1, 2],
        hog_features=True, hist_features=True, spatial_features=True,
        hog_lib="cv", size=(64,64), hog_bins=9, pix_per_cell=(8,8),
        cells_per_block=(2,2), block_stride=None, block_norm="L1",
        transform_sqrt=True, signed_gradient=False, hist_bins=16,
        spatial_size=(16,16))

 
    # Train SVM and return the classifier and parameters in a dict.
    # This function takes the dict from processFiles() as an input arg.
    # classifier_data = trainSVM(feature_data=feature_data)
    classifier_data = trainSVM(filepath=None, feature_data=feature_data, C=2333,
    loss="squared_hinge", penalty="l2", dual=False, fit_intercept=False,
    output_file=False, output_filename=None)

	##TODO: If you have trained your classifier and prepare to detect the video, 
	##      uncomment the code below.
	
    # Instantiate a Detector object and load the dict from trainSVM().
    #detector = Detector().loadClassifier(classifier_data=classifier_data)
    detector = Detector(init_size=(64,64), x_overlap=0.5, y_step=0.01,
            x_range=(0, 1), y_range=(0.5, 0.9), scale=1.5).loadClassifier(classifier_data=classifier_data)

    # Open a VideoCapture object for the video file.
    cap = cv2.VideoCapture(video_file)
    
    # Start the detector by supplying it with the VideoCapture object.
    # At this point, the video will be displayed, with bounding boxes
    # drawn around detected objects per the method detailed in README.md.
    #detector.detectVideo(video_capture=cap)
    detector.detectVideo(video_capture=cap, num_frames=9, threshold=120,
            min_bbox=None, show_video=True, draw_heatmap=True,
            draw_heatmap_size=0.2, write=True, write_fps=24)

#def experiment2
#    ...

if __name__ == "__main__":
    experiment1()
    #experiment2() may you need to try other parameters
	#experiment3 ...