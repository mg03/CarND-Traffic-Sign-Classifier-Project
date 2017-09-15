**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/training_set_histogram.jpg
[image2]: ./output_images/grayscale.jpg
[image3]: ./output_images/rotated.jpg
[image4]: ./output_images/web_signs.jpg
[image5]: ./output_images/softmax1.jpg
[image6]: ./output_images/softmax2.jpg


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. The submission includes the project code and here is a link to my [project code](https://github.com/mg03/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. 

* Number of training examples = 34799
* Number of testing examples = 12630
* Number of validation examples = 4410
* Image data shape = (32, 32, 3)
* Number of classes = 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of the training set

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because for traffic sign classification, shapes proved sufficient since no two traffic signs are of the same shape, since 
that would even confuse human drivers.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

I tried using normalized image but that didnt seem to help the training, so didnt use it

I decided to generate additional data because as seen in the histogram the distribution of classes
is uneven leading to bias in training.

To add more data to the the data set, I used the following techniques:
* Add rotated image 

Here is an example of an original image and an augmented image:

![alt text][image3] 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.)

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5   | 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					    |												|
| Max pooling	      | 2x2 stride,  outputs 14x14x6			|
| Convolution 5x5	  | 1x1 stride, VALID padding, outputs 10x10x16 	|
| RELU					    |												|
| Max pooling	      | 2x2 stride,  outputs 5x5x16			|
| Flatten			      | Input = (5, 5, 16) Ouput = 400
| Dropout           | Keep Prob = 0.5
| Fully connected	1	| Input = 400 Output = 120       									|
| RELU
| Dropout           | Keep Prob = 0.5
| Fully connected	2	| Input = 120 Output = 84      									|
| RELU
| Fully connected	3	| Input = 84 Output = 43   



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a Lenet model. I used one hot encoding of the labels.
I used Adam optimizer with learning rate of 0.0009 .  In the model used mu = 0 and sigma = 0.1
Epoch = 100 and Batch size = 128


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I initially faced issue with overfitting where training loss was low but validation loss was high.
Decided to use Dropout. I tested with many keep probabilities - 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5. Keep probability = 0.5 works the best

My final model results were:
* Train Loss per epoch= 0.271
* Validation Accuracy = 0.948 (94.8%)
* Validation Loss = 0.239
* Test Accuracy = 0.939 ( 93.9%)
* Test Loss = 0.221


If a well known architecture was chosen:
* What architecture was chosen?
  LeNet model was choosen

* Why did you believe it would be relevant to the traffic sign application?
  Honestly, I used LeNet since there was a course Lab using LeNet and I was familiar with it

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
  As you can see that the training loss and validation loss keep decreasing over the epochs (no overfitting of data) and the validation accuracy is 94.8% and the test accuracy is 93.9%.
  So we have a pretty decent model.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are traffic signs that I found on the web:

![alt text][image4]

The images choosen in the unseen dataset consists of 9 traffic sign images collected from the internet. When selecting these examples following two criteria were used:

- 1 Test the ability of the model to recognized already familiar traffic signs. 
- 2 Two of the examples were selected to measure the generalization ability of the network. These 2 examples were not represented in the training set. So obviously, the model may not correctly classify these examples, but guesses should be sensible.



####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set 

The provided images were 

outside_labels_desc = ['16-Veh over 3.5 mtons prohibited', '29 - Bicycles crossing',
                       'Unknown -   Elderly Crossing', '12 - Priority road', '25 - Road work', '14 - Stop', '33 - Turn right ahead', '13 - Yield', 
                       'Unkonwn to me']

Labels 200 and 400 are used for images with no labels in the training set

Here are the results of the prediction:

Inferred classes: [ 16.  29.  28.  12.  25.  14.  33.  13.  23.]

Prediction Accuracy = 0.778

The model was able to correctly guess 7 of the 9 traffic signs, and gives an accuracy of 77.8%.

But 7 out of the 9 are signs which were included in the training model , so actually we get a 7 on 7 for known labels/signs i.e. 100% prediction. This compares favorably to the accuracy on the test set of 93.9%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

- 1  Actual Label: 16 TopGuess:-Label: 16 Prob%: 100.0 2ndGuess:-Label: 9 Prob%: 0.0 3rdGuess:-Label: 10 Prob%: 0.0
- 2  Actual Label: 29 TopGuess:-Label: 29 Prob%: 99.0 2ndGuess:-Label: 28 Prob%: 1.33 3rdGuess:-Label: 23 Prob%: 0.0
- 3  Actual Label: 200 TopGuess:-Label: 28 Prob%: 100.0 2ndGuess:-Label: 29 Prob%: 0.2 3rdGuess:-Label: 30 Prob%: 0.07
- 4  Actual Label: 12 TopGuess:-Label: 12 Prob%: 100.0 2ndGuess:-Label: 25 Prob%: 0.0 3rdGuess:-Label: 40 Prob%: 0.0
- 5  Actual Label: 25 TopGuess:-Label: 25 Prob%: 100.0 2ndGuess:-Label: 22 Prob%: 0.0 3rdGuess:-Label: 11 Prob%: 0.0
- 6  Actual Label: 14 TopGuess:-Label: 14 Prob%: 100.0 2ndGuess:-Label: 17 Prob%: 0.0 3rdGuess:-Label: 38 Prob%: 0.0
- 7  Actual Label: 33 TopGuess:-Label: 33 Prob%: 100.0 2ndGuess:-Label: 17 Prob%: 0.0 3rdGuess:-Label: 35 Prob%: 0.0
- 8  Actual Label: 13 TopGuess:-Label: 13 Prob%: 100.0 2ndGuess:-Label: 12 Prob%: 0.0 3rdGuess:-Label: 15 Prob%: 0.0
- 9  Actual Label: 400 TopGuess:-Label: 23 Prob%: 26.0 2ndGuess:-Label: 12 Prob%: 19.75 3rdGuess:-Label: 41 Prob%: 12.18


The first unkown image is a sign For Elderly Crossing, the model classfies it as 28 (Children Crossing) with 100% probabity. So atleast it got the part that the unknown sign is a Crossing type sign

The second unknown image is something that I dont know. Hence I was interested in seeing what the
model predicts. The model predicts it to be 23 (Slipper Road) with probability 26% , 12 (Priority Road) with 20% probability, 41(End of no passing) with 13% probability. I believe if the image would be without its square/rectangle white board background, the model would be able to atleast tell us that the sign is of Regulatory signs type .
German signs have the following broad types: Regulatory signs, Speed limit signs, Right-of-way signs, Railway crossing signs, Warning signs, Supplemental signs.  
[Source](http://www.gettingaroundgermany.info/zeichen.shtml

Softmax Probabilities are as follows:

![alt text][image5]
![alt text][image6]

I have not tried to map labels to the Names using signnames.csv. I didnt think i will add any
value to the model training, validation or testing.