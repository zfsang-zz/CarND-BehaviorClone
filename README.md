
### CarND Project 3 Behavior Learning

Obtain and learn the driving behaviors from the Udacity driving simulator that uses real game physics to create a close approximation to real driving.


## Data

#### Data Source
Collecting data with keyboards control was quite difficult sharp and limited control over the wheel and steering angles. After experimenting with self-collected driving images for a while, I decided to use [sample data for track 1](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) released by Udacity by Dec 12,2016.

#### Data Augmentation

As Udacity Mentor suggested, data augmentation is the key for solving our problem. We can  increase the volume of training data and make the model learn complex road conditions to keep the car on the track.  For this project, I used the following data augmentation and preprocessing.

- Randomly choose images from center, left (angle offset +0.25), right cameras (angle offset -0.25 ) from the Udacity dataset
- Add randomness in the image brightness by using random multiplier 0.25 + np.random.uniform() for V channel in HSV
- crop and resize the images, only keep middle part
- Add random image translation and random angle translation
- Add random shadow mask
- Add random image flip

#### Random image from left, center, right cameras

The straight way is to use only center camera images for training. I did it in my first attempt. Most of the time the car is moving smoothly due to good prediction of zero steering angles but if there is a sharp turn, the car can't make it in time because the model can't predict large values of angles.

Using left and right camera images allows the model to predict values during the sharp turns. I simply add .25 offset to angle from left image sample and subtract .25 offset from angle from right image sample.

#### Random image and angle translation

Classic technic for images but in our case we need to implement it for angles as well.


#### Random brightness

The brightness of images from track 1 and track 2 is very different so adding multiplier .25 + np.random.uniform() for V channel (from HSV) will help our model to work correctly on both tracks.

## Model

#### Design and Consideration

Transfer learning requires a lot of transformation and data preprocessing, which may not be efficient for the small project.  Instead of using transfer learning, I use the multiple layer CNN to train my own model.


I trained the following models:
- Modified AlexNet
- Modified VGG16
- Navid Net
- Modified Comma.ai model with normalization and dropout layers

The modifications includes (may not all applied to the same model):
- Change the image input size
- Change the size of dense layers and convolution layers
- Add normalization layers
- Add random dropout layers
- Add color channel extract layers


One version of modified Comma.ai model achieved the best performance over all modi. It successfully finished first track after just 3 epochs of training. I noticed that 3x3 convolutions are not great choice for solving our kind of problem and we need to prevent overfitting for obtaining good results, especially on both tracks.



The detailed network structure is listed below.  Basically, I used 3 convolutional layers and 2 dense fully connected layers with random dropout and normalization layers.  First I used the image of the default size 320x160 for direct input, and found it was very slow for the training on my local CPU.  It took more than 40 minutes to finish 1 epoch on CPU, and I do not have a good GPU to play with.  So I decided to resize the images to 64X64 and use a much smaller network structure.  

Layer (type)                     Output Shape          Param #     Connected to

lambda_1 (Lambda)                (None, 64, 64, 3)     0           lambda_input_1[0][0]
____________________________________________________________________________________________________
Conv1 (Convolution2D)            (None, 16, 16, 16)    3088        lambda_1[0][0]
____________________________________________________________________________________________________
Conv2 (Convolution2D)            (None, 8, 8, 32)      12832       Conv1[0][0]
____________________________________________________________________________________________________
Conv3 (Convolution2D)            (None, 4, 4, 64)      51264       Conv2[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1024)          0           Conv3[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 1024)          0           flatten_1[0][0]
____________________________________________________________________________________________________
Dense1 (Dense)                   (None, 1024)          1049600     activation_1[0][0]
____________________________________________________________________________________________________
Dropout (Dropout)                (None, 1024)          0           Dense1[0][0]
____________________________________________________________________________________________________
Output (Dense)                   (None, 1)             1025        Dropout[0][0]

Total params: 1117809

## Training

#### Batch Generator
I take random image from random camera, randomly apply all previously described augmentations and generate batch for training or validation.

#### Hyperparameters tuning
I trained the model using batch generator, Adam optimizer, mean squared error loss function, batch size of 128 samples and 20 epochs of training.

#### Predictions
It was a good idea to plot real and predicted values after each training for understanding model behavior in particular cases. For example the following trained model will have some problems with straight parts of the road because it doesn't predict zero values with high accuracy.

## Performance

The car was driving track 1 for an infinite amount of time and track 2 till the end of the road with throttle equal to 0.1.

The car was jittering around for some of the time using a high throttle (=0.5),especially when it was making sharp turns.  To make the car drive and turn smoothly on both tracks, I arbitrarily decrease the speed of the car. Dynamic assigned throttle values are preferable and need to be added in the future.  

Another thing to improve is to utilize AWS GPU-instance to train and a much larger network to reflect the nature of behavior clone more precisely.

## References
- [Comma.ai steering model](https://github.com/commaai/research)
- [NVIDIA End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf)
- [An augmentation based deep neural network approach to learn human driving behavior](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.u48xp0cj4)
- https://github.com/dmitrylosev/clonedrivingbehavior
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)  
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980v8)  
