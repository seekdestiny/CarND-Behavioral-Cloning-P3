# **Behavioral Cloning** 

## Abstract

**Behavioral Cloning Project**

In this project, I followd the approach presented by [Nvidia](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). This is introduced in even more powerful network project video. I skipped trying LeNet because it is already used in traffic sign classifier project and more suitable for recognition work.

Data collection and testing are performed in a simulator provided by Udacity. Actually, I used Udacity sample
driving data for final training. Because my self collected data does not work well.

Since the sample data only consists first track, the trained model manages to successfully drive the car
indefinitely in track1. In track2, it will quickly crash into one guardrail. More time is needed on fine
tuned data collection on track2.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1a]: ./images/arch.png "Model Architecture"
[image1b]: ./images/model_summary.png "Model Summary"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](https://github.com/seekdestiny/CarND-Behavioral-Cloning-P3/blob/master/model.py) containing the script to create and train the model
* [preprocess_input.py](https://github.com/seekdestiny/CarND-Behavioral-Cloning-P3/blob/master/preprocess_input.py) is used to put all image preprocessed trials together
* [drive.py](https://github.com/seekdestiny/CarND-Behavioral-Cloning-P3/blob/master/drive.py) for driving the car in autonomous mode
* [model.h5](https://github.com/seekdestiny/CarND-Behavioral-Cloning-P3/blob/master/models/model.h5) containing a trained convolution neural network 
* [writeup_report.md](https://github.com/seekdestiny/CarND-Behavioral-Cloning-P3/blob/master/README.md) summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The basic pipeline is implemented in build_model function. It parsed the csv log file, define model and train
it. 

```python
def build_model(log_file_path, n_epochs, save_dir):
    """ Builds and trains the network given the input data in train_dir """

    # Get training and validation data
    X, y = get_training_data(log_file_path)

    # Build and train the network
    model = define_model()
    train_model(model, save_dir, n_epochs, X, y)
```

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I have implemented the convolutional neural network proposed by
[Nvidia](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).
The following picture summarizes the model:

![alt text][image1a]

The architecture is a combination of Convolutional layers
followed by Fully-Connected layers, since the input data is a raw RGB image.
This time, the architecture is applied to a **regression problem** (predicting
steering angle) instead of classification, so no activation function
or softmax must be applied at the last layer, which will have only one neuron.

I also considered transfer learning tech introduced in lecture. But
after some quick trial on AlexNet, I found it is more suitable for recognition job
instead of regression problem. And retraining all weights from the scatch is
computation resource limited which makes it impossible. 

The implemented network consists of the following layers:

- **Input**. Image of size (66, 200, 3). I crop original (160, 320, 3) size to this.
- **Normalization** to the range [-0.5, 0.5]. This is performed using a _Lambda_ in Keras.
- **Convolutional 1**. 24 filters of size 5x5x3 (since the input has 3 channels).
The filter is applied with strides of (2, 2) instead of using MaxPooling.
This can be done because the input image is relatively high resolution.
The used padding was 'valid', as proposed by Nvidia.

- **Convolutional 2**. 36 filters of size 5x5x24. Strides of (2, 2).
- **Convolutional 3**. 48 filters of size 5x5x36. Strides of (2, 2).
- **Convolutional 4**. 64 filters of size 3x3x48. Strides of (1, 1). As can be
observed, the filter size and strides are now reduced, given that the input
images are much smaller.
- **Convolutional 5**. 64 filters of size 3x3x64. Strides of (1, 1).

- **Flatten**. The input for the next layer will have size 1152.
- **Dropout** to mitigate the effects of overfitting.

- **Fully Connected 1**, with 100 neurons + Dropout.
- **Fully Connected 2**, with 50 neurons + Dropout.
- **Fully Connected 3**, with 10 neurons + Dropout.
- **Fully Connected 4**, with 1 neuron, being the output.

All the layers, except for the output layer, have a **ELU activation function**.
The motivation to prefer it over ReLU is that it has a continuous derivative
and x = 0 and does not kill negative activations. The result is a bit smoother
steering output.

**Dropout with probability of keeping = 0.25** is used 
in order to prevent overfitting and have a smooth output.

In addition, all the layers are initialized with the 'glorot_uniform' function,
default in Keras.
The main improvement over Nvidia's implementation is to add the Dropout
layers in order to fight against overfitting.

In total the network has **252219 parameters**, including weights and biases.
The model summary is attached by calling model.summary in Keras:

![alt text][image1b]

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 164 - 176). 

Four drop layers are added after flatten and fully-connected layers as shown above.

The model was trained and validated on different data sets to ensure that the model was not overfitting.
(model.py lines 65-117)

I use different generators for validation and train data set. Basically, train data set has data augmentation
technique involded while validation data set not.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

- **Optimization parameter**: Use Mean Square Error (mse) instead of cross_entropy, 
since this is a regression problem.

- **Optimizer**: Adam, given the great performance on the Traffic Signs Lab.
We use a learning rate of 0.001 (default value). Smaller values like 0.0001 and
0.00001 were also tested, but 0.001 gave the best performance.

- **Batch size**: 64 (or 128), to fit in GPU memory.

- **Number of training samples**: 8036. I tried to put own collected data and udacity
data together which has around 20000 samples. But later I found my own data just
introduce more loss so I finally use 8036 samples of udacity for final training.

- **Maximum number of epochs**: 1. I started by 20 epochs but I quickly noticed
it takes much effort to train it even if I run it in workspace. Then I tried
5 epochs, observed loss and then tested each epoch's model by using callbacks tech. 
I found with data augmentation and proper parameter tuning only one epoch can also 
provide good result on track1. And only one epoch can take 10 mins train because
I have only one tesla gpu available in workspace.

- **Callbacks**: I implement a callback to save the model after every epoch, in
case the validation loss oscillated during train procedure. The train time cost
is really high even if the architecture is relatively simple compared to other
famous CNN. So, it is important to save all midstep model. This way we can compare
different models while skipping the ones with worse perforance. This is
implemented in the `EpochSaverCallback` class:

```python
class EpochSaverCallback(Callback):
    def __init__(self, out_dir):
        self.out_dir = out_dir

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        current_loss = logs.get('val_loss')

        if not self.losses or current_loss < np.amin(self.losses):
            out_dir = os.path.join(self.out_dir, 'e' + str(epoch+1))
            save_model(out_dir, self.model)

        self.losses.append(current_loss)
```
#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 
ne of the tasks of this project was **data collection** using the simulator.
In the first phases of the project own data was collected using the keyboard
to drive the car. However after using the data for training we observed that
the keyboard input was really bad, since it provided almost binary commands,
instead of a continuous range of steering angles.

Most of students recommended the use of a joystick for the project, which
unfortunately we did not have access to. Fortunately, we were provided with
a **dataset from Udacity**, which is what we used in the project.

Nonetheless, it's worthwhile describing here the process that we followed
to manually record training data.

### Strategy
We recorded data in the following way, keeping a constant speed of 30 mph:

- **Normal driving**, with the vehicle kept in the center of the road.
Approximately 3 laps of driving. Example images:

![](res/normal1.jpg) ![](res/normal2.jpg)

- **Recovery**. This part is crucial to manage to get the car driving
the whole lap. Without it, it cannot recover from getting off-center (and
no matter what you do, this will always happen).
First, we drove towards the left or right edge of the road,
without recording. Then we turned on recording, and steered the vehicle back on
track. This was performed at different distances from the center of the lane.
We took 2 laps of recording the vehicle recovering from left to center,
and another 2 laps of recovery from right to center. Example images:

![](res/recovery1.jpg) ![](res/recovery2.jpg)

It was not necessary to drive in the opposite direction, since we extend
the dataset by flipping the image, as will be described later.

### Udacity's dataset
As mentioned before, the final model was trained using Udacity's dataset.
The log file contains 8036 timestamps. For each of them, we have 3 RGB images
and one steering angle, already normalized. Therefore, the complete
dataset contains **24108 images**.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
