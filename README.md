<div align="center">
  
[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

</div>

# <div align="center">Data Augmentation</div>
<div align="center"><img src="https://github.com/Pradnya1208/Data-Augmentation/blob/main/output/augment.gif?raw=true"></div>


## Overview:
Data augmentation is the process of increasing the amount and diversity of data. We do not collect new data, rather we transform the already present data. 
Data augmentation is an integral process in deep learning, as in deep learning we need large amounts of data and in some cases it is not feasible to collect thousands or millions of images, so data augmentation comes to the rescue.

It helps us to increase the size of the dataset and introduce variability in the dataset.
#### Operations in data augmentation
The most commonly used operations are-
- **Rotation**: Rotation operation as the name suggests, just rotates the image by a certain specified degree.
  In the example below, I specified the rotation degree as 40.
- **Shearing**: Shearing is also used to transform the orientation of the image.
- **Zooming**: Zooming operation allows us to either zoom in or zoom out.
- **Cropping**: Cropping allows us to crop the image or select a particular area from an image.
- **Flipping**: Flipping allows us to flip the orientation of the image. We can use horizontal or vertical flip.You should use this feature carefully as there will be scenarios where this operation might not make much sense e.g. suppose you are designing a facial recognition system, then it is highly unlikely that a person will stand upside down in front of a camera, so you can avoid using the vertical flip operation.
- **Changing the brightness level**: This feature helps us to combat illumination changes.You can encounter a scenario where most of your dataset comprises of images having a similar brightness level e.g. collecting the images of employees entering the office, by augmenting the images we make sure that our model is robust and is able to detect the person even in different surroundings
## Dataset:
[Flowers Dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers)
<br>
Checkout [Dataset features](https://knowyourdata-tfds.withgoogle.com/#tab=STATS&dataset=tf_flowers)
<br>
The flowers dataset has five classes.
### An image from the dataset:
<img src="https://github.com/Pradnya1208/Data-Augmentation/blob/main/output/tulip.PNG?raw=true">


## Implementation:

**Libraries:**  `NumPy` `pandas` `matplotlib` `tensorflow` `keras`### Keras preprocessing layer:
#### Resizing and Rescaling:
You can use the Keras preprocessing layers to resize your images to a consistent shape with `tf.keras.layers.Resizing`, and to rescale pixel values with `tf.keras.layers.Rescaling`.
<img src="https://github.com/Pradnya1208/Data-Augmentation/blob/main/output/resize,%20rescale.PNG?raw=true">
<br>
#### Rotation and Flip:
You can use the Keras preprocessing layers for data augmentation as well, such as `tf.keras.layers.RandomFlip` and `tf.keras.layers.RandomRotation`.
<img src="https://github.com/Pradnya1208/Data-Augmentation/blob/main/output/fliprotate.PNG?raw=true">
<br>
There are a variety of preprocessing layers you can use for data augmentation including `tf.keras.layers.RandomContrast`, `tf.keras.layers.RandomCrop`, `tf.keras.layers.RandomZoom`, and others.
<br>

#### Keras preprocessing layers:
There are two ways you can use these preprocessing layers, with important trade-offs.
-  Make the preprocessing layers part of your model<br>
```
model = tf.keras.Sequential([
  # Add the preprocessing layers you created earlier.
  resize_and_rescale,
  data_augmentation,
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  # Rest of your model.
])
```
<br>

-  Apply the preprocessing layers to your dataset<br>
```
aug_ds = train_ds.map(
  lambda x, y: (resize_and_rescale(x, training=True), y))
```

#### Custom data augmentation:
There are two ways:
- First, you will create a `tf.keras.layers.Lambda layer`. This is a good way to write concise code.<br>
```
def random_invert(factor=0.5):
  return layers.Lambda(lambda x: random_invert_img(x, factor))

random_invert = random_invert()
```
<br>
<img src="https://github.com/Pradnya1208/Data-Augmentation/blob/main/output/invert.PNG?raw=true">
<br>

- Next, you will write a new layer via subclassing, which gives you more control.<br>
```
class RandomInvert(layers.Layer):
  def __init__(self, factor=0.5, **kwargs):
    super().__init__(**kwargs)
    self.factor = factor

  def call(self, x):
    return random_invert_img(x)
```
<br>
<img src="https://github.com/Pradnya1208/Data-Augmentation/blob/main/output/subclassing.PNG?raw=true"><br>

#### Using tf.image:
The above Keras preprocessing utilities are convenient. But, for finer control, you can write your own data augmentation pipelines or layers using `tf.data` and `tf.image`.
- **flip an image**:<br>
  ```
  flipped = tf.image.flip_left_right(image)
  visualize(image, flipped)
  ```
  <img src = "https://github.com/Pradnya1208/Data-Augmentation/blob/main/output/flip.PNG?raw=true">

- **Grayscale an image**:<br>
  ```
  grayscaled = tf.image.rgb_to_grayscale(image)
  visualize(image, tf.squeeze(grayscaled))
  _ = plt.colorbar()
  ```
  <img src="https://github.com/Pradnya1208/Data-Augmentation/blob/main/output/grayscale.PNG?raw=true">
  <br>

- **Image Saturation**:<br>
  ```
  saturated = tf.image.adjust_saturation(image, 3)
  visualize(image, saturated)
  ```
  <img src="https://github.com/Pradnya1208/Data-Augmentation/blob/main/output/saturation.PNG?raw=true">
  <br>

- **Image brightness**:<br>
  ```
  bright = tf.image.adjust_brightness(image, 0.4)
  visualize(image, bright)
  ```
  <img src = "https://github.com/Pradnya1208/Data-Augmentation/blob/main/output/brightness.PNG?raw=true">
  <br>

- **Center cropping**:<br>
  ```
  cropped = tf.image.central_crop(image, central_fraction=0.5)
  visualize(image,cropped)
  ```
  <img src="https://github.com/Pradnya1208/Data-Augmentation/blob/main/output/crop.PNG?raw=true">


#### Random Transformations:
Applying random transformations to the images can further help generalize and expand the dataset. The current tf.image API provides eight such random image operations (ops):
- tf.image.stateless_random_brightness
- tf.image.stateless_random_contrast
- tf.image.stateless_random_crop
- tf.image.stateless_random_flip_left_right
- tf.image.stateless_random_flip_up_down
- tf.image.stateless_random_hue
- tf.image.stateless_random_jpeg_quality
- tf.image.stateless_random_saturation
Random brightness and contrast:<br>
<img src="https://github.com/Pradnya1208/Data-Augmentation/blob/main/output/random_brightness.PNG?raw=true" width="40%"> <img src="https://github.com/Pradnya1208/Data-Augmentation/blob/main/output/random_contrast.PNG?raw=true" width="40%">

### Learnings:
`Data Augmentation`
`Subclassing`






## References:
[Data Augmentation | Tensorflow](https://www.tensorflow.org/tutorials/images/data_augmentation)
<br>
[Subclassing](https://www.tensorflow.org/guide/keras/custom_layers_and_models)

### Feedback

If you have any feedback, please reach out at pradnyapatil671@gmail.com


### 🚀 About Me
#### Hi, I'm Pradnya! 👋
I am an AI Enthusiast and  Data science & ML practitioner


[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]



