# Discord Bot of image classification with deep learning.
A model that learns how to sort images from a dataset with multiple road signs.

![image](https://user-images.githubusercontent.com/61854138/193703647-0b0f8261-e818-43ec-8957-bab82d934350.png)

<img src="https://user-images.githubusercontent.com/61854138/193703706-71c30f4f-fd82-453b-b1ba-a66f827f71cc.png" align="right"
     alt="Img by Thaysa Zille" width="210" height="200">
     
## Construction
<p align="justify" >Architecting a model is one of the most fundamental parts of starting training a **convolutional neural network**. In the project, a sequential model was used, which is basically a linear stack of layers. This model creates several layers of an image, analyzes and then compares them to obtain a result. In the sequential model there are some important parameters for its preparation, one of them is conv2D which basically creates a convolution kernel that serves as input layers that helps to produce a more organized output, the next one is Dense, it is used to classify the image based on the conv2D output. There are several other parameters, but these 2 are the most common and useful ones.
</p>

* Detail: The **model** has been reduced to 17 epochs, so it maintains 99% accuracy and lowers the overfit (loss), while models with 25 epochs up end up having a larger overfit.

## Model Construction
```py
model = Sequential()

# 1ª layer
model.add(Conv2D(filters = 64, kernel_size = (3,3), input_shape = x_train.shape[1:], activation = 'relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))

# 2ª layer
model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))

# 3ª layer
model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())

# Dense layer
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))

# Output
model.add(Dense(43, activation = 'softmax'))
```

## Event registration and bot preparation
<details><summary><b>Show instructions</b></summary>
    
  1. Register event:
    ```py
    # Register event
    @client.event

    # Bot ready to start using
    async def on_ready():
    print('Conect {0.user}'.format(client))
    ```

 2. Commands in discord and image analysis:
    ```py
    # Fires whenever a non-authorial message is received
    @client.event
    async def on_message(message):
      if message.author == client.user:
       return
      if message.content.startswith('$oi'):
       await message.channel.send('Eae meu consagrado')

    image_shape = (50,50)
    img = Image.open(dir_path + 'to.jpg').convert('RGB')
    img = img.resize(image_shape)
    img = np.expand_dims(img, axis=0)
    img = np.array(img)

    index = (np.argmax(predictions, axis = 1))
    board = boardClasses[int(index)]
    ```
</details>
<img src="https://user-images.githubusercontent.com/61854138/193707319-eea7942a-7e51-485d-b6b2-effe8ebe6ccb.png" align="left"
     alt="Img by Thaysa Zille" width="310" height="300">

