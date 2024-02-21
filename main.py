from library import lib
import os
"""
# Specify the API URL and token of the Hugging Face API.

In this tutorial, we are going to use the Hugging Face API.
API means the Application Programming Interface, which allows computer programs to talk to each other.
API_URL is the Hugging Face API URL that points to a model that we want to use.
For more information about the model, see the following page:
- https://huggingface.co/google/vit-base-patch16-224

API_TOKEN is the Hugging Face API token for authentication.
Please do not make the API token public.
For more information about how to use the API, see the following page:
- https://api-inference.huggingface.co/docs/python/html/quicktour.html
"""
API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
API_TOKEN = os.environ['API_TOKEN']
"""
# Use the Hugging Face API to ask a model to make predictions.

Now we use the above "query" function to ask the API to predict what is in this image.
You can replace the my_image variable with your own images.
Note that my_image is a path that points to a file.
In this case, "data/000000039769.jpeg" is a relative path.
This means that "000000039769.jpeg" is placed in the "data" folder.
And the "data" folder is placed together with the main.py script in the same folder.
"""
my_image = "data/000000039769.jpeg"
data = lib.query(my_image, API_URL, API_TOKEN)
"""
# Print the output of the model returned by the API.

The output looks like below:
    [{'score': 0.937, 'label': 'Egyptian cat'}, {'score': 0.038, 'label': 'tabby, tabby cat'}, {'score': 0.014, 'label': 'tiger cat'}, {'score': 0.003, 'label': 'lynx, catamount'}, {'score': 0.001, 'label': 'Siamese cat, Siamese'}]

The output is an array of five dictionaries that represent the top 5 predictions from the model.
Array and dictionary are both data structures.
An array looks like [0, 1, 2, 3], which represents a list of elements (such as numbers).
A dictionary looks like {"key1": "value1", "key2", "value2"}, which represents pairs of keys and values.
In this case, the first element in the array {'score': 0.937, 'label': 'Egyptian cat'} is the first prediction.
It means that the model thinks there are Egyptian cats in the image, with 0.937 probability (which is very high).
"""
print(data)

# The image path for the bicycle image
my_image = 'data/bike/1/271758547_496887881809012_1375450742634622577_n.jpg'
# API key for detr-resnet-50 to recognize bicycle at the proven image
API_URL = 'https://api-inference.huggingface.co/models/facebook/detr-resnet-50'

# To print the result (json format) return by Hugging Face
print(lib.query(my_image, API_URL, API_TOKEN))
# To count how many bicycles in the proven image and print the result
print(lib.count_objects(my_image, API_URL, API_TOKEN, label='bicycle'))
# To draw the bounding boxes around bicycles in the proven image and save the result as test.png (shown in the Files columns) by default
lib.draw_rect(my_image, API_URL, API_TOKEN, label='bicycle')
