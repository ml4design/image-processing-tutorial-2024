# This is the python script for the image processing module.
"""
# Import python packages.

Often you need to import packages to do fancy works.
In this tutorial, the tool (Replit) takes care of the package installation for you.
But, in the future you may find yourself having the need to install packages.
In that situation, you can use a package manager, such as pip (https://github.com/pypa/pip).
"""
import json
import requests
import cv2


# Below is a reusable function for interacting with the Hugging Face API.
def query(file_path, api_url, api_token):
  """
    Ask the Hugging Face API to run the model and return the result.

    Attributes
    ----------
    file_path : str
        The path to the image file that we want to send to the Hugging Face API.
    api_url : str
        The API URL that points to a specific machine learning model.
    api_token : str
        The API token for authentication.
    """
  # Construct the header of the HTTP request that includes the API token.
  headers = {"Authorization": f"Bearer " + api_token}
  # Read the input data.
  with open(file_path, "rb") as f:
    data = f.read()
  # Make a POST request to the API with the token and input data.
  response = requests.request("POST", api_url, headers=headers, data=data)
  # Return the output from the API
  return json.loads(response.content.decode("utf-8"))


# Below is another resuable function for counting the number of objects in an image
def count_objects(file_path, api_url, api_token, label):
  """
    Ask the Hugging Face API to run the model and count the number of objects.

    Usage example:
        data = count_objects("data/000000039769.jpeg", API_URL, API_TOKEN, "bicycle")

    Attributes
    ----------
    file_path : str
        The path to the image file that we want to send to the Hugging Face API.
    api_url : str
        The API URL that points to a specific machine learning model.
    api_token : str
        The API token for authentication.
    label : str
        The label of the object that we want to count (e.g., "bicycle").
    """
  data = query(file_path, api_url, api_token)
  count = 0
  for d in data:
    if d["label"] == label:
      count += 1
  return count


def draw_rect(file_path, api_url, api_token, label, image_name='./test.png'):
  img = cv2.imread(file_path)
  data = query(file_path, api_url, api_token)

  for d in data:
    if d["label"] == label:
      cv2.rectangle(img, (d['box']['xmin'], d['box']['ymin']),
                    (d['box']['xmax'], d['box']['ymax']), 255, 5)

  cv2.imwrite(image_name, img)
