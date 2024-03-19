import cohere
import json
from cohere.responses.classify import Example
co = cohere.Client('jY0mYl6MTtHnKLxMAPYBGg69rWrxt2CW9Kh0J1TJ')


examples=[ 
          Example("I want to have lunch", "eat out"),  Example("I want to have apple", "eat out"),  Example("I want to go to cafe", "eat out"),  Example("I want to have try new restaurants", "eat out"),  Example("I want to train", "gym buddy"),  Example("I want to exercise", "gym buddy"),  
          Example("I want to do cardio", "gym buddy"),Example("I want some to teach me", "Looking for mentor"), Example("I want to learn css", "Looking for mentor"), Example("I want to learn skiing", "Looking for mentor")
          ]
inputs=["I want to have lunch", "I want to go to the gym", "I want to find a mentor"]

response = co.classify(
  model='embed-english-v2.0',
  inputs=inputs,
  examples=examples,
)
classifications_dict = vars(response)
print('The confidence levels of the labels are: {}'.format(response.classifications))
