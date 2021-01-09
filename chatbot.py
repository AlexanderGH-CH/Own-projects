# -*- coding: utf-8 -*-
"""Chatbot.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14R05AEDlxHskdhRyGscA4nWikyXW4PZh
"""

#A chatbot created using Python

#About Chatbots:
#Two variants of chatbots
#Self learning bots: using a machine learning approach to chat
#rule based bot, answers questions based on rules that its trained on
#This program makes use of the second variant, thus this is a rule based bot

#with rule based approach it may occure that the computer doesn't know the answer

#import the library
from nltk.chat.util import Chat, reflections

#create a list of patterns and responses
pairs = [
        ['my name is (.*)', ['Hi %1']],
        ['(hi|hello|hola|holla|bonjour|bienvenuti)', ['Hey there', 'hi there','haayy','whatsup buddy']]
        ['(.*) in(.*) is fun', ['%1 in %2 is indeed fun']]
        ['(.*)(location|city) ?', 'Tokyo', 'Japan', 'Bangkok']
        ['(.*) created you ?', ['Alexander Häfeli did using NLTK']]
        ['How is the weather in (.*) ?', ['The weather in %1 is amazing like always']]
        ['(.*)help(.*)', [' I can help you or google it :-)']]
        ['(.*) your name ?', ['My name is Evian']]
    ]

reflections

my_dummy_reflections ={
    'go': 'gone',
    'hello': 'hey there'
}

chat = Chat(pairs, reflections) #for your own relfections replace it with my_dummy_reflections
#chat._substitute('you are amazing')
chat.converse()