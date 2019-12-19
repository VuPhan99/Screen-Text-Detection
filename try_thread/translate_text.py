import threading
import cv2
from googletrans import Translator

def translateText(text):
    translator = Translator(service_urls=[
      'translate.google.com.vn',
    ])
    textTranslated = translator.translate(
		text, src='en', dest='vi'
	)
    return textTranslated.text
