import asyncio
import sys
import io
import os
import nltk
from pydub import AudioSegment
from speech_recognition import Recognizer, AudioFile, UnknownValueError, RequestError
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import FSInputFile, CallbackQuery
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, WebAppInfo
from aiogram.utils.keyboard import InlineKeyboardBuilder
import logging
import librosa
import torch
import torch.nn.functional as f
from transformers import AutoConfig, Wav2Vec2Processor, AutoModelForAudioClassification


file_put = 'audio/audio_2024-12-01_23-11-51.ogg'
audio = AudioSegment.from_file(file_put, 'ogg')
audio.export(file_put, format='wav')
reka = Recognizer()
with AudioFile(file_put) as source:
    audio_data = reka.record(source)
    textovik = reka.recognize_google(audio_data, language='ru-RU')
print(textovik)
  
# except UnknownValueError:
#     await callback.message.answer('Не удалось распознать текст, попробуйте ещё раз')
# except RequestError:
#     await callback.message.answer('Не удалось распознать текст, попробуйте ещё раз')
# moodila = predict_mood(textovik)
# await callback.message.answer(f'Человек чувствует {moodila}')
