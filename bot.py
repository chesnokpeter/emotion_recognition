import asyncio, sys, io
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import FSInputFile
import logging
import librosa
import torch
import torch.nn.functional as F
from transformers import AutoConfig, Wav2Vec2Processor, AutoModelForAudioClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name_or_path = "KELONMYOSA/wav2vec2-xls-r-300m-emotion-ru"
config = AutoConfig.from_pretrained(model_name_or_path)
processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
sampling_rate = processor.feature_extractor.sampling_rate
model = AutoModelForAudioClassification.from_pretrained(model_name_or_path, trust_remote_code=True).to(device)

API_TOKEN = '7260511059:AAFfHvH7AOnBuc3ByUS5LX2f2Gu2ScVZ8ig'

bot = Bot(token=API_TOKEN)
dp = Dispatcher()


def predict(audio_buffer):
    speech, sr = librosa.load(io.BytesIO(audio_buffer), sr=sampling_rate)
    features = processor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"label": config.id2label[i], "score": round(score, 5)} for i, score in enumerate(scores)]
    return outputs


@dp.message(Command(commands=['start']))
async def start(message: types.Message):
    await message.answer(f'Здравствуй, {message.from_user.first_name}. Отправь мне аудио, чтобы по нему я смог определить эмоцию человека')

@dp.message()
async def handle_audio_voice(message: types.Message):
    if message.content_type not in [types.ContentType.AUDIO, types.ContentType.VOICE]:
        await message.answer(f'Здравствуй, {message.from_user.first_name}. Отправь мне аудио, чтобы по нему я смог определить эмоцию человека')
        return

    if message.audio:
        file_info = await bot.get_file(message.audio.file_id)
        file_path = file_info.file_path
    elif message.voice:
        file_info = await bot.get_file(message.voice.file_id)
        file_path = file_info.file_path

    downloaded_file = await bot.download_file(file_path)
    audio_buffer = io.BytesIO(downloaded_file.read())

    predictions: list = predict(audio_buffer.getvalue())

    await message.answer(f'Предсказанные эмоции: {max(predictions, key=lambda x: x["score"])["label"]}')



async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
