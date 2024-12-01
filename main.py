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


def predict(path):
    speech, sr = librosa.load(path, sr=sampling_rate)
    features = processor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"label": config.id2label[i], "score": round(score, 5)} for i, score in enumerate(scores)]
    return outputs


print(predict("speech.wav"))