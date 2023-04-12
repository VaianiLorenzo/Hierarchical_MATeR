import torch.utils.data as data
import librosa

class MATeR_Dataset(data.Dataset):

    def __init__(self, path_audio, start_time, end_time, text_sentences, scores, audio_processor, text_processor):
        self.path_audio = path_audio
        self.start_time = start_time
        self.end_time = end_time
        self.text_sentences = text_sentences
        self.scores = scores
        self.audio_processor = audio_processor
        self.text_processor = text_processor

    def __getitem__(self, index):
        data_mono, _ = librosa.load(self.path_audio[index], sr=16000, mono=True, offset=self.start_time[index], duration=self.end_time[index]-self.start_time[index], res_type='soxr_qq')
        text_values  = self.text_processor(self.text_sentences[index], padding="max_length", max_length=384, truncation=True, return_tensors='pt')
        audio_values = self.audio_processor(data_mono, return_tensors="pt", sampling_rate=16000, padding="max_length", max_length=500000, truncation=True).input_values[0]
        return audio_values, text_values, self.scores[index], self.path_audio[index], self.start_time[index], self.end_time[index], self.text_sentences[index]
        #return 0, 0, self.scores[index], self.path_audio[index], self.start_time[index], self.end_time[index], self.text_sentences[index]

    def __len__(self):
        return len(self.text_sentences)