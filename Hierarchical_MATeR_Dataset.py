import torch.utils.data as data
import librosa
import numpy as np

class Hierarchical_MATeR_Dataset(data.Dataset):

    def __init__(self, multimodal_embeddings, sentence_list):
        self.multimodal_embeddings = multimodal_embeddings
        self.groups = sentence_list.groupby('podcast_id')
        self.group_names = list(self.groups.groups.keys())


    def __getitem__(self, index):
        group = self.groups.get_group(self.group_names[index])

        scores = group['score'].values
        indexes = group['index'].values
        starts = group['start'].values
        ends = group['end'].values
        sentences = group['sentence'].values
        path = group['path'].values[0]
        podcast_id = group['podcast_id'].values[0]

        starts, scores, indexes, ends, sentences = zip(*sorted(zip(starts, scores, indexes, ends, sentences)))
        starts = np.array(starts)
        scores = np.array(scores)
        indexes = np.array(indexes)
        ends = np.array(ends)

        embeddings = [self.multimodal_embeddings[i] for i in indexes]

        if len(embeddings) > 128:
            starts = starts[:128]
            ends = ends[:128]
            sentences = sentences[:128]
            scores = scores[:128]
            attention_mask = np.ones(128)
            embeddings = embeddings[:128]
            embeddings = np.array(embeddings)
        elif len(embeddings) < 128:
            starts = np.concatenate((starts, np.zeros(128-len(starts))))
            ends = np.concatenate((ends, np.zeros(128-len(ends))))
            scores = np.concatenate((scores, np.zeros(128-len(scores))))
            sentences = list(sentences) + ['']*(128-len(sentences))
            attention_mask = np.concatenate((np.ones(len(embeddings)), np.zeros(128-len(embeddings))))
            embeddings = np.array(embeddings)
            embeddings = np.concatenate((embeddings, np.zeros((128-len(embeddings), 1536))))
        else:
            embeddings = np.array(embeddings)
            attention_mask = np.ones(128)
            
        return embeddings, attention_mask, starts, ends, sentences, scores, podcast_id, path


    def __len__(self):
        return len(self.group_names)
