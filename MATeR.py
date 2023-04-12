import torch
import torch.nn as nn
import torch.nn.functional as F
from mlp import MLP

from transformers import Wav2Vec2Processor, Wav2Vec2Model
from transformers import AutoTokenizer, AutoModel



class MATeR(nn.Module):
    def __init__(self, input_dim=1536, output_dim=1, 
        dropout_value = 0.25, audio_model_name="facebook/wav2vec2-base",
        text_model_name = "bert-base-cased",
        device=None, text_tokenizer=None
    ):
        super().__init__()

        self.device = device
        
        # instantiate HuBERT
        # with HuBERT processor
        self.audio_model = Wav2Vec2Model.from_pretrained(audio_model_name)
        self.audio_model = self.audio_model.to(self.device)
        self.len_audio_embeddings = self.audio_model.config.hidden_size

        # instantiate sBERT
        # with sBERT tokenizer ---
        self.text_model = AutoModel.from_pretrained(text_model_name)
        if text_tokenizer == None:
            text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_model.resize_token_embeddings(len(text_tokenizer))
        self.text_model = self.text_model.to(self.device)
        self.len_text_embeddings = self.text_model.config.hidden_size

        self.intermediate_embeddings_len = self.len_audio_embeddings + self.len_text_embeddings

        # instantiate MLP
        self.mlp = MLP(input_dim=self.intermediate_embeddings_len, output_dim = 1)
        self.mlp = self.mlp.to(self.device)


    def forward(self, x_audio, x_text):

        # pass through HuBERT
        #x_audio = torch.stack(x_audio)
        hidden_states_audio = self.audio_model(x_audio).last_hidden_state # model usage
        # avg pooling
        audio_embedding = self.audio_mean_pooling(hidden_states_audio)

        # pass through sBERT (BERT for sequence classification...)
        #model_output = self.text_model(**encoded_input)
        #x_text = torch.stack(x_text)
        input_ids = [x['input_ids'][0] for x in x_text]
        attention_mask = [x['attention_mask'][0] for x in x_text]
        input_ids = torch.stack(input_ids).to(self.device)

        #input_ids = torch.FloatTensor(input_ids)
        model_output = self.text_model(input_ids.to(self.device))
        sentence_embeddings = self.text_mean_pooling(model_output, torch.stack(attention_mask).to(self.device))
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        # concat
        # obtain 768*2 vector = 1536
        overall_embedding = torch.cat((audio_embedding, sentence_embeddings), 1)

        # pass to MLP ()
        # return MLP.forward(...)
        return torch.flatten(self.mlp(overall_embedding)), overall_embedding

    def audio_mean_pooling(self, model_output):
        result = torch.sum(model_output, 1) / model_output.size()[1]
        return result


    def text_mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
