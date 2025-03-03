import json
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import os

import nltk
nltk.download('wordnet')


def score_computation(input_file, summarizer, gt_file, max_sentences=5):

    # avg_rouge_1_p = [0, 0, 0, 0, 0]
    # avg_rouge_1_r = [0, 0, 0, 0, 0]
    # avg_rouge_1_f = [0, 0, 0, 0, 0]
    # avg_rouge_2_p = [0, 0, 0, 0, 0]
    # avg_rouge_2_r = [0, 0, 0, 0, 0]
    # avg_rouge_2_f = [0, 0, 0, 0, 0]

    # avg_rouge_l_p = [0, 0, 0, 0, 0]
    # avg_rouge_l_r = [0, 0, 0, 0, 0]
    # avg_rouge_l_f = [0, 0, 0, 0, 0]

    # avg_sbert = [0, 0, 0, 0, 0]

    # avg_bleu = [0, 0, 0, 0, 0]
    # avg_meteor = [0, 0, 0, 0, 0]

    avg_rouge_1_p = [0]*max_sentences
    avg_rouge_1_r = [0]*max_sentences
    avg_rouge_1_f = [0]*max_sentences

    avg_rouge_2_p = [0]*max_sentences
    avg_rouge_2_r = [0]*max_sentences
    avg_rouge_2_f = [0]*max_sentences

    avg_rouge_l_p = [0]*max_sentences
    avg_rouge_l_r = [0]*max_sentences
    avg_rouge_l_f = [0]*max_sentences

    avg_sbert = [0]*max_sentences

    avg_bleu = [0]*max_sentences
    avg_meteor = [0]*max_sentences


    with open(input_file, "r") as input_file, open(gt_file, "r") as description_file:
        data = json.load(input_file)
        descriptions = json.load(description_file)

        for k in tqdm(descriptions.keys()):
            gt = descriptions[k]
            for j in range(1,max_sentences+1):
                summary = " ".join(data[k][str(j)])

                try:
                    rouge = Rouge()
                    rouge_score = rouge.get_scores(summary, gt)
                    rouge_1_p = rouge_score[0]["rouge-1"]["p"]
                    rouge_1_r = rouge_score[0]["rouge-1"]["r"]
                    rouge_1_f = rouge_score[0]["rouge-1"]["f"]
                    rouge_2_p = rouge_score[0]["rouge-2"]["p"]
                    rouge_2_r = rouge_score[0]["rouge-2"]["r"]
                    rouge_2_f = rouge_score[0]["rouge-2"]["f"]
                    rouge_l_p = rouge_score[0]["rouge-l"]["p"]
                    rouge_l_r = rouge_score[0]["rouge-l"]["r"]
                    rouge_l_f = rouge_score[0]["rouge-l"]["f"]
                    
                    with torch.no_grad():
                        sentence_embedding = textModel.encode(summary, convert_to_tensor=True)
                        sentence_embedding = sentence_embedding.detach().cpu()
                        description_embedding = textModel.encode(gt, convert_to_tensor=True)
                        description_embedding = description_embedding.detach().cpu()
                        cosine_scores = util.pytorch_cos_sim(sentence_embedding, description_embedding)
                        sbert_score = cosine_scores[0][0]

                    #sbert_score=torch.tensor(0)

                    bleu_score = sentence_bleu([gt.split()], summary.split())
                    meteor_score = single_meteor_score(gt, summary)

                except Exception as e:
                    print("ERROR: ", e)
                    rouge_1_p = 0
                    rouge_1_r = 0
                    rouge_1_f = 0
                    rouge_2_p = 0
                    rouge_2_r = 0
                    rouge_2_f = 0
                    rouge_l_p = 0
                    rouge_l_r = 0
                    rouge_l_f = 0
                    sbert_score = torch.tensor(0)
                    bleu_score = 0
                    meteor_score = 0

                avg_rouge_1_p[j - 1] += rouge_1_p
                avg_rouge_1_r[j - 1] += rouge_1_r
                avg_rouge_1_f[j - 1] += rouge_1_f
                avg_rouge_2_p[j - 1] += rouge_2_p
                avg_rouge_2_r[j - 1] += rouge_2_r
                avg_rouge_2_f[j - 1] += rouge_2_f
                avg_rouge_l_p[j - 1] += rouge_l_p
                avg_rouge_l_r[j - 1] += rouge_l_r
                avg_rouge_l_f[j - 1] += rouge_l_f
                avg_sbert[j -1] += sbert_score.item()
                avg_bleu[j - 1] += bleu_score
                avg_meteor[j - 1] += meteor_score

        with open(os.path.join("output", summarizer + ".txt"), "w") as output_file:
            for i in range(len(avg_sbert)):
                output_file.write(str(i + 1) + "-sentences summary rouge_1 -> \tP:" + str(avg_rouge_1_p[i] / len(descriptions)) + "\tR:" + str(avg_rouge_1_r[i] / len(descriptions)) + "\tF:" + str(avg_rouge_1_f[i] / len(descriptions)) + "\n")
                output_file.write(str(i + 1) + "-sentences summary rouge_2 -> \tP:" + str(avg_rouge_2_p[i] / len(descriptions)) + "\tR:" + str(avg_rouge_2_r[i] / len(descriptions)) + "\tF:" + str(avg_rouge_2_f[i] / len(descriptions)) + "\n")
                output_file.write(str(i + 1) + "-sentences summary rouge_l -> \tP:" + str(avg_rouge_l_p[i] / len(descriptions)) + "\tR:" + str(avg_rouge_l_r[i] / len(descriptions)) + "\tF:" + str(avg_rouge_l_f[i] / len(descriptions)) + "\n")
                output_file.write(str(i + 1) + "-sentences summary sbert   -> \t" + str(avg_sbert[i] / len(descriptions)) + "\n")
                output_file.write(str(i + 1) + "-sentences summary bleu    -> \t" + str(avg_bleu[i] / len(descriptions)) + "\n")
                output_file.write(str(i + 1) + "-sentences summary meteor  -> \t" + str(avg_meteor[i] / len(descriptions)) + "\n")
                output_file.write("\n")


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    torch.device('cpu')
    device = torch.device('cpu')
textModel = SentenceTransformer('paraphrase-mpnet-base-v2', device=device)
textModel.eval()


# score_computation(os.path.join("output", "Hierarchical-MATeR.json"), "Hierarchical-MATeR", os.path.join("output", "gt.json"))
# score_computation(os.path.join("output", "MATeR.json"), "MATeR", os.path.join("output", "gt.json"))
# score_computation(os.path.join("output", "HiBERT.json"), "HiBERT", os.path.join("output", "gt.json"))
# score_computation(os.path.join("output", "oracle_sbert.json"), "oracle_sbert", os.path.join("output", "gt.json"))
#score_computation(os.path.join("output", "oracle_grounding_chunks.json"), "oracle_grounding_chunks", os.path.join("output", "gt.json"), max_sentences=1)


#score_computation(os.path.join("output", "abstractive-text-summarization_128_1_Hierarchical-MATeR_test.json"), "text-summarization_128_1_Hierarchical-MATeR", os.path.join("output", "gt.json"), max_sentences=1)
#score_computation(os.path.join("output", "abstractive-t5-base_128_1_Hierarchical-MATeR_test.json"), "t5-base_128_1_Hierarchical-MATeR", os.path.join("output", "gt.json"), max_sentences=1)
#score_computation(os.path.join("output", "abstractive-t5-large_128_1_Hierarchical-MATeR_test.json"), "t5-large_128_1_Hierarchical-MATeR", os.path.join("output", "gt.json"), max_sentences=1)
#score_computation(os.path.join("output", "abstractive-bart-large-cnn_128_1_Hierarchical-MATeR_test.json"), "bart-large-cnn_128_1_Hierarchical-MATeR", os.path.join("output", "gt.json"), max_sentences=1)
#score_computation(os.path.join("output", "abstractive-bart-large-cnn_128_2_Hierarchical-MATeR_test.json"), "bart-large-cnn_128_2_Hierarchical-MATeR", os.path.join("output", "gt.json"), max_sentences=1)
#score_computation(os.path.join("output", "abstractive-bart-large-cnn_128_3_Hierarchical-MATeR_test.json"), "bart-large-cnn_128_3_Hierarchical-MATeR", os.path.join("output", "gt.json"), max_sentences=1)
#score_computation(os.path.join("output", "abstractive-bart-large_128_1_Hierarchical-MATeR_test.json"), "bart-large_128_1_Hierarchical-MATeR", os.path.join("output", "gt.json"), max_sentences=1)
#score_computation(os.path.join("output", "abstractive-bart-large_128_2_Hierarchical-MATeR_test.json"), "bart-large_128_2_Hierarchical-MATeR", os.path.join("output", "gt.json"), max_sentences=1)
#score_computation(os.path.join("output", "abstractive-bart-large_128_3_Hierarchical-MATeR_test.json"), "bart-large_128_3_Hierarchical-MATeR", os.path.join("output", "gt.json"), max_sentences=1)

#score_computation(os.path.join("output", "abstractive-bart-large-cnn_128_1_MATeR_test.json"), "bart-large-cnn_128_1_MATeR", os.path.join("output", "gt.json"), max_sentences=1)
#score_computation(os.path.join("output", "abstractive-bart-large-cnn_128_2_MATeR_test.json"), "bart-large-cnn_128_2_MATeR", os.path.join("output", "gt.json"), max_sentences=1)
#score_computation(os.path.join("output", "abstractive-bart-large-cnn_128_3_MATeR_test.json"), "bart-large-cnn_128_3_MATeR", os.path.join("output", "gt.json"), max_sentences=1)

#score_computation(os.path.join("output", "abstractive-bart-large-cnn_128_1_HiBERT_test.json"), "bart-large-cnn_128_1_HiBERT", os.path.join("output", "gt.json"), max_sentences=1)
#score_computation(os.path.join("output", "abstractive-bart-large-cnn_128_2_HiBERT_test.json"), "bart-large-cnn_128_2_HiBERT", os.path.join("output", "gt.json"), max_sentences=1)
#score_computation(os.path.join("output", "abstractive-bart-large-cnn_128_3_HiBERT_test.json"), "bart-large-cnn_128_3_HiBERT", os.path.join("output", "gt.json"), max_sentences=1)

#score_computation(os.path.join("output", "abstractive-chatGPT_def_2_Hierarchical-MATeR_test.json"), "ChatGPT_def_2_Hierarchical-MATeR", os.path.join("output", "gt.json"), max_sentences=1)
#score_computation(os.path.join("output", "abstractive-chatGPT_def_2_0.0_Hierarchical-MATeR_test.json"), "ChatGPT_def_2_0.0_Hierarchical-MATeR", os.path.join("output", "gt.json"), max_sentences=1)
#score_computation(os.path.join("output", "abstractive-chatGPT_def_2_0.5_Hierarchical-MATeR_test.json"), "ChatGPT_def_2_0.5_Hierarchical-MATeR", os.path.join("output", "gt.json"), max_sentences=1)
#score_computation(os.path.join("output", "abstractive-chatGPT_def_2_1.0_Hierarchical-MATeR_test.json"), "ChatGPT_def_2_1.0_Hierarchical-MATeR", os.path.join("output", "gt.json"), max_sentences=1)
score_computation(os.path.join("output", "abstractive-chatGPT_50_2_0.0_Hierarchical-MATeR_test.json"), "ChatGPT_50_2_0.0_Hierarchical-MATeR", os.path.join("output", "gt.json"), max_sentences=1)
score_computation(os.path.join("output", "abstractive-chatGPT_50_2_0.5_Hierarchical-MATeR_test.json"), "ChatGPT_50_2_0.5_Hierarchical-MATeR", os.path.join("output", "gt.json"), max_sentences=1)
score_computation(os.path.join("output", "abstractive-chatGPT_50_2_1.0_Hierarchical-MATeR_test.json"), "ChatGPT_50_2_1.0_Hierarchical-MATeR", os.path.join("output", "gt.json"), max_sentences=1)
score_computation(os.path.join("output", "abstractive-chatGPT_def_2_0.0_5-shot_Hierarchical-MATeR_test.json"), "ChatGPT_def_2_0.0_5-shot_Hierarchical-MATeR", os.path.join("output", "gt.json"), max_sentences=1)
score_computation(os.path.join("output", "abstractive-chatGPT_def_2_0.5_5-shot_Hierarchical-MATeR_test.json"), "ChatGPT_def_2_0.5_5-shot_Hierarchical-MATeR", os.path.join("output", "gt.json"), max_sentences=1)
score_computation(os.path.join("output", "abstractive-chatGPT_def_2_1.0_5-shot_Hierarchical-MATeR_test.json"), "ChatGPT_def_2_1.0_5-shot_Hierarchical-MATeR", os.path.join("output", "gt.json"), max_sentences=1)

#score_computation(os.path.join("output", "Hierarchical-MATeR_test_2020.json"), "TEST2020_Hierarchical-MATeR", os.path.join("output", "test_2020_gt.json"))

#score_computation(os.path.join("output", "Hierarchical-MATeR_test.json"), "Hierarchical-MATeR_with_advertisement", os.path.join("output", "gt_with_advertisement.json"))
#score_computation(os.path.join("output", "MATeR_test.json"), "MATeR_with_advertisement", os.path.join("output", "gt_with_advertisement.json"))
#score_computation(os.path.join("output", "HiBERT_test.json"), "HiBERT_with_advertisement", os.path.join("output", "gt_with_advertisement.json"))

# score_computation(os.path.join("output", "abstractive-bart-large-cnn_128_1_Hierarchical-MATeR_TEMP-0.1_test.json"), "bart-large-cnn_128_1_Hierarchical-MATeR_TEMP-0.1", os.path.join("output", "gt.json"), max_sentences=1)
# score_computation(os.path.join("output", "abstractive-bart-large-cnn_128_1_Hierarchical-MATeR_TEMP-0.5_test.json"), "bart-large-cnn_128_1_Hierarchical-MATeR_TEMP-0.5", os.path.join("output", "gt.json"), max_sentences=1)
#score_computation(os.path.join("output", "abstractive-bart-large-cnn_128_1_Hierarchical-MATeR_TEMP-1.0_test.json"), "bart-large-cnn_128_1_Hierarchical-MATeR_TEMP-1.0", os.path.join("output", "gt.json"), max_sentences=1)
# score_computation(os.path.join("output", "abstractive-bart-large-cnn_128_1_Hierarchical-MATeR_TEMP-2.0_test.json"), "bart-large-cnn_128_1_Hierarchical-MATeR_TEMP-2.0", os.path.join("output", "gt.json"), max_sentences=1)

# score_computation(os.path.join("output", "abstractive-bart-large-cnn_128_1_Hierarchical-MATeR_TEMP-5.0_test.json"), "bart-large-cnn_128_1_Hierarchical-MATeR_TEMP-5.0", os.path.join("output", "gt.json"), max_sentences=1)
