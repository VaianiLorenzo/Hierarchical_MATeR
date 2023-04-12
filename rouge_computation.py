import json
from rouge import Rouge
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import os


def score_computation(input_file, summarizer, gt_file):

    avg_rouge_1_p = [0, 0, 0, 0, 0]
    avg_rouge_1_r = [0, 0, 0, 0, 0]
    avg_rouge_1_f = [0, 0, 0, 0, 0]
    
    avg_rouge_2_p = [0, 0, 0, 0, 0]
    avg_rouge_2_r = [0, 0, 0, 0, 0]
    avg_rouge_2_f = [0, 0, 0, 0, 0]

    avg_rouge_l_p = [0, 0, 0, 0, 0]
    avg_rouge_l_r = [0, 0, 0, 0, 0]
    avg_rouge_l_f = [0, 0, 0, 0, 0]

    avg_sbert = [0, 0, 0, 0, 0]

    with open(input_file, "r") as input_file, open(gt_file, "r") as description_file:
        data = json.load(input_file)
        descriptions = json.load(description_file)

        for k in tqdm(descriptions.keys()):
            gt = descriptions[k]
            for j in range(1,6):
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

                    sentence_embedding = textModel.encode(summary, convert_to_tensor=True)
                    description_embedding = textModel.encode(gt, convert_to_tensor=True)
                    cosine_scores = util.pytorch_cos_sim(sentence_embedding, description_embedding)
                    sbert_score = cosine_scores[0][0]

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
                    sbert_score = 0

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

        with open(os.path.join("output", summarizer + "_output.txt"), "w") as output_file:
            for i in range(len(avg_sbert)):
                output_file.write(str(i + 1) + "-sentences summary rouge_1 -> \tP:" + str(avg_rouge_1_p[i] / len(descriptions)) + "\tR:" + str(avg_rouge_1_r[i] / len(descriptions)) + "\tF:" + str(avg_rouge_1_f[i] / len(descriptions)) + "\n")
                output_file.write(str(i + 1) + "-sentences summary rouge_2 -> \tP:" + str(avg_rouge_2_p[i] / len(descriptions)) + "\tR:" + str(avg_rouge_2_r[i] / len(descriptions)) + "\tF:" + str(avg_rouge_2_f[i] / len(descriptions)) + "\n")
                output_file.write(str(i + 1) + "-sentences summary rouge_l -> \tP:" + str(avg_rouge_l_p[i] / len(descriptions)) + "\tR:" + str(avg_rouge_l_r[i] / len(descriptions)) + "\tF:" + str(avg_rouge_l_f[i] / len(descriptions)) + "\n")
                output_file.write(str(i + 1) + "-sentences summary sbert   -> \tP:" + str(avg_sbert[i] / len(descriptions)) + "\n")
                output_file.write("\n")




if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    torch.device('cpu')
    device = torch.device('cpu')
textModel = SentenceTransformer('paraphrase-mpnet-base-v2', device=device)


score_computation(os.path.join("output", "Hierarchical_MATeR_output.json"), "Hierarchical_MATeR", os.path.join("output", "gt.json"))
