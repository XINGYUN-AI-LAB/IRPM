import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from transformers import AutoModel, AutoTokenizer

chencherry = SmoothingFunction()

class EasyTextEmbedding:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)

    def encode(self, texts):
        # Tokenize the texts
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)    #['last_hidden_state', 'past_key_values']

        # Mean Pooling
        attention_mask = encoded_input['attention_mask']
        token_embeddings = model_output[0]    #[1, L, 3584]

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)   #[1, 3584] 

        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def similarity(self, text1, text2):
        # Compute embeddings
        embedding1 = self.encode([text1])[0]    #3584
        embedding2 = self.encode([text2])[0]    #3584

        # Compute cosine similarity
        similarity = F.cosine_similarity(embedding1, embedding2, dim=0)
        return similarity.item()

model = EasyTextEmbedding('sentence-transformers/all-MiniLM-L6-v2')

def compute_score(solution, ground_truth):
    ## length reward
    len_solution=len(solution)
    len_truth=len(ground_truth)

    if abs(len_solution-len_truth)<100:
        length_reward=1
    elif abs(len_solution-len_truth)<500:
        length_reward=1-((abs(len_solution-len_truth)-100)/400)
    else:
        length_reward=0

    ## blue reward
    solution_split=[word for word in solution]
    ground_truth_split=[word for word in ground_truth]

    blue_reward=sentence_bleu([ground_truth_split], solution_split, smoothing_function=chencherry.method1)
    blue1_reward=sentence_bleu([ground_truth_split], solution_split, weights=[1, 0, 0, 0], smoothing_function=chencherry.method1)

    ## similarity reward
    similarity_reward = model.similarity(solution, ground_truth)

    #reward=0.08*length_reward+0.6*blue_reward+0.2*blue1_reward+0.12*similarity_reward
    reward=0.1*length_reward+0.6*blue_reward+0.3*similarity_reward
    return reward
