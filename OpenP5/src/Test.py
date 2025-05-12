import re
import numpy as np
import torch
import copy
import random

# idx2word = {}
# word2idx = {}
# print("Building vocab...")
# counter_fitting_embeddings_path = r'/home/lemon/LLM/OpenP5/checkpoint/counter-fitted-vectors.txt'
# with open(counter_fitting_embeddings_path, 'r') as ifile:
#     for line in ifile:
#         word = line.split()[0]
#         if word not in idx2word:
#             idx2word[len(idx2word)] = word
#             word2idx[word] = len(idx2word) - 1

# counter_fitting_cos_sim_path = r'/home/lemon/LLM/OpenP5/checkpoint/counter-fitted-vectors-embedding.npy'
# print('Load pre-computed cosine similarity matrix from {}'.format(counter_fitting_cos_sim_path))
# cos_sim = np.load(counter_fitting_cos_sim_path)

# def TextFooler(synonym_num=50):

#     word = 'test'
#     def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count=10, threshold=0.):
#         sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
#         sim_words, sim_values = [], []
#         for idx, src_word in enumerate(src_words):
#             sim_value = sim_mat[src_word][sim_order[idx]]
#             mask = sim_value >= threshold
#             sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
#             sim_word = [idx2word[id] for id in sim_word]
#             sim_words.append(sim_word)
#             sim_values.append(sim_value)
#         return sim_words, sim_values

#     synonym_words, _ = pick_most_similar_words_batch([word2idx[word]], cos_sim, idx2word, synonym_num, 0.5)

# def replace_token(original_tensor, replace_position, replace_word):
#     profile_list = re.split(r'[ , ]+', original_tensor)
#     replaced_prompt = profile_list[:replace_position] + [replace_word] + profile_list[replace_position + 1:]
#     replaced_prompt = ' '.join(replaced_prompt)
#     return replaced_prompt

if __name__ == '__main__':
    # TextFooler()

    from parrot import Parrot
    import torch
    import warnings
    warnings.filterwarnings("ignore")

    parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=True)

    phrases = ["Can you recommed some upscale restaurants in Newyork?",
            "What are the famous places we should not miss in Russia?"
    ]

    for phrase in phrases:
        print("-"*100)
        print("Input_phrase: ", phrase)
        print("-"*100)
        para_phrases = parrot.augment(input_phrase=phrase)
        for para_phrase in para_phrases:
            print(para_phrase)

'''
    Original Text Attack
'''
# base_loss = self.generate_loss(batch[0], batch[2], batch[1], output_ids, output_attention)
# adversarial_prompt_input = self.generate_prompt_tokens(adversarial_prompt.format(**batch[6][0]))

# # ========================== Candidate Optimization =================================
# adversarial_prompt = batch[7][0]['Input']
# for posit in list(candidate_set.keys()):
#     adversarial_prompt = self.insert_to_prompt(adversarial_prompt, int(posit), self.tokenizer.batch_decode(candidate_set[int(float(posit))], skip_special_tokens=True))

# candidate_ids = list(set([i for sublist in candidate_set.values() for i in sublist if i not in [1, 3]]))
# # print('Candidate IDS', candidate_ids)

# self.optimizer, self.scheduler = self.create_optimizer_and_scheduler(self.model.module.shared.weight, train_epoch)

# gradient_mask = torch.zeros_like(self.model.module.shared.weight)
# gradient_mask[candidate_ids, :] = 1.0

# self.model.zero_grad()
# train_losses = []
# valid_losses = []
# best_epoch = -1
        
# dist.barrier()
# for epoch in (range(train_epoch)):
#     self.model.eval()
#     losses = []
            
#     adversarial_prompt_input = self.generate_prompt_tokens(adversarial_prompt.format(**batch[6][0]))
#     input_ids = adversarial_prompt_input[0].to(self.device)
#     attn = adversarial_prompt_input[1].to(self.device)
#     whole_input_ids = adversarial_prompt_input[2].to(self.device)
#     output_ids = batch[3].to(self.device)
#     output_attention = batch[4].to(self.device)
#     hook = self.model.module.shared.weight.register_hook(lambda grad: grad.mul_(gradient_mask))

#     output = self.model.module(
#         input_ids=input_ids,
#         whole_word_ids=whole_input_ids,
#         attention_mask=attn,
#         labels=output_ids,
#         alpha=self.args.alpha,
#         return_dict=True,
#     )
#     # compute loss masking padded tokens
#     loss = output["loss"]

#     lm_mask = output_attention != 0
#     lm_mask = lm_mask.float()
#     B, L = output_ids.size()
#     loss = loss.view(B, L) * lm_mask
#     loss = -10 * (loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)).mean()

#     # update
#     loss.backward()
#     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                
#     dist.barrier()
                
#     self.optimizer.step()
#     self.scheduler.step()
#     self.model.zero_grad()
#     hook.remove()
                
#     dist.barrier()
# # ================= Candidate Remanpping ===========================
# reshape_matrix = {}
# for i in range(len(candidate_ids)):
#     optimized_token_embedding = self.model.module.shared.weight[candidate_ids[i], :].unsqueeze(0)
#     difference = torch.norm((original_token_embeddings - optimized_token_embedding), p=1, dim=1)
#     index = torch.argmin(difference).detach().cpu().item()
#     for value in self.tokenizer.get_vocab().keys():
#         if index in self.tokenizer.encode(value):
#             break
#     reshape_matrix[candidate_ids[i]] = value 

# adversarial_prompt = batch[7][0]['Input']
# for posit in list(candidate_set.keys()):
#     index = [i for i in candidate_set[int(float(posit))] if i not in [1, 3]][0]
#     candidate_set[int(float(posit))] = reshape_matrix[index]
#     adversarial_prompt = self.insert_to_prompt(adversarial_prompt, int(posit), candidate_set[int(float(posit))])
        
# dist.barrier()
# self.model.module.shared.weight = torch.nn.Parameter(original_token_embeddings)
# adversarial_prompt_ids = self.generate_prompt_tokens(adversarial_prompt.format(**batch[6][0]))

# return adversarial_prompt_ids