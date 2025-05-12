import os
from typing import Any
from runner.SingleRunner import SingleRunner
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.distributed as dist
import logging
from tqdm import tqdm
from utils import utils
import torch
import utils.generation_trie as gt
import utils.evaluate as evaluate
from torch.utils.data.distributed import DistributedSampler
from data.TestDataset import TestDataset
from torch.utils.data import DataLoader
from processor.Collator import Collator, TestCollator
import time
import numpy as np
import random
from torch.autograd import Variable
import copy
import re
import torch
from runner.SingleRunner import SingleRunner 
from transformers import AutoTokenizer, AutoModel, BertForMaskedLM, AutoModelForSeq2SeqLM
from transformers import AdamW, get_linear_schedule_with_warmup
from .multi_agent.RL import RL
import sys



class adversarial_attack(SingleRunner):
    def __init__(self, model, tokenizer, train_loader, valid_loader, device, args, rank):
        super().__init__(model, tokenizer, train_loader, valid_loader, device, args)
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(self.device)
        self.model_embedding = self.model.get_input_embeddings().weight
        self.mean, self.std = self.model_embedding.mean(0), self.model_embedding.std(0)

        self.victim_model = model
        self.victim_tokenizer = tokenizer

        self.model.eval()
        self.victim_model.eval()

        self.bge_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en")
        self.bge_model = AutoModel.from_pretrained("BAAI/bge-large-en").to(self.device)
        self.bge_word_embedding = self.bge_model.embeddings.word_embeddings.weight
        self.bge_vocab = self.bge_tokenizer.get_vocab()
        self.bge_vocab_reversed = {v: k for k, v in self.bge_tokenizer.get_vocab().items()}
        self.bge_model.eval()

    """
           Hybrid Attack 
    """
    def hybrid_attack(self, batch, testloader, perturbation_intensity=3, train_epoch=0, inner_iteration=3, neighbour=10):
        def is_list_empty_str(lst):
            return all(element == '' for element in lst)

        # with torch.no_grad():
        if self.args.test_mode == 2:
            batch = batch[:5] + (0,) + batch[5:] # Add a new axis as the user_ids to keep the axis consistent.
            candidates = testloader.dataset.all_items
            candidate_trie = gt.Trie(
                [
                    [0] + self.tokenizer.encode(f"{testloader.dataset.dataset} item_{candidate}")
                    for candidate in candidates
                ]
            )
            prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(candidate_trie)
        elif self.args.test_mode == 1:
            user_idx = int(batch[5][0])
            candidates = set(testloader.dataset.all_items)
            positive = testloader.dataset.positive[testloader.dataset.id2user[user_idx]]
            user_candidate = candidates - positive
            candidate_trie = gt.Trie([[0] + self.tokenizer.encode(f"item_{candidate}") for candidate in user_candidate]
            )
            prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(candidate_trie)

        original_batch = copy.deepcopy(batch)
        # ========================== Define Prefix_allowed_tokens ==============================
        ids_range = self.tokenizer.get_vocab().keys()
        vocabulary_candidate_trie = gt.Trie([ 
            [0] + self.tokenizer.encode(candidate) 
            for candidate in ids_range 
            if is_list_empty_str(self.tokenizer.batch_decode([0] + self.tokenizer.encode(candidate), skip_special_tokens=True)) == False
            ])
        prefix_allowed_tokens_whole_vocabulary = gt.prefix_allowed_tokens_fn(vocabulary_candidate_trie)

        original_sents = batch[7][0]['Input']
        output_ids = batch[3].to(self.device)
        output_attention = batch[4].to(self.device)
        original_profile = copy.deepcopy(batch[6][0]['history'])
        prompt_template = batch[7][0]

        # ========================== Proposed Position Generation ==============================
        prompt_list = re.split(r'[ , ]+', original_sents)
        masked_prompt_list = []
        for mask_position in range(len(prompt_list)):
            if prompt_list[mask_position] not in ['{history}']:
                masked_prompt_list.append(self.token_mask(original_sents, mask_position).format(**batch[6][0]))
            else:
                masked_prompt_list.append(original_sents.format(**batch[6][0]))
        masked_prompt_batch = self.generate_prompt_tokens(masked_prompt_list, True)
        importance_prompt = self.generate_loss(masked_prompt_batch[0], masked_prompt_batch[2], masked_prompt_batch[1], output_ids.repeat(len(prompt_list), 1), output_attention.repeat(len(prompt_list), 1), batch=True).detach().cpu().numpy().tolist()

        masked_profile_list = []
        for position in range(len(re.split(r'[ , ]+', batch[6][0]['history'])) - 1):
        # for position in range(len(re.split(r'[ , ]+', batch[6][0]['history']))):
            masked_profile = self.profile_mask(batch[6][0]['history'], position)
            batch[6][0]['history'] = masked_profile
            masked_profile_list.append(prompt_template['Input'].format(**batch[6][0]))
            batch[6][0]['history'] = original_profile

        masked_profile_batch = self.generate_prompt_tokens(masked_profile_list, True)
        importance_profile = self.generate_loss(masked_profile_batch[0], masked_profile_batch[2], masked_profile_batch[1], output_ids.repeat(len(masked_profile_list), 1), output_attention.repeat(len(masked_profile_list), 1), batch=True).detach().cpu().numpy().tolist()

        importance = importance_prompt + importance_profile

        position = sorted(range(len(importance)), key=lambda k: importance[k], reverse=True)[:perturbation_intensity]
        position = sorted(position, reverse=True)
        # print("The position is : ", position)

        # ========================== Proposed Candidates Generation ==============================

        # with torch.no_grad():
        candidate_set = {}

        for posit in range(perturbation_intensity):
            if position[posit] < len(importance_prompt):
                prompt = f"The original input is \"{original_sents}\" and the prediction is \"{batch[7][0]['Output'].format(**batch[6][0])}\". Please generate a few letters or words that can change your prediction after inserting it as the {position[posit]}-th word of the input."
                prediction_ids = self.prompt_tuning(prompt, True, prefix_allowed_tokens_whole_vocabulary, position=position[posit], batch=batch)

            else:
                prompt = f"According to the user's historical sequence {batch[6][0]['history']}, please predict the items that the user is not interested in."
                prediction_ids = self.prompt_tuning(prompt, False, prefix_allowed_tokens, position=int(float(position[posit])) - len(importance_prompt), batch=batch)

            perturbation_candidates = self.tokenizer.batch_decode(
                prediction_ids, skip_special_tokens=True
            )

            perturbation_candidates = [i for i in perturbation_candidates if i != '']
            # print("The word is : ", perturbation_candidates)

            adversarial_prompt_list = []
            for perturbation_candidate in perturbation_candidates:
                if position[posit] < len(importance_prompt):
                    adversarial_prompt = self.insert_to_prompt(batch[7][0]['Input'], int(float(position[posit])), perturbation_candidate)
                    adversarial_prompt_list.append(adversarial_prompt.format(**batch[6][0]))
                else:
                    # ================ Need to consider ',' when tranfering to TALLRec
                    poisoned_profile = self.profile_insertion(batch[6][0]['history'], int(float(position[posit])) - len(importance_prompt), perturbation_candidate)
                    batch[6][0]['history'] = poisoned_profile
                    adversarial_prompt_list.append(prompt_template['Input'].format(**batch[6][0]))
                    batch[6][0]['history'] = original_profile

            adversarial_prompt_batch = self.generate_prompt_tokens(adversarial_prompt_list, True)
            select_loss = self.generate_loss(adversarial_prompt_batch[0], adversarial_prompt_batch[2], adversarial_prompt_batch[1], output_ids.repeat(len(perturbation_candidates), 1), output_attention.repeat(len(perturbation_candidates), 1), batch=True)

            # # ===================== Similarity Computation ===========================
            encoded_input = self.bge_tokenizer(adversarial_prompt_list + [batch[7][0]['Input'].format(**batch[6][0])], padding=True, truncation=True, return_tensors='pt').to(self.device)
            model_output = self.bge_model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            adversarial_sentence_embeddings = sentence_embeddings[:len(adversarial_prompt_list)]
            original_sentence_embeddings = sentence_embeddings[-1]
            embdedding_difference = cos(adversarial_sentence_embeddings, original_sentence_embeddings)
            select_loss += embdedding_difference * 0.01

            select_loss = np.array(select_loss.detach().cpu().numpy())
            perturbation_ids = perturbation_candidates[np.argmax(select_loss)]
            candidate_set[int(float(position[posit]))] = perturbation_ids
        # print('Candidate set', candidate_set)

        adversarial_prompt = batch[7][0]['Input']
        for posit in list(candidate_set.keys()):
            if posit < len(importance_prompt):
                if candidate_set[int(float(posit))] in ['{', '}']:
                    candidate_set[int(float(posit))] = f'{candidate_set[int(float(posit))]}{candidate_set[int(float(posit))]}'
                adversarial_prompt = self.insert_to_prompt(adversarial_prompt, int(posit), candidate_set[int(float(posit))])
            else:
                poisoned_profile = self.profile_insertion(batch[6][0]['history'], int(posit) - len(importance_prompt), candidate_set[int(float(posit))])
                batch[6][0]['history'] = poisoned_profile
        adversarial_prompt_input = self.generate_prompt_tokens(adversarial_prompt.format(**batch[6][0]))

        # ================== Semantic Similarity =================
        encoded_input = self.bge_tokenizer([adversarial_prompt.format(**batch[6][0])] + [batch[7][0]['Input'].format(**original_batch[6][0])], padding=True, truncation=True, return_tensors='pt').to(self.device)
        model_output = self.bge_model(**encoded_input)
        sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        adversarial_sentence_embeddings = sentence_embeddings[0]
        original_sentence_embeddings = sentence_embeddings[-1]
        cos_sim = cos(adversarial_sentence_embeddings, original_sentence_embeddings)
        norm_1 = torch.norm(adversarial_sentence_embeddings - original_sentence_embeddings, p=1)
        norm_2 = torch.norm(adversarial_sentence_embeddings - original_sentence_embeddings, p=2)

        return adversarial_prompt_input, cos_sim, norm_1, norm_2, adversarial_prompt, batch[6][0]['history']
    
    """
            Prompt Tuning
    """
    def prompt_tuning(self, prompt, prompt_or_profile, prefix_allowed_tokens, position, batch, epochs=5, initialization=10, generation_num=10):
        original_profile = batch[6][0]['history']
        output_ids = batch[3].to(self.device)
        output_attention = batch[4].to(self.device)

        # ============================== Initialization =========================
        with torch.no_grad():
            if epochs == 0:
                random_prefix = torch.randn(initialization, 512).to(self.device)
            else:
                random_prefix = torch.randn(initialization, 512).to(self.device) * self.std + self.mean
            tokenizing = self.tokenizer(prompt, padding="longest", truncation=True, max_length=512)
            prompt_ids, prompt_attention_mask = tokenizing.input_ids, tokenizing.attention_mask
            prompt_embedding = self.model_embedding[prompt_ids, :]
            input_embedding = torch.cat((random_prefix.unsqueeze(1).to(self.device), prompt_embedding.unsqueeze(0).repeat(initialization, 1, 1).to(self.device)), 1)
            attn = torch.cat((torch.tensor([1]).to(self.device), torch.tensor(prompt_attention_mask).to(self.device)), 0).unsqueeze(0).repeat(initialization, 1)

            prediction = self.model.generate(
                    inputs_embeds=input_embedding.to(self.device),
                    attention_mask=attn.to(self.device),
                    max_length=512,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    num_beams=generation_num,
                    num_return_sequences=generation_num,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            prediction_ids, prediction_scores = prediction["sequences"], prediction["sequences_scores"]
            perturbations_whole = self.tokenizer.batch_decode(
                prediction_ids, skip_special_tokens=True
            )
            perturbations = [perturbations_whole[i:i+initialization] for i in range(0, len(perturbations_whole), initialization)]
            # print("The word is : ", perturbation_candidates)

            adversarial_prompt_list = []
            for perturbation_candidates in perturbations:
                for perturbation_candidate in perturbation_candidates:
                    if prompt_or_profile:
                        adversarial_prompt = self.insert_to_prompt(batch[7][0]['Input'], position, perturbation_candidate)
                        adversarial_prompt_list.append(adversarial_prompt.format(**batch[6][0]))
                    else:
                        poisoned_profile = self.profile_insertion(batch[6][0]['history'], position, perturbation_candidate)
                        batch[6][0]['history'] = poisoned_profile
                        adversarial_prompt_list.append(batch[7][0]['Input'].format(**batch[6][0]))
                        batch[6][0]['history'] = original_profile

            adversarial_prompt_batch = self.generate_prompt_tokens(adversarial_prompt_list, True)
            loss = self.generate_loss(adversarial_prompt_batch[0], adversarial_prompt_batch[2], adversarial_prompt_batch[1], output_ids.repeat(len(adversarial_prompt_list), 1), output_attention.repeat(len(adversarial_prompt_list), 1), batch=True)
            loss, indices = torch.max(loss.reshape(initialization, generation_num), dim=1)

        prefix = random_prefix[torch.argmax(loss)].detach()
        prefix.requires_grad = True
        optimizer = AdamW([prefix], lr=1e-1)
        optimizer.zero_grad()

        max_loss = loss.max()

        # ========================== Optimization ============================
        for epoch in range(epochs):
            input_embedding = torch.cat((prefix.unsqueeze(0).to(self.device), prompt_embedding.to(self.device)), 0).unsqueeze(0)
            attn = torch.cat((torch.tensor([1]).to(self.device), torch.tensor(prompt_attention_mask).to(self.device)), 0).unsqueeze(0)

            prediction = self.model.generate(
                    inputs_embeds=input_embedding.to(self.device),
                    attention_mask=attn.to(self.device),
                    max_length=512,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    num_beams=generation_num,
                    num_return_sequences=generation_num,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            prediction_ids, prediction_scores = prediction["sequences"], prediction["sequences_scores"]
            perturbation_candidates = self.tokenizer.batch_decode(
                prediction_ids, skip_special_tokens=True
            )
            perturbation_candidates = [i for i in perturbation_candidates if i != '']
            # print("The word is : ", perturbation_candidates)

            adversarial_prompt_list = []
            for perturbation_candidate in perturbation_candidates:
                if prompt_or_profile:
                    adversarial_prompt = self.insert_to_prompt(batch[7][0]['Input'], position, perturbation_candidate)
                    adversarial_prompt_list.append(adversarial_prompt.format(**batch[6][0]))
                else:
                    # ================ Need to consider ',' when tranfering to TALLRec
                    poisoned_profile = self.profile_insertion(batch[6][0]['history'], position, perturbation_candidate)
                    batch[6][0]['history'] = poisoned_profile
                    adversarial_prompt_list.append(batch[7][0]['Input'].format(**batch[6][0]))
                    batch[6][0]['history'] = original_profile

            adversarial_prompt_batch = self.generate_prompt_tokens(adversarial_prompt_list, True)
            select_loss = self.generate_loss(adversarial_prompt_batch[0], adversarial_prompt_batch[2], adversarial_prompt_batch[1], output_ids.repeat(len(adversarial_prompt_list), 1), output_attention.repeat(len(adversarial_prompt_list), 1), batch=True)

            tokenizing = self.tokenizer(perturbation_candidates, padding="longest", truncation=True, max_length=512, return_tensors="pt")
            suboptimal_perturbation_ids, suboptimal_output_attention = tokenizing.input_ids.to(self.device), tokenizing.attention_mask.to(self.device)

            output = self.model(
                inputs_embeds=input_embedding.repeat(suboptimal_perturbation_ids.shape[0], 1, 1).to(self.device),
                attention_mask=attn.repeat(suboptimal_perturbation_ids.shape[0], 1, 1).to(self.device),
                labels=suboptimal_perturbation_ids.to(self.device),
                return_dict=True,
            )
            # compute loss masking padded tokens
            # loss = output["loss"]
            logits = output['logits']
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            loss = loss_fct(logits.view(-1, logits.size(-1)), suboptimal_perturbation_ids.to(self.device).view(-1))

            lm_mask = suboptimal_output_attention != 0
            lm_mask = lm_mask.float()
            B, L = suboptimal_perturbation_ids.size()
            loss = loss.view(B, L) * lm_mask
            loss = (loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1))

            weight = select_loss - max_loss
            weight[weight>=0] = torch.clamp(torch.exp(weight[weight>=0]), 0, 2)
            weight[weight<0] = torch.clamp(-1 * torch.exp(-1 * weight[weight<0]), -2, 0)
            loss = torch.mean(loss * weight)

            # # update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # hook.remove()

        if epochs == 0:
            input_embedding = torch.cat((prefix.unsqueeze(0).to(self.device), prompt_embedding.to(self.device)), 0).unsqueeze(0)
            attn = torch.cat((torch.tensor([1]).to(self.device), torch.tensor(prompt_attention_mask).to(self.device)), 0).unsqueeze(0)

        with torch.no_grad():
            prediction = self.model.generate(
                    inputs_embeds=input_embedding.to(self.device),
                    attention_mask=attn.to(self.device),
                    max_length=512,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    num_beams=generation_num,
                    num_return_sequences=generation_num,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            prediction_ids, prediction_scores = prediction["sequences"], prediction["sequences_scores"]
        return prediction_ids



    '''
    Some useful functions below
    '''
    def write_data_to_file(self, folder_path, file_name, data):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'a') as file:
            file.write(str(data) + '\n')


    def calculate_whole_word_ids(self, tokenized_text, input_ids):
        whole_word_ids = []
        curr = 0
        for i in range(len(tokenized_text)):
            if tokenized_text[i] == "<pad>":
                curr = 0
            if tokenized_text[i].startswith("▁"):
                curr += 1
                whole_word_ids.append(curr)
            else:
                whole_word_ids.append(curr)
        return whole_word_ids[: len(input_ids) - 1] + [0]  # [0] for </s>
        
    def generate_prompt_tokens(self, prompt, batch=False, whole_word=True):
        if batch:
            input_texts = prompt
        else:
            input_texts = [prompt]
        inputs = self.victim_tokenizer.batch_encode_plus(
            input_texts, padding="longest", truncation=True, max_length=512
        )
        input_ids = inputs["input_ids"]
        input_attention = inputs["attention_mask"]

        if whole_word:
            whole_word_ids = []
            for input_id in input_ids:
                tokenized_text = self.victim_tokenizer.convert_ids_to_tokens(input_id)
                whole_word_id = self.calculate_whole_word_ids(tokenized_text, input_id)
                whole_word_ids.append(whole_word_id)
            return (
                torch.tensor(input_ids),
                torch.tensor(input_attention),
                torch.tensor(whole_word_ids),
            )
        else:
            return (
                torch.tensor(input_ids),
                torch.tensor(input_attention),
            )

    def generate_prediction(self, input_ids, attn, prefix_allowed_tokens=None, max_length=256, num_beams=10, num_return_sequences=10):
        with torch.no_grad():
            self.model.eval()
            prediction = self.model.generate(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attn.to(self.device),
                    max_length=max_length,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

        return prediction["sequences"], prediction["sequences_scores"]
    
    def generate_loss(self, input_ids, whole_word_ids, attention_mask, output_ids, output_attention, batch=False):
        with torch.no_grad():
            self.victim_model.eval()
            output = self.victim_model.module(
                input_ids=input_ids.to(self.device),
                whole_word_ids=whole_word_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                labels=output_ids,
                alpha=self.args.alpha,
                return_dict=True,
            )
            # compute loss masking padded tokens
            loss = output["loss"]
            lm_mask = output_attention != 0
            lm_mask = lm_mask.float()
            B, L = output_ids.size()
            loss = loss.view(B, L) * lm_mask
            loss = (loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1))
        return loss

    def profile_mask(self, profile, mask_position, mask_pair=True):
        profile_list = re.split(r'[ , ]+', profile)
        if mask_pair:
            if mask_position + 2 >= len(profile_list):
                masked_profile = profile_list[:-2]
            else:
                masked_profile = profile_list[:mask_position] + profile_list[mask_position + 2:]
        else:
            masked_profile = profile_list[:mask_position] + profile_list[mask_position+1:]
        masked_profile = ' , '.join(masked_profile)
        return masked_profile

    def profile_insertion(self, profile, insert_position, inserted_item, left_or_right=None):# left_or_right : True -> Left; False -> Right; None -> Middle
        profile_list = re.split(r'[ , ]+', profile)
        if left_or_right == None:
            poisoned_profile = profile_list[:insert_position + 1] + [inserted_item] + profile_list[insert_position + 1:]
        else:
            if left_or_right:
                poisoned_profile = profile_list[:insert_position] + [inserted_item] + profile_list[insert_position:]
            else:
                poisoned_profile = profile_list[:insert_position+1] + [inserted_item] + profile_list[insert_position+1:]
        poisoned_profile = ' , '.join(poisoned_profile)
        return poisoned_profile
    
    def token_mask(self, prompt, mask_position):
        prompt_list = re.split(r'[ , ]+', prompt)
        masked_profile = prompt_list[:mask_position] + prompt_list[mask_position+1:]
        masked_profile = ' '.join(masked_profile)
        return masked_profile

    def insert_to_prompt(self, original_tensor, insert_position, inserted_word, left_or_right=True): # left_or_right : True -> Left; False -> Right
        if isinstance(inserted_word, list):
            inserted_word = ''.join(inserted_word)
            if inserted_word == '':
                return original_tensor
        profile_list = re.split(r'[ , ]+', original_tensor)
        if left_or_right:
            if insert_position == len(profile_list):
                inserted_prompt = profile_list[:insert_position] + [inserted_word]
            else:
                inserted_prompt = profile_list[:insert_position] + [inserted_word] + profile_list[insert_position:]
        else:
            if insert_position == len(profile_list):
                inserted_prompt = profile_list[:insert_position] + [inserted_word]
            else:
                inserted_prompt = profile_list[:insert_position+1] + [inserted_word] + profile_list[insert_position+1:]
        inserted_prompt = ' '.join(inserted_prompt)
        return inserted_prompt

    def replace_token(self, original_tensor, replace_position, replace_word):
        profile_list = re.split(r'[ , ]+', original_tensor)
        replaced_prompt = profile_list[:replace_position] + [replace_word] + profile_list[replace_position + 1:]
        replaced_prompt = ' '.join(replaced_prompt)
        return replaced_prompt

    def __call__(self, batch, testloader, prefix_prompt=None, prefix_profile=None, optimizer_prompt=None, optimizer_profile=None) -> tuple:
        if self.args.attack_mode in ['CheatAgent']:
            adversarial_batch, cos_sim, norm_1, norm_2, prompt, profile = self.hybrid_attack(batch, testloader)
        else:
            if self.args.test_mode == 2:
                batch = batch[:5] + (0,) + batch[5:] # Add a new axis as the user_ids to keep the axis consistent.
            adversarial_batch = batch
            cos_sim, norm_1, norm_2 = torch.tensor(1.0).to(self.device), torch.tensor(0.0).to(self.device), torch.tensor(0.0).to(self.device)
            prompt, profile = batch[7][0]['Input'], batch[6][0]

        return adversarial_batch, cos_sim, norm_1, norm_2