import sys
from typing import Any
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.distributed as dist
import logging
from tqdm import tqdm
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import time
import numpy as np
import random
from torch.autograd import Variable
import copy
import re
import torch
import generation_trie as gt
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel, BertForMaskedLM, LlamaForCausalLM, AutoModelForCausalLM, AutoModelForSeq2SeqLM, T5Config
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import AdamW, get_linear_schedule_with_warmup

from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer  # noqa: F402
from sklearn.metrics import roc_auc_score
from multi_agent.RL import RL
import sys



class adversarial_attack():
    def __init__(self, model, tokenizer, attack_mode):
        self.device = torch.device("cuda") 
        self.attack_mode = attack_mode
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(self.device)
        self.model_embedding = self.model.get_input_embeddings().weight
        self.mean, self.std = self.model_embedding.mean(0), self.model_embedding.std(0)

        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        self.victim_model = model
        self.victim_tokenizer = tokenizer

        def is_list_empty_str(lst):
            return all(element == '' for element in lst)

        self.ids_range = self.tokenizer.get_vocab().keys()
        whole_vocabulary_candidate_trie = gt.Trie([ 
            [0] + self.tokenizer.encode(candidate) 
            for candidate in self.ids_range 
            if is_list_empty_str(self.tokenizer.batch_decode([0] + self.tokenizer.encode(candidate), skip_special_tokens=True)) == False])
        self.prefix_allowed_tokens_whole_vocabulary = gt.prefix_allowed_tokens_fn(whole_vocabulary_candidate_trie)

        f = open('./data/ml1m_raw/movies.dat', 'r', encoding='ISO-8859-1')
        movies = f.readlines()
        self.movie_names = [_.split('::')[1] for _ in movies] # movie_names[0] = 'Toy Story (1995)'
        profile_vocabulary_candidate_trie = gt.Trie([ [0] + self.tokenizer.encode(f"{candidate}") for candidate in self.movie_names])
        self.prefix_allowed_tokens_profile = gt.prefix_allowed_tokens_fn(profile_vocabulary_candidate_trie)

        self.bge_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en")
        self.bge_model = AutoModel.from_pretrained("BAAI/bge-large-en").to(self.device)
        self.bge_word_embedding = self.bge_model.embeddings.word_embeddings.weight
        self.bge_vocab = self.bge_tokenizer.get_vocab()
        self.bge_vocab_reversed = {v: k for k, v in self.bge_tokenizer.get_vocab().items()}
        self.bge_model.eval()

        if torch.__version__ >= "2" and sys.platform != "win32":
            self.bge_model = torch.compile(self.bge_model)


    """
           Hybrid Attack 
    """
    def hybrid_attack(self, instruction, input, gold, perturbation_intensity=3, train_epoch=1, inner_iteration=3, neighbour=10):

        # original_sents = self.tokenizer.batch_decode(batch[0], skip_special_tokens=True)
        original_sents = instruction
        history_sequence = input
        profile = copy.deepcopy(input)

        # ========================== Proposed Position Generation ==============================

        # ================= Token Mask ==================
        prompt_list = re.split(r'[ , ]+', original_sents)
        importance = []
        masked_prompt_list = []
        for mask_position in range(len(prompt_list)):
            masked_prompt = self.generate_prompt(self.token_mask(original_sents, mask_position), history_sequence)
            masked_prompt_list.append(masked_prompt)
        importance_prompt = self.generate_loss(masked_prompt_list, torch.tensor([gold]*len(masked_prompt_list)))

        # ================= Profile Mask ==================
        result = re.split('User Preference:', profile)[1]
        preference = re.findall('\"(.*?)\"',re.split('User Unpreference:', result)[0])
        unpreference = re.split('User Unpreference:', result)[1]
        unpreference = re.findall('\"(.*?)\"',re.split('Whether', unpreference)[0])

        if len(preference) > len(unpreference):
            mark = 'preference'
            item_history = preference
        else:
            mark = 'unpreference'
            item_history = unpreference 

        masked_profile_list = []
        if len(item_history) < 3:
            insertion_position = 0
        else:
            for position in range(len(item_history) - 1):
                masked_profile = self.profile_mask(profile, position, mark)
                masked_profile_list.append(self.generate_prompt(instruction, masked_profile))
            importance_profile = self.generate_loss(masked_prompt_list, torch.tensor([gold]*len(masked_prompt_list)))

        importance = importance_prompt + importance_profile

        position = sorted(range(len(importance)), key=lambda k: importance[k], reverse=True)[:perturbation_intensity]
        position = sorted(position, reverse=True)
        # print("The position is : ", position)

        # ========================== Proposed Candidates Generation ==============================

        # with torch.no_grad():
        candidate_set = {}

        for posit in range(perturbation_intensity):
            if position[posit] < len(importance_prompt):
                prompt = f"The original input is \"{original_sents}\" and the prediction is \"{gold}\". Please generate a few letters or words that can change your prediction after inserting it as the {position[posit]}-th word of the input."

                prediction_ids = self.prompt_tuning(instruction, input, gold, prompt, True, self.prefix_allowed_tokens_whole_vocabulary, position=position[posit])

            else:
                if mark == 'preference':
                    prompt = f"According to the user's historical sequence {item_history}, please predict the items that the user is not interested in."
                else:
                    prompt = f"According to the user's historical sequence {item_history}, please predict the items that the user is interested in."
                
                prediction_ids = self.prompt_tuning(instruction, input, gold, prompt, False, self.prefix_allowed_tokens, position=int(float(position[posit])) - len(importance_prompt), mark=mark)

            perturbation_candidates = self.tokenizer.batch_decode(
                prediction_ids, skip_special_tokens=True
            )

            perturbation_candidates = [i for i in perturbation_candidates if i != '']
            # print("The word is : ", perturbation_candidates)

            adversarial_prompt_list = []
            for perturbation_candidate in perturbation_candidates:
                if position[posit] < len(importance_prompt):
                    adversarial_prompt = self.insert_to_prompt(instruction, int(float(position[posit])), perturbation_candidate)
                    adversarial_prompt_list.append(self.generate_prompt(adversarial_prompt, history_sequence))
                else:
                    # ================ Need to consider ',' when tranfering to TALLRec
                    poisoned_profile = self.profile_insertion(profile, int(float(position[posit])) - len(importance_prompt), perturbation_candidate, mark)
                    adversarial_prompt_list.append(self.generate_prompt(instruction, poisoned_profile))
            select_loss = self.generate_loss(adversarial_prompt_list, torch.tensor([gold]*len(adversarial_prompt_list)))

            # print('Original:', select_loss.max())

            # # ===================== Similarity Computation ===========================
            encoded_input = self.bge_tokenizer(adversarial_prompt_list + [self.generate_prompt(instruction, input)], padding=True, truncation=True, return_tensors='pt').to(self.device)
            model_output = self.bge_model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            adversarial_sentence_embeddings = sentence_embeddings[:len(adversarial_prompt_list)]
            original_sentence_embeddings = sentence_embeddings[-1]
            embdedding_difference = cos(adversarial_sentence_embeddings, original_sentence_embeddings)
            select_loss += embdedding_difference * 0.01
            # ===================== Similarity Computation ===========================

            select_loss = np.array(select_loss.detach().cpu().numpy())
            # perturbation_ids = self.tokenizer.encode(perturbation_candidates[np.argmax(select_loss)])
            perturbation_ids = perturbation_candidates[np.argmax(select_loss)]
            candidate_set[int(float(position[posit]))] = perturbation_ids
        # print('Candidate set', candidate_set)

        # print(candidate_set)

        adversarial_prompt = instruction
        poisoned_profile = profile
        for posit in list(candidate_set.keys()):
            if posit < len(importance_prompt):
                if candidate_set[int(float(posit))] in ['{', '}']:
                    candidate_set[int(float(posit))] = f'{candidate_set[int(float(posit))]}{candidate_set[int(float(posit))]}'
                adversarial_prompt = self.insert_to_prompt(adversarial_prompt, int(posit), candidate_set[int(float(posit))])
            else:
                poisoned_profile = self.profile_insertion(poisoned_profile, int(posit) - len(importance_prompt), candidate_set[int(float(posit))], mark)
        # print(candidate_set, adversarial_prompt.format(**batch[6][0]))

        return adversarial_prompt, poisoned_profile
    
    """
            Prompt Tuning
    """
    def prompt_tuning(self, instruction, input, gold, prompt, prompt_or_profile, prefix_allowed_tokens, position, mark=None, epochs=5, initialization=6, generation_num=12):

        original_profile = input
        # ============================== Initialization =========================
        with torch.no_grad():
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
                        adversarial_prompt = self.insert_to_prompt(instruction, position, perturbation_candidate)
                        adversarial_prompt_list.append(self.generate_prompt(adversarial_prompt, input))
                    else:
                        # ================ Need to consider ',' when tranfering to TALLRec
                        poisoned_profile = self.profile_insertion(input, position, perturbation_candidate, mark)
                        adversarial_prompt_list.append(self.generate_prompt(instruction, poisoned_profile))

            loss = self.generate_loss(adversarial_prompt_list, torch.tensor([gold]*len(adversarial_prompt_list)))
            loss, indices = torch.max(loss.reshape(initialization, generation_num), dim=1)

        prefix = random_prefix[torch.argmax(loss)].detach()
        prefix.requires_grad = True
        optimizer = AdamW([prefix], lr=1e-1)
        optimizer.zero_grad()

        max_loss = loss.max()

        # print('Before Prompt Tuning:', max_loss)

        # ========================== Optimization ============================
        for epoch in range(epochs):
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
                perturbation_candidates = self.tokenizer.batch_decode(
                    prediction_ids, skip_special_tokens=True
                )
                perturbation_candidates = [i for i in perturbation_candidates if i != '']
                # print("The word is : ", perturbation_candidates)

                adversarial_prompt_list = []
                for perturbation_candidate in perturbation_candidates:
                    if prompt_or_profile:
                        adversarial_prompt = self.insert_to_prompt(instruction, position, perturbation_candidate)
                        adversarial_prompt_list.append(self.generate_prompt(adversarial_prompt, input))
                    else:
                        # ================ Need to consider ',' when tranfering to TALLRec
                        poisoned_profile = self.profile_insertion(input, position, perturbation_candidate, mark)
                        adversarial_prompt_list.append(self.generate_prompt(instruction, poisoned_profile))

                select_loss = self.generate_loss(adversarial_prompt_list, torch.tensor([gold]*len(adversarial_prompt_list)))

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

            # mean = ((mean * epoch * len(select_loss)) + torch.sum(select_loss)) / ((epoch + 1) * len(select_loss))
            weight = select_loss - max_loss
            weight[weight>=0] = torch.clamp(torch.exp(weight[weight>=0]), 0, 2)
            weight[weight<0] = torch.clamp(-1 * torch.exp(-1 * weight[weight<0]), -2, 0)
            loss = torch.mean(loss * weight)

            # # update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # hook.remove()

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
    def generate_prompt_tokens(self, prompt, batch=False):
        if batch:
            input_texts = prompt
        else:
            input_texts = [prompt]
        inputs = self.tokenizer.batch_encode_plus(
            input_texts, padding="longest", truncation=True, max_length=512
        )
        input_ids = inputs["input_ids"]
        input_attention = inputs["attention_mask"]
        return (
            torch.tensor(input_ids),
            torch.tensor(input_attention),
        )

    def generate_prediction(self, input_ids, attn, prefix_allowed_tokens=None, max_length=256, num=10):
        with torch.no_grad():
            self.model.eval()
            prediction = self.model.generate(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attn.to(self.device),
                    max_length=max_length,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    num_beams=num,
                    num_return_sequences=num,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

        return prediction["sequences"], prediction["sequences_scores"]
    
    def generate_loss(self, prompt, gold):
    # def generate_loss(self, input_ids, attention_mask, output_ids, output_attention, batch=False):
        with torch.no_grad():
            inputs = self.victim_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
            generation_config = GenerationConfig(
                temperature=0,
                top_p=1.0,
                top_k=40,
                num_beams=1,
            )
            with torch.no_grad():
                generation_output = self.victim_model.generate(
                    **inputs,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=128,
                    # batch_size=batch_size,
                )
            s = generation_output.sequences

            # KDD's Loss
            if gold[0] == 0:
                index = 3782
            elif gold[0] == 1: 
                index = 8241
            scores = generation_output.scores[0].softmax(-1)[:, index]
            scores = torch.tensor([i if i > 1e-10 else torch.tensor(1e-20, dtype=torch.float32) for i in scores], dtype=torch.float32).to(self.device) # Avoid Inf
            loss = -1 * torch.log(scores)

        return loss

    def token_mask(self, prompt, mask_position):
        prompt_list = re.split(r'[ , ]+', prompt)
        masked_profile = prompt_list[:mask_position] + prompt_list[:mask_position+1:]
        masked_profile = ' '.join(masked_profile)
        return masked_profile

    def insert_to_prompt(self, original_tensor, insert_position, inserted_word):
        # =============== Need to be changed =====================
        profile_list = re.split(r'[ , ]+', original_tensor)
        if insert_position == len(profile_list):
            inserted_prompt = profile_list[:insert_position] + [inserted_word]
        else:
            inserted_prompt = profile_list[:insert_position] + [inserted_word] + profile_list[insert_position:]
        inserted_prompt = ' '.join(inserted_prompt)
        inserted_prompt = inserted_prompt.replace('identify', f", identify")
        return inserted_prompt

    def profile_mask(self, profile, mask_position, mark):
        result = re.split('User Preference:', profile)[1]
        preference = re.findall('\"(.*?)\"',re.split('User Unpreference:', result)[0])
        unpreference = re.split('User Unpreference:', result)[1]
        unpreference = re.findall('\"(.*?)\"',re.split('Whether', unpreference)[0])
        target = re.findall('\"(.*?)\"', profile)[-1]

        if mark == 'preference':
            profile_list = preference
        else:
            profile_list = unpreference

        if mask_position + 2 >= len(profile_list):
            masked_profile = profile_list[:-2]
        else:
            masked_profile = profile_list[:mask_position] + profile_list[mask_position + 2:]

        if mark == 'preference':
            preference = list(map(lambda x: '"' + str(x) + '"', masked_profile))
            unpreference = list(map(lambda x: '"' + str(x) + '"', unpreference))
        else:
            unpreference = list(map(lambda x: '"' + str(x) + '"', masked_profile))
            preference = list(map(lambda x: '"' + str(x) + '"', preference))

        preference = ' , '.join(preference)
        unpreference = ' , '.join(unpreference)
        masked_profile = f"User Preference: {preference}\nUser Unpreference: {unpreference}\nWhether the user will like the target movie \"{target}\"?"

        return masked_profile

    def profile_insertion(self, profile, insert_position, inserted_item, mark):
        result = re.split('User Preference:', profile)[1]
        preference = re.findall('\"(.*?)\"',re.split('User Unpreference:', result)[0])
        unpreference = re.split('User Unpreference:', result)[1]
        unpreference = re.findall('\"(.*?)\"',re.split('Whether', unpreference)[0])
        target = re.findall('\"(.*?)\"', profile)[-1]

        if mark == 'preference':
            profile_list = preference
        else:
            profile_list = unpreference

        poisoned_profile = profile_list[:insert_position + 1] + [inserted_item] + profile_list[insert_position + 1:]

        if mark == 'preference':
            preference = list(map(lambda x: '"' + str(x) + '"', poisoned_profile))
            unpreference = list(map(lambda x: '"' + str(x) + '"', unpreference))
        else:
            unpreference = list(map(lambda x: '"' + str(x) + '"', poisoned_profile))
            preference = list(map(lambda x: '"' + str(x) + '"', preference))

        preference = ' , '.join(preference)
        unpreference = ' , '.join(unpreference)
        poisoned_profile = f"User Preference: {preference}\nUser Unpreference: {unpreference}\nWhether the user will like the target movie \"{target}\"?"
        return poisoned_profile

    def replace_token(self, original_tensor, replace_position, replace_word):
        profile_list = re.split(r'[ , ]+', original_tensor)
        replaced_prompt = profile_list[:replace_position] + [replace_word] + profile_list[replace_position + 1:]
        replaced_prompt = ' '.join(replaced_prompt)
        replaced_prompt = replaced_prompt.replace('identify', f", identify")

        return replaced_prompt
    
    def generate_prompt(self, instruction, input=None):
        if input:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501

    ### Instruction:
    {instruction}

    ### Input:
    {input}

    ### Response:
    """
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501

    ### Instruction:
    {instruction}

    ### Response:

    """

    def __call__(self, instructions, inputs, golds) -> tuple:
        adversarial_instruction, adversarial_input = [], []
        for instruction, input, gold in zip(instructions, inputs, golds):
            if self.attack_mode in ['CheatAgent']:
                adversarial_batch = self.hybrid_attack(instruction, input, gold)
                adversarial_instruction.append(adversarial_batch[0])
                adversarial_input.append(adversarial_batch[1])
            else:
                adversarial_instruction.append(instruction)
                adversarial_input.append(input)
        return adversarial_instruction, adversarial_input

if __name__ == '__main__':
    # Load model directly

    device = 'cuda:2'
    from transformers import LlamaForCausalLM, LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained("linhvu/decapoda-research-llama-7b-hf")
    model = LlamaForCausalLM.from_pretrained("linhvu/decapoda-research-llama-7b-hf").to(device)
    
    prompt = "Please rephrase the following sentence with the similar semantic content: What is the next item the user will interact?"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        generation_output = model.generate(
            **inputs,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256,
            # batch_size=batch_size,
        )
    s = generation_output.sequences
    output = tokenizer.batch_decode(s, skip_special_tokens=True)
    print(output)