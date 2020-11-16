import argparse
import json
import os
import random
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
import copy
from torch.nn import CrossEntropyLoss

from transformers import (
    BertConfig,
    BertForPreTraining,
    BertForPathLM,
    BertForSequenceClassification,
    BertForNextSentencePrediction,
    BERTPathTokenizer,
    XLNetConfig,
    XLNetLMHeadModel,
    XLNetForSequenceClassification,
    XLNetForPathLM,
    XLNetPathTokenizer,
)

from data_utils import glue_processors as processors
from data_utils import glue_output_modes as output_modes
from data_utils import glue_convert_examples_to_features
from print_hook import redirect_stdout
from utils import get_adamw, report_results, get_tokenizer, get_criterion, set_seed

from glue_lm_ft import load_and_cache_examples

try:
  from torch.utils.tensorboard import SummaryWriter
except ImportError:
  from tensorboardX import SummaryWriter

MODEL_CLASSES = {
    'bertpath-lmnsp': (BertConfig, BertForPreTraining, BERTPathTokenizer),
    'bertpath-mlmnspsp': (BertConfig, BertForPathLM, BERTPathTokenizer),
    'xlnetpath-clm': (XLNetConfig, XLNetLMHeadModel, XLNetPathTokenizer),
    'xlnetpath-clmnsp': (XLNetConfig, XLNetForPathLM, XLNetPathTokenizer),
    # 'xlnetpath-sp': (XLNetConfig, XLNetLMHeadModel, XLNetPathTokenizer),
    # 'xlnetpath-nsp': (XLNetConfig, XLNetLMHeadModel, XLNetPathTokenizer),
}


def load_and_cache_examples(args, task, subtask, tokenizer, evaluate=False, return_features=False, load_id=False,
                            load_element_id=False):
    '''
    https://github.com/huggingface/transformers/blob/master/examples/run_glue.py
    '''

    if (hasattr(args, 'local_rank') and args.local_rank not in [-1, 0]) and not evaluate:
            torch.distributed.barrier()    # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    # genre = args.tar_genre if evaluate else args.src_genre
    cached_features_file = "cached_{}_{}_{}_{}".format(
        "dev" if evaluate else "train",
        list(filter(None, args.model_name_or_path.split("/"))).pop(),
        str(args.max_seq_length),
        str(subtask) if subtask else str(task),  # important! new for subtask
    )
    cached_features_file = os.path.join(args.data_dir, cached_features_file)
    if os.path.exists(cached_features_file) and (not hasattr(args, 'overwrite_cache') or not args.overwrite_cache):
        print("Loading features from cached file %s" % cached_features_file)
        features = torch.load(cached_features_file)
    else:
        print("Creating features from dataset file at %s" % args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(args.data_dir) if evaluate else
            processor.get_train_examples(args.data_dir)
        )
        examples = examples[subtask]
        print('subtask', subtask)
        print('examples_', subtask, len(examples))
        features = glue_convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(args.model_type in ["xlnet"]),    # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            load_id=load_id,
            load_element_id=load_element_id
        )
        # print('features lml', features)
        if not hasattr(args, 'local_rank') or args.local_rank in [-1, 0]:
            print("Saving features into cached file %s" % cached_features_file)
            torch.save(features, cached_features_file)

    if (hasattr(args, 'local_rank') and args.local_rank not in [-1, 0]) and not evaluate:
        torch.distributed.barrier()    # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if return_features:
        return features

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if load_element_id:
        all_element_type_ids = torch.tensor([f.element_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    if load_element_id:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_element_type_ids, all_labels)
    else:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

def load_path_dataset_nsp(args, task, subtask, tokenizer, mlm=True, evaluate=False, replica=1, load_id=True,
                          load_element_id=True):
    '''
    add next_sentence_label
    :param args:
    :param task:
    :param tokenizer:
    :param evaluate:
    :param replica:
    :return:
    '''
    # load glue data (sentence pairs), tokenizer.encode_plus()
    features = load_and_cache_examples(args, task, subtask, tokenizer, evaluate, return_features=True, load_id=load_id,
                                       load_element_id=load_element_id)
    # in: input_ids, attention_mask, token_type_ids, label
    # out: input_ids, attention_mask, token_type_ids, masked_lm_labels
    all_input_ids = []
    all_attention_masks = []
    all_token_type_ids = []
    if load_element_id:
        all_element_type_ids = []
    all_next_sentence_label = []
    if load_id:
        all_example_id = []
    if mlm:
        all_masked_lm_labels = []
        masked_lm_prob = 0.15
        max_predictions_per_seq = 20
        rng = random.Random(args.seed)
        MASK_id = tokenizer.mask_token_id
    for (ft_index, feature) in enumerate(features):
        for _ in range(replica):
            next_sentence_label = feature.label
            if load_id:
                example_id = feature.example_id

            if not mlm:
                input_ids = feature.input_ids  # [CLS] A [SEP] B [SEP]
            else:
                init_ids = feature.input_ids  # [CLS] A [SEP] B [SEP]
                input_len = sum(feature.attention_mask)
                sep1_index = input_len - sum(feature.token_type_ids) - 1

                masked_lm_labels = [-100] * args.max_seq_length

                # sep1
                cand_indexes = list(range(1, sep1_index)) + list(range(sep1_index + 1, input_len - 1))
                rng.shuffle(cand_indexes)
                len_cand = len(cand_indexes)

                input_ids = copy.copy(init_ids)
                num_to_predict = min(max_predictions_per_seq,
                               max(1, int(round(input_len * masked_lm_prob))))

                masked_lms_pos = []
                covered_indexes = set()
                for index in cand_indexes:
                    if len(masked_lms_pos) >= num_to_predict:
                        break
                    if index in covered_indexes:
                        continue
                    covered_indexes.add(index)

                    masked_token = None
                    if rng.random() < 0.8:
                        masked_token = MASK_id
                    else:
                        if rng.random() < 0.5:
                            masked_token = init_ids[index]
                        else:
                            masked_token = init_ids[cand_indexes[rng.randint(0, len_cand - 1)]]

                    masked_lm_labels[index] = init_ids[index]
                    input_ids[index] = masked_token
                    masked_lms_pos.append(index)

            assert len(input_ids) == args.max_seq_length
            if mlm:
                assert len(masked_lm_labels) == args.max_seq_length
            assert len(feature.attention_mask) == args.max_seq_length
            assert len(feature.token_type_ids) == args.max_seq_length
            if load_element_id:
                assert len(feature.element_type_ids) == args.max_seq_length

            if ft_index < 0:
                print("*** Example ***")
                if load_id:
                    print(" example_id: %s" % " ".join(example_id))
                print(" tokens: %s" % " ".join([str(x) for x in tokenizer.convert_ids_to_tokens(feature.input_ids)]))
                if mlm:
                    print(" init_ids: %s" % " ".join([str(x) for x in init_ids]))
                    print(' masked tokens: %s' % ' '.join([str(x) for x in tokenizer.convert_ids_to_tokens(input_ids)]))
                print(" input_ids: %s" % " ".join([str(x) for x in input_ids]))
                print(" attention_mask: %s" % " ".join([str(x) for x in feature.attention_mask]))
                print(" token_type_ids: %s" % " ".join([str(x) for x in feature.token_type_ids]))
                if load_element_id:
                    print(" element_type_ids: %s" % " ".join([str(x) for x in feature.element_type_ids]))
                if mlm:
                    print(" masked_lm_labels: %s" % " ".join([str(x) for x in masked_lm_labels]))
                print(" next_sentence_label: %s" % " ".join([str(x) for x in next_sentence_label]))

            all_input_ids.append(input_ids)
            all_attention_masks.append(feature.attention_mask)
            all_token_type_ids.append(feature.token_type_ids)
            if load_element_id:
                all_element_type_ids.append(feature.element_type_ids)
            if mlm:
                all_masked_lm_labels.append(masked_lm_labels)
            all_next_sentence_label.append(next_sentence_label)
            if load_id:
                all_example_id.append(int(example_id))

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_masks = torch.tensor(all_attention_masks, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    if load_element_id:
        all_element_type_ids = torch.tensor(all_element_type_ids, dtype=torch.long)
    if mlm:
        all_masked_lm_labels = torch.tensor(all_masked_lm_labels, dtype=torch.long)
    all_next_sentence_label = torch.tensor(all_next_sentence_label, dtype=torch.long)
    if load_id:
        # print(all_example_id)
        all_example_id = torch.tensor(all_example_id, dtype=torch.long)

    # if load_id:
    #     if load_element_id:
    #         dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_element_type_ids,
    #                                 all_masked_lm_labels, all_next_sentence_label, all_example_id)
    #     else:
    #         dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids,
    #                                 all_masked_lm_labels, all_next_sentence_label, all_example_id)
    # else:
    #     if load_element_id:
    #         dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_element_type_ids,
    #                                 all_masked_lm_labels, all_next_sentence_label)
    #     else:
    #         dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids,
    #                                 all_masked_lm_labels, all_next_sentence_label)
    if not load_id:
        all_example_id = all_token_type_ids
    if not load_element_id:
        all_element_type_ids = all_token_type_ids
    if not mlm:
        all_masked_lm_labels = all_token_type_ids # can not use None, TensorDataset do not take None as input. Use all_token_type_ids to occupy it.
    dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_element_type_ids,
                            all_masked_lm_labels, all_next_sentence_label, all_example_id)

    return dataset


def load_path_dataset_sp(args, task, subtask, tokenizer, evaluate=False, replica=1, load_id=False,
                         load_element_id=True):
    # load glue data (sentence pairs), tokenizer.encode_plus()
    features = load_and_cache_examples(args, task, subtask, tokenizer, evaluate, return_features=True,
                                       load_id=load_id,
                                       load_element_id=load_element_id)
    # in: input_ids, attention_mask, token_type_ids, label
    # out: input_ids, attention_mask, token_type_ids, masked_lm_labels
    all_input_ids = []
    all_attention_masks = []
    all_token_type_ids = []
    if load_element_id:
        all_element_type_ids = []
    # all_masked_lm_labels = []
    all_sentence_label = []
    if load_id:
        all_example_id = []
    # masked_lm_prob = 0.15
    # max_predictions_per_seq = 20
    # rng = random.Random(args.seed)
    # MASK_id = tokenizer.mask_token_id
    for (ft_index, feature) in enumerate(features):
        for _ in range(replica):
            input_ids = feature.input_ids  # [CLS] A [SEP] B [SEP]
            # input_len = sum(feature.attention_mask)
            # sep1_index = input_len - sum(feature.token_type_ids) - 1
            sentence_label = feature.label
            if load_id:
                example_id = feature.example_id

            # masked_lm_labels = [-100] * args.max_seq_length
            #
            # # sep1
            # cand_indexes = list(range(1, sep1_index)) + list(range(sep1_index + 1, input_len - 1))
            # rng.shuffle(cand_indexes)
            # len_cand = len(cand_indexes)
            #
            # input_ids = copy.copy(init_ids)
            # num_to_predict = min(max_predictions_per_seq,
            #                max(1, int(round(input_len * masked_lm_prob))))
            #
            # masked_lms_pos = []
            # covered_indexes = set()
            # for index in cand_indexes:
            #     if len(masked_lms_pos) >= num_to_predict:
            #         break
            #     if index in covered_indexes:
            #         continue
            #     covered_indexes.add(index)
            #
            #     masked_token = None
            #     if rng.random() < 0.8:
            #         masked_token = MASK_id
            #     else:
            #         if rng.random() < 0.5:
            #             masked_token = init_ids[index]
            #         else:
            #             masked_token = init_ids[cand_indexes[rng.randint(0, len_cand - 1)]]
            #
            #     masked_lm_labels[index] = init_ids[index]
            #     input_ids[index] = masked_token
            #     masked_lms_pos.append(index)

            assert len(input_ids) == args.max_seq_length
            # assert len(masked_lm_labels) == args.max_seq_length
            assert len(feature.attention_mask) == args.max_seq_length
            assert len(feature.token_type_ids) == args.max_seq_length
            if load_element_id:
                assert len(feature.element_type_ids) == args.max_seq_length

            if ft_index < 0:
                print("*** Example ***")
                if load_id:
                    print(" example_id: %s" % " ".join(example_id))
                print(" tokens: %s" % " ".join([str(x) for x in tokenizer.convert_ids_to_tokens(feature.input_ids)]))
                # print(" init_ids: %s" % " ".join([str(x) for x in init_ids]))
                # print(' masked tokens: %s' % ' '.join([str(x) for x in tokenizer.convert_ids_to_tokens(input_ids)]))
                print(" input_ids: %s" % " ".join([str(x) for x in input_ids]))
                print(" attention_mask: %s" % " ".join([str(x) for x in feature.attention_mask]))
                print(" token_type_ids: %s" % " ".join([str(x) for x in feature.token_type_ids]))
                if load_element_id:
                    print(" element_type_ids: %s" % " ".join([str(x) for x in feature.element_type_ids]))
                # print(" masked_lm_labels: %s" % " ".join([str(x) for x in masked_lm_labels]))
                print(" sentence_label: %s" % " ".join([str(x) for x in sentence_label]))

            all_input_ids.append(input_ids)
            all_attention_masks.append(feature.attention_mask)
            all_token_type_ids.append(feature.token_type_ids)
            if load_element_id:
                all_element_type_ids.append(feature.element_type_ids)
            # all_masked_lm_labels.append(masked_lm_labels)
            all_sentence_label.append(sentence_label)
            if load_id:
                all_example_id.append(int(example_id))

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_masks = torch.tensor(all_attention_masks, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    if load_element_id:
        all_element_type_ids = torch.tensor(all_element_type_ids, dtype=torch.long)
    # all_masked_lm_labels = torch.tensor(all_masked_lm_labels, dtype=torch.long)
    all_sentence_label = torch.tensor(all_sentence_label, dtype=torch.long)
    if load_id:
        # print(all_example_id)
        all_example_id = torch.tensor(all_example_id, dtype=torch.long)

    # if load_id:
    #     if load_element_id:
    #         dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_element_type_ids,
    #                                 # all_masked_lm_labels,
    #                                 all_sentence_label, all_example_id)
    #     else:
    #         dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids,
    #                                 # all_masked_lm_labels,
    #                                 all_sentence_label, all_example_id)
    # else:
    #     if load_element_id:
    #         dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_element_type_ids,
    #                                 # all_masked_lm_labels,
    #                                 all_sentence_label)
    #     else:
    #         dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids,
    #                                 # all_masked_lm_labels,
    #                                 all_sentence_label)
    if not load_id:
        all_example_id = all_token_type_ids # None can not be put in the TensorDataset
    if not load_element_id:
        all_element_type_ids = all_token_type_ids
    dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_element_type_ids,
                            all_sentence_label, all_example_id)

    return dataset


def load_and_cache_dataset(args, task, tokenizer, evaluate=False, replica=1, load_id=False,
                           load_element_id=True):
    if args.model_type == 'bertpath-lmnsp':
        if task == 'lmnsp':
            dataset_lmnsp = load_path_dataset_nsp(args, task, 'nsp', tokenizer, mlm=True,
                                                  evaluate=evaluate, replica=replica,
                                                  load_id=load_id,
                                                  load_element_id=load_element_id)
            return dataset_lmnsp
        else:
            print('[ERROR] No ', task, ' in model ', args.model_type)
            raise NotImplementedError
    elif args.model_type == 'bertpath-mlmnspsp':
        if task == 'mlmnspsp': #args.model_type.startswith('bertpath'):
            # generate MLM input
            dataset_nsp = load_path_dataset_nsp(args, task, 'nsp', tokenizer, mlm=True,
                                                evaluate=evaluate, replica=replica,
                                                load_id=load_id,
                                                load_element_id=load_element_id)
            dataset_sp = load_path_dataset_sp(args, task, 'sp', tokenizer,
                                              evaluate=evaluate, replica=replica,
                                              load_id=load_id,
                                              load_element_id=load_element_id)
            return dataset_nsp, dataset_sp
        elif task == 'spnsp': #args.model_type.startswith('bertpath'):
            # generate MLM input
            dataset_nsp = load_path_dataset_nsp(args, task, 'nsp', tokenizer, mlm=False,
                                                evaluate=evaluate, replica=replica,
                                                load_id=load_id,
                                                load_element_id=load_element_id)
            dataset_sp = load_path_dataset_sp(args, task, 'sp', tokenizer,
                                              evaluate=evaluate, replica=replica,
                                              load_id=load_id,
                                              load_element_id=load_element_id)
            return dataset_nsp, dataset_sp
        elif task == 'sp':
            dataset_sp = load_path_dataset_sp(args, task, 'sp', tokenizer,
                                              evaluate=evaluate, replica=replica,
                                              load_id=load_id,
                                              load_element_id=load_element_id)
            return dataset_sp
        elif task == 'mlmnsp':
            dataset_nsp = load_path_dataset_nsp(args, task, 'nsp', tokenizer, mlm=True,
                                                   evaluate=evaluate, replica=replica,
                                                   load_id=load_id,
                                                   load_element_id=load_element_id)
            return dataset_nsp
        elif task == 'nsp':
            dataset_nsp = load_path_dataset_nsp(args, task, 'nsp', tokenizer, mlm=False,
                                                evaluate=evaluate, replica=replica,
                                                load_id=load_id,
                                                load_element_id=load_element_id)
            return dataset_nsp
        else:
            print('[ERROR] No ', task, ' in model ', args.model_type)
            raise NotImplementedError
    elif args.model_type == 'xlnetpath-clm':
        if task == 'clm':
            # generate CLM input
            dataset_lm = load_path_dataset_sp(args, task, 'lm', tokenizer, evaluate=evaluate, replica=replica, load_id=load_id,
                                         load_element_id=load_element_id)
            return dataset_lm
        else:
            print('[ERROR] No ', task, ' in model ', args.model_type)
            raise NotImplementedError
    elif args.model_type == 'xlnetpath-clmnsp':
        if task == 'clmnsp':
            dataset_nsp = load_path_dataset_nsp(args, task, 'nsp', tokenizer, mlm=False,
                                                evaluate=evaluate, replica=replica,
                                                load_id=load_id,
                                                load_element_id=load_element_id)
            dataset_lm = load_path_dataset_sp(args, task, 'lm', tokenizer, evaluate=evaluate,
                                              replica=replica,
                                              load_id=load_id,
                                              load_element_id=load_element_id)
            return dataset_nsp, dataset_lm
        else:
            print('[ERROR] No ', task, ' in model ', args.model_type)
            raise NotImplementedError
    else:
        raise NotImplementedError


def get_loss(model_type, model, criterion, batch, evaluate=False, load_id=False, load_element_id=True, subtask='sp'):
    # print(batch)
    if load_id:
        example_id_list = batch[-1].view(-1)  # .tolist()
        example_id_list = [element.item() for element in example_id_list.flatten()]

    if model_type == 'bertpath-lmnsp':
        input_ids, attention_masks, token_type_ids, element_type_ids, \
            mlm_label, nsp_label, example_id = batch
        if not evaluate:
            model_output = model(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids,
                                     element_type_ids=element_type_ids,
                                     masked_lm_labels=mlm_label, next_sentence_label=nsp_label)
            # print('model_output', model_output)
            loss, prediction_scores, seq_relationship_score = model_output[:3]
            masked_lm_loss = model_output[-2].item()
            next_sentence_loss = model_output[-1].item()
            _, predicted_nsp = torch.max(seq_relationship_score.view(-1, 2), 1)
            correct_nsp = (predicted_nsp == nsp_label.view(-1)).sum().item()
            acc_nsp = correct_nsp / float(nsp_label.size(0))
            return loss, masked_lm_loss, next_sentence_loss, acc_nsp
        else:
            ## ??? Why -> only evaluate the masked_lm_labels performance (ignore lm_labels prediction)
            model_output = model(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids,
                                 element_type_ids=element_type_ids)
            prediction_scores, seq_relationship_score = model_output[:2]
            _, predicted_nsp = torch.max(seq_relationship_score.view(-1, 2), 1)
            correct_nsp = (predicted_nsp == nsp_label.view(-1)).sum().item()
            # print('eval correct_nsp', correct_nsp)
            masked_lm_loss = criterion(prediction_scores.view(-1, model.config.vocab_size), mlm_label.view(-1))
            next_sentence_loss = criterion(seq_relationship_score.view(-1, 2), nsp_label.view(-1))
            loss = masked_lm_loss + next_sentence_loss
            if load_id:
                seq_relationship_score_list = seq_relationship_score.view(-1, 2) #[:, 1] #.tolist()
                seq_relationship_score_list = [(element1.item(), element2.item()) for element1, element2 in seq_relationship_score_list] #.flatten()
                return loss, masked_lm_loss, next_sentence_loss, correct_nsp, example_id_list, seq_relationship_score_list
            return loss, masked_lm_loss, next_sentence_loss, correct_nsp
    elif model_type == 'bertpath-mlmnspsp':

        if subtask == 'sp':
            input_ids, attention_masks, token_type_ids, element_type_ids, \
                sequence_label, example_id = batch
            if not load_element_id:
                element_type_ids = None
            if evaluate:
                sequence_label_input = None
            else:
                sequence_label_input = sequence_label
            model_output = model(input_ids=input_ids, attention_mask=attention_masks,
                                 token_type_ids=token_type_ids,
                                 element_type_ids=element_type_ids,
                                 sequence_label=sequence_label_input,
                                 lm=False,
                                 nsp=False,
                                 sp=True,
                                 sametoken=False)
            # read model_output
            seq_loss, losses, scores = model_output[:3]
            if not evaluate:
                return seq_loss
            else:
                seq_score = scores['sp']
                _, predicted_sp = torch.max(seq_score.view(-1, 2), 1)
                correct_sp = (predicted_sp == sequence_label.view(-1)).sum().item()
                # print('seq_score', seq_score)
                seq_loss = criterion(seq_score.view(-1, 2), sequence_label.view(-1))
                if load_id:
                    seq_score_list = seq_score.view(-1, 2)  # [:, 1] #.tolist()
                    seq_score_list = [(element1.item(), element2.item()) for element1, element2 in
                                                   seq_score_list]  # .flatten()
                    return seq_loss, correct_sp, example_id_list, seq_score_list
                return seq_loss, correct_sp
        elif subtask == 'mlmnsp':
            input_ids, attention_masks, token_type_ids, element_type_ids, \
                mlm_label, nsp_label, example_id = batch
            if evaluate:
                mlm_label_input = None
                nsp_label_input = None
            else:
                mlm_label_input = mlm_label
                nsp_label_input = nsp_label
            model_output = model(input_ids=input_ids, attention_mask=attention_masks,
                                 token_type_ids=token_type_ids,
                                 element_type_ids=element_type_ids,
                                 masked_lm_labels=mlm_label_input,
                                 next_sentence_label=nsp_label_input,
                                 lm=True,
                                 nsp=True,
                                 sp=False,
                                 sametoken=False)
            total_loss_mlmnsp, losses, scores = model_output[:3]
            if not evaluate:
                masked_lm_loss = losses['mlm']
                next_sentence_loss = losses['nsp']
                return total_loss_mlmnsp, masked_lm_loss, next_sentence_loss
            else:
                seq_relationship_score = scores['nsp']
                prediction_scores = scores['mlm']
                _, predicted_nsp = torch.max(seq_relationship_score.view(-1, 2), 1)
                correct_nsp = (predicted_nsp == nsp_label.view(-1)).sum().item()
                masked_lm_loss = criterion(prediction_scores.view(-1, model.config.vocab_size), mlm_label.view(-1))
                next_sentence_loss = criterion(seq_relationship_score.view(-1, 2), nsp_label.view(-1))
                loss = masked_lm_loss + next_sentence_loss
                # print('seq_relationship_score', seq_relationship_score)
                # print('nsp_label', nsp_label)
                if load_id:
                    seq_relationship_score_list = seq_relationship_score.view(-1, 2)  # [:, 1] #.tolist()
                    seq_relationship_score_list = [(element1.item(), element2.item()) for element1, element2 in
                                                   seq_relationship_score_list]  # .flatten()
                    return loss, masked_lm_loss, next_sentence_loss, correct_nsp, example_id_list, seq_relationship_score_list
                return loss, masked_lm_loss, next_sentence_loss, correct_nsp
    elif model_type == 'xlnetpath-clm':
        input_ids, attention_masks, token_type_ids, element_type_ids, \
            sentence_label, example_id = batch
        if not load_element_id:
            element_type_ids = None
        if not evaluate:
            # input_ids = batch[0]
            # print('input_ids', input_ids)
            # input_permutation = model.prepare_inputs_for_generation(input_ids, None, use_cache=False)
            # input_ids = input_permutation["input_ids"]
            # perm_mask = input_permutation["perm_mask"]
            # target_mapping = input_permutation["target_mapping"]
            # use_cache = input_permutation["use_cache"]
            # print('input_ids\'', input_ids)
            # print('perm_mask', perm_mask)
            # print('target_mapping', target_mapping)
            model_output = model(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids,
                                 element_type_ids=element_type_ids,
                                 # perm_mask=perm_mask, target_mapping=target_mapping, use_cache=use_cache
                                 labels=input_ids)
            loss, prediction_scores = model_output[:2]
            return loss, prediction_scores
        else:
            # no labels are input

            model_output = model(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids,
                                 element_type_ids=element_type_ids)
            logits = model_output[0]
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.view(-1, logits.size(-1)), input_ids.view(-1)).view(input_ids.size(0), -1)
            loss = torch.sum(loss, dim=1)
            loss_list = [element.item() for element in loss.flatten()]
            loss = torch.sum(loss, dim=0)
            # print(loss.size())
            # prediction_scores = torch.index_select(prediction_scores.view(-1, prediction_scores.size(-1)), dim=1, index=input_ids.view(-1))
            # print('prediction_scores1', prediction_scores.size())
            # print('input_ids', input_ids.unsqueeze(2).size())
            # input_ids_select = input_ids.unsqueeze(2)
            # # prediction_scores, _ = torch.max(prediction_scores, dim=2)
            # prediction_scores = torch.index_select(prediction_scores, 2, input_ids)
            # # prediction_scores = prediction_scores[input_ids_select[0]][input_ids_select[1]][input_ids_select[2]]
            # print('prediction_scores2', prediction_scores.size())
            # prediction_scores = torch.sum(prediction_scores, dim=1)
            # print('prediction_scores3', prediction_scores.size())
            if load_id:
                return loss, loss_list, example_id_list
            return loss, loss_list
    elif model_type == 'xlnetpath-clmnsp':
        if subtask == 'lm':
            input_ids, attention_masks, token_type_ids, element_type_ids, \
                sequence_label, example_id = batch
            if not load_element_id:
                element_type_ids = None
            if evaluate:
                labels_clm_input = None
            else:
                labels_clm_input = input_ids
            # print('model', model)
            model_output = model(input_ids=input_ids, attention_mask=attention_masks,
                                 token_type_ids=token_type_ids,
                                 element_type_ids=element_type_ids,
                                 labels_clm=labels_clm_input,
                                 lm=True,
                                 nsp=False)
            # read model_output
            total_loss, losses, scores = model_output[:3]
            if not evaluate:
                return losses['clm']
            else:
                logits = scores['clm']
                loss_fct = CrossEntropyLoss(reduction='none')
                loss = loss_fct(logits.view(-1, logits.size(-1)), input_ids.view(-1)).view(input_ids.size(0), -1)
                loss = torch.sum(loss, dim=1)
                loss_list = [element.item() for element in loss.flatten()]
                loss = torch.sum(loss, dim=0)
                if load_id:
                    return loss, loss_list, example_id_list
                return loss, loss_list
        elif subtask == 'nsp':
            input_ids, attention_masks, token_type_ids, element_type_ids, \
                mlm_label, nsp_label, example_id = batch
            if evaluate:
                nsp_label_input = None
            else:
                nsp_label_input = nsp_label
            model_output = model(input_ids=input_ids, attention_mask=attention_masks,
                                 token_type_ids=token_type_ids,
                                 element_type_ids=element_type_ids,
                                 labels_seq=nsp_label_input,
                                 lm=False,
                                 nsp=True)
            total_loss_mlmnsp, losses, scores = model_output[:3]
            if not evaluate:
                return losses['nsp']
            else:
                seq_relationship_score = scores['nsp']
                _, predicted_nsp = torch.max(seq_relationship_score.view(-1, 2), 1)
                correct_nsp = (predicted_nsp == nsp_label.view(-1)).sum().item()
                next_sentence_loss = criterion(seq_relationship_score.view(-1, 2), nsp_label.view(-1))
                if load_id:
                    seq_relationship_score_list = seq_relationship_score.view(-1, 2)  # [:, 1] #.tolist()
                    seq_relationship_score_list = [(element1.item(), element2.item()) for element1, element2 in
                                                   seq_relationship_score_list]  # .flatten()
                    return next_sentence_loss, correct_nsp, example_id_list, seq_relationship_score_list
                return next_sentence_loss, correct_nsp
    else:
        raise NotImplementedError


def train(args, train_dataset, model, criterion, tokenizer, load_id=False, load_element_id=True):
    tb_writer = SummaryWriter(args.output_dir)

    if args.model_type == 'bertpath-mlmnspsp':
        train_dataset_nsp, train_dataset_sp = train_dataset
        train_sampler = RandomSampler(train_dataset_nsp)
        train_dataloader = DataLoader(train_dataset_nsp, sampler=train_sampler,
                                      batch_size=args.train_batch_size)
        train_sampler_sp = RandomSampler(train_dataset_sp)
        train_dataloader_sp = DataLoader(train_dataset_sp, sampler=train_sampler_sp,
                                         batch_size=args.train_batch_size * 4)
    elif args.model_type == 'xlnetpath-clmnsp':
        train_dataset_nsp, train_dataset_lm = train_dataset
        train_sampler = RandomSampler(train_dataset_nsp)
        train_dataloader = DataLoader(train_dataset_nsp, sampler=train_sampler, batch_size=args.train_batch_size)
        train_sampler_lm = RandomSampler(train_dataset_lm)
        train_dataloader_lm = DataLoader(train_dataset_lm, sampler=train_sampler_lm,
                                         batch_size=args.train_batch_size * 2)
    else:
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // len(train_dataloader) + 1
    else:
        t_total = int(len(train_dataloader) * args.num_train_epochs)
        args.num_train_epochs = int(np.ceil(args.num_train_epochs))

    optimizer, scheduler = get_adamw(model, t_total, args.warmup_steps, args.learning_rate, weight_decay=args.weight_decay)

    train_desc = args.task_name
    print(f'***** Fine-tuning {args.model_name_or_path} {train_desc} *****')
    print(f'  Num examples = {len(train_dataset)}')
    print(f'  Num Epochs = {args.num_train_epochs}')
    print(f'  Train batch size = {args.train_batch_size}')
    print(f'  Total optimization steps = {t_total}')

    ckpt_steps = set([int(x) for x in np.linspace(0, t_total, args.num_ckpts + 1)[1:]])

    model.train()
    model.zero_grad()

    global_step = 0
    step_loss = []
    if args.model_type == 'bertpath-lmnsp':
        step_masked_lm_loss = []
        step_next_sentence_loss = []
        step_acc_nsp = []
    eval_results = []
    # scores = []

    pbar = tqdm(total=t_total, desc=f'train')
    set_seed(args)
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            # print('batch', batch)
            if args.model_type == 'bertpath-lmnsp':
                loss_output = get_loss(args.model_type, model, criterion, batch, evaluate=False,
                                       load_id=load_id, load_element_id=load_element_id)
                loss, masked_lm_loss, next_sentence_loss, acc_nsp = loss_output
                step_masked_lm_loss.append(masked_lm_loss)
                step_next_sentence_loss.append(next_sentence_loss)
                step_acc_nsp.append(acc_nsp)
            elif args.model_type == 'bertpath-mlmnspsp':
                # task 1: mlm nsp
                loss_output = get_loss(args.model_type, model, criterion, batch, evaluate=False,
                                       load_id=load_id, load_element_id=load_element_id,
                                       subtask='mlmnsp')
                total_loss_mlmnsp, masked_lm_loss, next_sentence_loss = loss_output
                # task 2:
                batch_sp = next(iter(train_dataloader_sp))
                batch_sp = tuple(t.to(args.device) for t in batch_sp)
                # print('batch_sp', batch_sp)
                loss_output = get_loss(args.model_type, model, criterion, batch_sp, evaluate=False,
                                       load_id=load_id, load_element_id=load_element_id,
                                       subtask='sp')
                total_loss_sp = loss_output
                loss = total_loss_mlmnsp + total_loss_sp
            elif args.model_type == 'xlnetpath-clm':
                loss_output = get_loss(args.model_type, model, criterion, batch, evaluate=False,
                                       load_id=load_id, load_element_id=load_element_id)
                loss, lm_logits = loss_output
            elif args.model_type == 'xlnetpath-clmnsp':
                # task 1: lm
                batch_lm = next(iter(train_dataloader_lm))
                batch_lm = tuple(t.to(args.device) for t in batch_lm)
                total_loss_lm = get_loss(args.model_type, model, criterion, batch_lm, evaluate=False,
                                       load_id=load_id, load_element_id=load_element_id,
                                       subtask='lm')
                # task 2: nsp
                total_loss_nsp = get_loss(args.model_type, model, criterion, batch, evaluate=False,
                                       load_id=load_id, load_element_id=load_element_id,
                                       subtask='nsp')
                loss = total_loss_lm + total_loss_nsp
            else:
                return NotImplementedError
            step_loss.append(loss.item())
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            # print('optimize loss')

            global_step += 1
            pbar.update(1)
            pbar.set_description_str(
                f'train: {train_desc} (loss = {step_loss[-1]:.2f}, lr = {scheduler.get_lr()[0]:.2g})')

            if global_step in ckpt_steps:
                ckpt_path = os.path.join(args.output_dir, f'step_{global_step}.bin')
                torch.save(model, ckpt_path)

                if args.do_eval:

                    if args.model_type == 'bertpath-lmnsp':
                        step_eval_results = evaluate(args, model, criterion, tokenizer,
                                                     load_id=load_id, load_element_id=load_element_id)
                        train_loss = np.mean(step_loss)
                        example_id = step_eval_results[args.task_name+'_example_id']
                        eval_loss = step_eval_results[args.task_name+'_loss']
                        eval_loss_mm = step_eval_results[args.task_name+'_masked_lm']
                        eval_loss_nsp = step_eval_results[args.task_name+'_next_sentence']

                        eval_acc_nsp = step_eval_results[args.task_name+'_next_sentence_acc']
                        score_seq_relationship = step_eval_results[args.task_name+'_seq_relationship_score']
                        # scores.append(dict(zip(example_id, score_seq_relationship)))
                        json.dump(dict(zip(example_id, score_seq_relationship)),
                                  open(os.path.join(args.output_dir, f"eval_scores_step_{global_step}.json"), 'w'),
                                  ensure_ascii=False,
                                  indent=2)
                        eval_results.append([global_step, train_loss, eval_loss, eval_loss_mm, eval_loss_nsp, eval_acc_nsp])
                        print(
                            f'\nSaving model checkpoint to {ckpt_path}, avg_loss = {train_loss:.2f}, eval_loss = {eval_loss:.2f}, eval_mm_loss = {eval_loss_mm:.2f}, '
                            f'eval_loss_nsp = {eval_loss_nsp:.2f}, eval_acc_nsp = {eval_acc_nsp:.2f}\n')
                    # elif args.model_type == 'bertpath-mlmnspsp':
                    #     step_eval_results = evaluate(args, model, criterion, tokenizer,
                    #                                  load_id=load_id, load_element_id=load_element_id,
                    #                                  eval_subtask='mlmnspsp')
                    #     train_loss = np.mean(step_loss)
                    #     example_id_nsp = step_eval_results[args.task_name + '_example_id_nsp']
                    #     example_id_sp = step_eval_results[args.task_name + '_example_id_sp']
                    #     eval_loss = step_eval_results[args.task_name + '_loss']
                    #     eval_loss_mm = step_eval_results[args.task_name + '_masked_lm']
                    #     eval_loss_nsp = step_eval_results[args.task_name + '_next_sentence']
                    #     eval_loss_sp = step_eval_results[args.task_name + '_sentence']
                    #
                    #     eval_acc_nsp = step_eval_results[args.task_name + '_next_sentence_acc']
                    #     eval_acc_sp = step_eval_results[args.task_name + '_sentence_acc']
                    #     score_seq_relationship = step_eval_results[args.task_name + '_seq_relationship_score']
                    #     score_seq = step_eval_results[args.task_name + '_seq_score']
                    #     # scores.append(dict(zip(example_id, score_seq_relationship)))
                    #     json.dump(dict(zip(example_id_nsp, score_seq_relationship)), # [:, 1]
                    #               open(os.path.join(args.output_dir, f"eval_scores_nsp_step_{global_step}.json"), 'w'),
                    #               ensure_ascii=False,
                    #               indent=2)
                    #     json.dump(dict(zip(example_id_sp, score_seq)),
                    #               open(os.path.join(args.output_dir, f"eval_scores_sp_step_{global_step}.json"), 'w'),
                    #               ensure_ascii=False,
                    #               indent=2)
                    #     eval_results.append(
                    #         [global_step, train_loss, eval_loss, eval_loss_mm, eval_loss_nsp, eval_loss_sp, eval_acc_nsp, eval_acc_sp])
                    #     print(
                    #         f'\nSaving model checkpoint to {ckpt_path}, avg_loss = {train_loss:.2f}, eval_mm_loss = {eval_loss_mm:.2f}, '
                    #         f'eval_loss_nsp = {eval_loss_nsp:.2f}, eval_loss_sp = {eval_loss_sp:.2f}, '
                    #         f'eval_acc_nsp = {eval_acc_nsp:.2f}, eval_acc_sp = {eval_acc_sp:.2f}\n')

                    elif args.model_type == 'xlnetpath-clm':
                        step_eval_results = evaluate(args, model, criterion, tokenizer,
                                                     load_id=load_id, load_element_id=load_element_id)
                        train_loss = np.mean(step_loss)
                        example_id = step_eval_results[args.task_name+'_example_id']
                        eval_loss = step_eval_results[args.task_name+'_loss']
                        eval_loss_list = step_eval_results[args.task_name + '_loss_score']
                        eval_results.append(
                            [global_step, train_loss, eval_loss])
                        json.dump(dict(zip(example_id, eval_loss_list)),
                                  open(os.path.join(args.output_dir, f"test_loss_step_{global_step}.json"), 'w'),
                                  ensure_ascii=False,
                                  indent=2)
                        print(
                            f'\nSaving model checkpoint to {ckpt_path}, avg_loss = {train_loss:.2f}, eval_loss = {eval_loss:.2f}')

                    elif args.model_type == 'xlnetpath-clmnsp':
                        step_eval_results = evaluate(args, model, criterion, tokenizer,
                                                     load_id=load_id, load_element_id=load_element_id,
                                                     eval_subtask='clmnsp')
                        train_loss = np.mean(step_loss)
                        example_id_nsp = step_eval_results[args.task_name + '_example_id_nsp']
                        example_id_lm = step_eval_results[args.task_name + '_example_id_lm']
                        eval_loss = step_eval_results[args.task_name + '_loss']
                        eval_loss_nsp = step_eval_results[args.task_name + '_next_sentence']
                        eval_loss_lm = step_eval_results[args.task_name + '_lm']

                        eval_acc_nsp = step_eval_results[args.task_name + '_next_sentence_acc']
                        score_seq_relationship = step_eval_results[args.task_name + '_seq_relationship_score']
                        score_lm = step_eval_results[args.task_name + '_loss_score']
                        json.dump(dict(zip(example_id_nsp, score_seq_relationship)),  # [:, 1]
                                  open(os.path.join(args.output_dir, f"test_scores_nsp_step_{global_step}.json"), 'w'),
                                  ensure_ascii=False,
                                  indent=2)
                        json.dump(dict(zip(example_id_lm, score_lm)),
                                  open(os.path.join(args.output_dir, f"test_loss_lm_step_{global_step}.json"), 'w'),
                                  ensure_ascii=False,
                                  indent=2)
                        eval_results.append(
                            [global_step, train_loss, eval_loss, eval_loss_lm, eval_loss_nsp,
                             eval_acc_nsp])
                        print(
                            f'\nSaving model checkpoint to {ckpt_path}, avg_loss = {train_loss:.2f}, eval_lm_loss = {eval_loss_lm:.2f}, '
                            f'eval_loss_nsp = {eval_loss_nsp:.2f}, '
                            f'eval_acc_nsp = {eval_acc_nsp:.2f}\n')
                    else:
                        pass
                else:
                    print(f'\nSaving model checkpoint to {ckpt_path}\n')

            if global_step % args.logging_steps == 0:
                tb_writer.add_scalar('learning_rate', scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar('loss', np.mean(step_loss), global_step)
                step_loss = []
                if args.model_type == 'bertpath-lmnsp':
                    tb_writer.add_scalar('masked_lm_loss', np.mean(step_masked_lm_loss), global_step)
                    tb_writer.add_scalar('next_sentence_loss', np.mean(step_next_sentence_loss), global_step)
                    tb_writer.add_scalar('acc_nsp', np.mean(step_acc_nsp), global_step)
                    step_masked_lm_loss = []
                    step_next_sentence_loss = []
                    step_acc_nsp = []

            if global_step == args.max_steps:
                pbar.close()
                break

    if args.do_eval:
        if args.model_type == 'bertpath-lmnsp':
            header = ['step', 'avg_loss', 'eval_loss', 'eval_mm_loss', 'eval_nsp_loss', 'eval_nsp_acc']
        else:
            header = ['step', 'avg_loss', 'eval_loss']
        best_results = report_results(header, eval_results, 2)
        best_step = best_results[0]
        # best_scores = scores[best_step]
        # json.dump(best_scores, open(os.path.join(args.output_dir, 'best_simscores.json'), 'w'), ensure_ascii=False,
        #           indent=2)
        # print('Saving best similarity info in %s' % os.path.join(args.output_dir, 'best_simscores.json'))
        print(f'best_ckpt = {os.path.join(args.output_dir, f"step_{best_step}.bin")}\n')


def evaluate(args, model, criterion, tokenizer, eval_subtask=None, load_id=False, load_element_id=True):
    if eval_subtask:
        eval_task_names = (eval_subtask, )
    else:
        eval_task_names = (args.task_name,)


    model.eval()
    results = {}
    for eval_task in eval_task_names:
        eval_dataset = load_and_cache_dataset(args, eval_task, tokenizer, evaluate=True, load_id=load_id,
                                              load_element_id=load_element_id)
        if args.model_type == 'bertpath-mlmnspsp':
            eval_dataset_nsp, eval_dataset_sp = eval_dataset
            eval_sampler = SequentialSampler(eval_dataset_nsp)
            eval_dataloader = DataLoader(eval_dataset_nsp, sampler=eval_sampler, batch_size=args.eval_batch_size)
            eval_sampler_sp = SequentialSampler(eval_dataset_sp)
            eval_dataloader_sp = DataLoader(eval_dataset_sp, sampler=eval_sampler_sp,
                                             batch_size=args.eval_batch_size * 4)
        elif args.model_type == 'xlnetpath-clmnsp':
            eval_dataset_nsp, eval_dataset_lm = eval_dataset
            eval_sampler = SequentialSampler(eval_dataset_nsp)
            eval_dataloader = DataLoader(eval_dataset_nsp, sampler=eval_sampler, batch_size=args.eval_batch_size)
            eval_sampler_lm = SequentialSampler(eval_dataset_lm)
            eval_dataloader_lm = DataLoader(eval_dataset_lm, sampler=eval_sampler_lm,
                                             batch_size=args.eval_batch_size * 4)
        else:
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        eval_desc = eval_task

        if args.model_type == 'bertpath-lmnsp':
            eval_loss = 0
            eval_masked_lm_loss = 0
            num_masked_lm_elements = 0
            eval_next_sentence_loss = 0
            num_next_sentence_elements = 0
            eval_corrected_nsp = 0
            if load_id:
                example_id_all = list()
                seq_relationship_score_all = list()
            with torch.no_grad():
                for batch in tqdm(eval_dataloader, desc=f'eval: {eval_desc}', leave=False):
                    # in: input_ids, attention_mask, token_type_ids, masked_lm_labels
                    batch = tuple(t.to(args.device) for t in batch)
                    input_ids, attention_masks, token_type_ids, element_type_ids, \
                        mlm_label, nsp_label, example_id = batch
                    eval_result_batch = get_loss(args.model_type, model, criterion, batch, evaluate=True, load_id=load_id, load_element_id=load_element_id,)
                    if load_id:
                        loss, masked_lm_loss, next_sentence_loss, corrected_nsp, example_id, seq_relationship_score = eval_result_batch
                        example_id_all.extend(example_id)
                        seq_relationship_score_all.extend(seq_relationship_score)
                        # print(example_id)
                    else:
                        loss, masked_lm_loss, next_sentence_loss, corrected_nsp = eval_result_batch
                    eval_loss += loss.item()
                    eval_masked_lm_loss += masked_lm_loss.item()
                    eval_next_sentence_loss += next_sentence_loss.item()
                    num_masked_lm_elements += (mlm_label.detach().cpu().numpy().flatten() != criterion.ignore_index).sum()
                    num_next_sentence_elements += nsp_label.size(0)
                    eval_corrected_nsp += corrected_nsp

                eval_masked_lm_loss /= num_masked_lm_elements

                eval_next_sentence_loss /= num_next_sentence_elements
                eval_next_sentence_acc = float(eval_corrected_nsp) / num_next_sentence_elements

            results[eval_task + '_loss'] = eval_loss
            results[eval_task + '_masked_lm'] = eval_masked_lm_loss
            results[eval_task + '_next_sentence'] = eval_next_sentence_loss
            results[eval_task + '_next_sentence_acc'] = eval_next_sentence_acc
            if load_id:
                results[eval_task + '_example_id'] = example_id_all
                results[eval_task + '_seq_relationship_score'] = seq_relationship_score_all
        elif args.model_type == 'bertpath-mlmnspsp':
            eval_loss = 0
            if 'mlmnsp' in eval_task:
                eval_masked_lm_loss = 0
                num_masked_lm_elements = 0
                eval_next_sentence_loss = 0
                num_next_sentence_elements = 0
                eval_corrected_nsp = 0
            if 'sp' in eval_task:
                eval_sentence_loss = 0
                num_sentence_elements = 0
                eval_corrected_sp = 0
            if load_id:
                example_id_sp_all = list()
                example_id_nsp_all = list()
                seq_relationship_score_all = list()
                seq_score_all = list()

            # maybe multiple tasks, so initialize to 0, and then add loss of each task
            results[eval_task + '_loss'] = 0

            if 'mlmnsp' in eval_task:
                with torch.no_grad():
                    for batch in tqdm(eval_dataloader, desc=f'eval: {eval_desc}', leave=False):
                        # in: input_ids, attention_mask, token_type_ids, masked_lm_labels
                        batch = tuple(t.to(args.device) for t in batch)
                        # print('batch_eval', batch)
                        input_ids, attention_masks, token_type_ids, element_type_ids, \
                            mlm_label, nsp_label, example_id = batch
                        # print('input_ids_nsp', input_ids.size())
                        eval_result_batch = get_loss(args.model_type, model, criterion, batch, evaluate=True,
                                                     load_id=load_id, load_element_id=load_element_id,
                                                     subtask='mlmnsp')
                        if load_id:
                            loss, masked_lm_loss, next_sentence_loss, correct_nsp, example_id_list, seq_relationship_score_list = eval_result_batch
                            example_id_nsp_all.extend(example_id_list)
                            # print('example_id_list_nsp', len(example_id_list))
                            seq_relationship_score_all.extend(seq_relationship_score_list)
                            # print(example_id)
                        else:
                            loss, masked_lm_loss, next_sentence_loss, correct_nsp = eval_result_batch
                        eval_loss += loss.item()
                        eval_masked_lm_loss += masked_lm_loss.item()
                        eval_next_sentence_loss += next_sentence_loss.item()
                        num_masked_lm_elements += (
                                mlm_label.detach().cpu().numpy().flatten() != criterion.ignore_index).sum()
                        num_next_sentence_elements += nsp_label.size(0)
                        eval_corrected_nsp += correct_nsp

                    eval_masked_lm_loss /= num_masked_lm_elements
                    eval_next_sentence_loss /= num_next_sentence_elements
                    eval_next_sentence_acc = float(eval_corrected_nsp) / num_next_sentence_elements
                results[eval_task + '_loss'] += eval_loss
                results[eval_task + '_masked_lm'] = eval_masked_lm_loss
                results[eval_task + '_next_sentence'] = eval_next_sentence_loss
                results[eval_task + '_next_sentence_acc'] = eval_next_sentence_acc
                if load_id:
                    results[eval_task + '_example_id_nsp'] = example_id_nsp_all
                    # print('example_id_nsp_all', len(example_id_nsp_all))
                    results[eval_task + '_seq_relationship_score'] = seq_relationship_score_all
            if 'sp' in eval_task:
                with torch.no_grad():
                    for batch_sp in tqdm(eval_dataloader_sp, desc=f'eval: {eval_desc}', leave=False):
                        batch_sp = tuple(t.to(args.device) for t in batch_sp)
                        input_ids, attention_masks, token_type_ids, element_type_ids, \
                            sequence_label, example_id = batch_sp
                        # print('input_ids_sp', input_ids.size())
                        eval_result_batch = get_loss(args.model_type, model, criterion, batch_sp, evaluate=True,
                                               load_id=load_id, load_element_id=load_element_id,
                                               subtask='sp')
                        if load_id:
                            seq_loss, correct_sp, example_id_list, seq_score_list = eval_result_batch
                            # print('example_id_list_sp', len(example_id_list))
                            example_id_sp_all.extend(example_id_list)
                            seq_score_all.extend(seq_score_list)
                        else:
                            seq_loss, correct_sp = eval_result_batch
                        eval_loss += seq_loss.item()
                        eval_sentence_loss += seq_loss.item()
                        num_sentence_elements += sequence_label.size(0)
                        eval_corrected_sp += correct_sp
                    eval_sentence_loss /= num_sentence_elements
                    eval_sentence_acc = float(eval_corrected_sp) / num_next_sentence_elements
                results[eval_task + '_loss'] += eval_loss
                results[eval_task + '_sentence'] = eval_sentence_loss
                results[eval_task + '_sentence_acc'] = eval_sentence_acc
                if load_id:
                    results[eval_task + '_example_id_sp'] = example_id_sp_all
                    # print('example_id_sp_all_sp', len(example_id_sp_all))
                    results[eval_task + '_seq_score'] = seq_score_all

        elif args.model_type == 'xlnetpath-clm':
            eval_loss = 0
            num_lm_elements = 0
            if load_id:
                example_id_all = list()
                loss_all = list()
            with torch.no_grad():
                for batch in tqdm(eval_dataloader, desc=f'eval: {eval_desc}', leave=False):
                    batch = tuple(t.to(args.device) for t in batch)
                    all_input_ids, all_attention_masks, all_token_type_ids, all_element_type_ids, \
                    all_sentence_label, all_example_id = batch
                    eval_result_batch = get_loss(args.model_type, model, criterion, batch, evaluate=True, load_id=load_id, load_element_id=load_element_id,)
                    if load_id:
                        loss, lm_logits, example_id = eval_result_batch
                        example_id_all.extend(example_id)
                        loss_all.extend(lm_logits)
                    else:
                        loss, lm_logits = eval_result_batch
                    eval_loss += loss.item()
                    # num_lm_elements += example_id_all#(lm_logits.detach().cpu().numpy().flatten() != criterion.ignore_index).sum()

            eval_loss = eval_loss / len(example_id_all) #num_lm_elements
            results[eval_task + '_loss'] = eval_loss
            results[eval_task + '_perplexity'] = math.exp(eval_loss)
            if load_id:
                results[eval_task + '_example_id'] = example_id_all
                results[eval_task + '_loss_score'] = loss_all

        elif args.model_type == 'xlnetpath-clmnsp':
            eval_loss = 0
            if load_id:
                example_id_lm_all = list()
                example_id_nsp_all = list()
                seq_relationship_score_all = list()
                seq_loss_all = list()

            # maybe multiple tasks, so initialize to 0, and then add loss of each task
            results[eval_task + '_loss'] = 0

            if 'sp' in eval_task:
                eval_lm_loss = 0
                num_lm_elements = 0
                with torch.no_grad():
                    for batch_lm in tqdm(eval_dataloader_lm, desc=f'eval: {eval_desc}', leave=False):
                        batch_lm = tuple(t.to(args.device) for t in batch_lm)
                        input_ids, attention_masks, token_type_ids, element_type_ids, \
                            sequence_label, example_id = batch_lm
                        eval_result_batch = get_loss(args.model_type, model, criterion, batch_lm, evaluate=True,
                                                     load_id=load_id, load_element_id=load_element_id,
                                                     subtask='lm')
                        if load_id:
                            seq_loss, loss_list, example_id_list = eval_result_batch
                            example_id_lm_all.extend(example_id_list)
                            seq_loss_all.extend(loss_list)
                        else:
                            seq_loss, loss_list = eval_result_batch
                        eval_loss += seq_loss.item()
                        eval_lm_loss += seq_loss.item()
                        # num_lm_elements += (input_ids.detach().cpu().numpy().flatten() != criterion.ignore_index).sum()
                        # num_sentence_elements += sequence_label.size(0)
                        # eval_corrected_sp += correct_sp
                    # eval_sentence_loss /= num_sentence_elements
                    eval_lm_loss /= len(example_id_lm_all) #num_lm_elements
                    # eval_sentence_acc = float(eval_corrected_sp) / num_next_sentence_elements
                results[eval_task + '_loss'] += eval_loss
                # results[eval_task + '_masked_lm'] = eval_masked_lm_loss
                results[eval_task + '_lm'] = eval_lm_loss
                # results[eval_task + '_perplexity'] = math.exp(eval_lm_loss)
                # results[eval_task + '_sentence_acc'] = eval_sentence_acc
                if load_id:
                    results[eval_task + '_example_id_lm'] = example_id_lm_all
                    # print('example_id_sp_all_sp', len(example_id_sp_all))
                    results[eval_task + '_loss_score'] = seq_loss_all

            if 'nsp' in eval_task:
                eval_next_sentence_loss = 0
                num_next_sentence_elements = 0
                eval_corrected_nsp = 0
                with torch.no_grad():
                    for batch in tqdm(eval_dataloader, desc=f'eval: {eval_desc}', leave=False):
                        # in: input_ids, attention_mask, token_type_ids, masked_lm_labels
                        batch = tuple(t.to(args.device) for t in batch)
                        # print('batch_eval', batch)
                        input_ids, attention_masks, token_type_ids, element_type_ids, \
                            mlm_label, nsp_label, example_id = batch
                        # print('input_ids_nsp', input_ids.size())
                        eval_result_batch = get_loss(args.model_type, model, criterion, batch, evaluate=True,
                                                     load_id=load_id, load_element_id=load_element_id,
                                                     subtask='nsp')
                        if load_id:
                            next_sentence_loss, correct_nsp, example_id_list, seq_relationship_score_list = eval_result_batch
                            example_id_nsp_all.extend(example_id_list)
                            # print('example_id_list_nsp', len(example_id_list))
                            seq_relationship_score_all.extend(seq_relationship_score_list)
                            # print(example_id)
                        else:
                            next_sentence_loss, correct_nsp = eval_result_batch
                        eval_loss += next_sentence_loss.item()
                        eval_next_sentence_loss += next_sentence_loss.item()
                        num_next_sentence_elements += nsp_label.size(0)
                        eval_corrected_nsp += correct_nsp

                    eval_next_sentence_loss /= num_next_sentence_elements
                    eval_next_sentence_acc = float(eval_corrected_nsp) / num_next_sentence_elements
                results[eval_task + '_loss'] += eval_loss
                results[eval_task + '_next_sentence'] = eval_next_sentence_loss
                results[eval_task + '_next_sentence_acc'] = eval_next_sentence_acc
                if load_id:
                    results[eval_task + '_example_id_nsp'] = example_id_nsp_all
                    results[eval_task + '_seq_relationship_score'] = seq_relationship_score_all

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/ace', type=str)
    parser.add_argument('--model_type', default='bertpath-mlmnspsp', type=str, help='model_type')
    parser.add_argument('--model_name_or_path', default='bertpath-mlmnspsp-ace', type=str, help='model_name_or_path')
    parser.add_argument('--task_name', default='mlmnspsp')
    parser.add_argument('--eval_subtask_name', default=None)
    parser.add_argument('--output_dir', default='', type=str, required=True)
    parser.add_argument('--config_name', default='', type=str)
    parser.add_argument('--tokenizer_name', default='', type=str)
    parser.add_argument('--max_seq_length', default=128, type=int)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--load_id', action='store_true')
    parser.add_argument('--load_element_id', action='store_true')
    parser.add_argument('--train_batch_size', default=16, type=int)
    parser.add_argument('--eval_batch_size', default=64, type=int)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--num_train_epochs', default=1.0, type=float)
    parser.add_argument('--max_steps', default=-1, type=int)
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--num_ckpts', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    log_file = open(os.path.join(args.output_dir, 'log'), 'a')
    redirect_stdout(log_file)

    set_seed(args)

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError('Task not found: %s' % (args.task_name))
    # processor = processors[args.task_name]()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    print('Training/evaluation parameters %s' % str(args))

    if args.do_train:
        config = config_class.from_pretrained(
            args.config_name or args.model_name_or_path,
            finetuning_task=args.task_name,
        )
        tokenizer = get_tokenizer(args.model_type, args.tokenizer_name or args.model_name_or_path, do_lower_case=False)
        criterion = get_criterion(args.model_type, tokenizer)
        print(f'*** Criterion ignore_index = {criterion.ignore_index} ***')
        criterion.to(args.device)

        model = model_class(config=config)
        model.to(args.device)

        model.config.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        args_dict = copy.copy(args.__dict__)
        del args_dict['device']
        json.dump(args_dict, open(os.path.join(args.output_dir, 'args.json'), 'w'), ensure_ascii=False, indent=2)

        train_dataset = load_and_cache_dataset(args, args.task_name, tokenizer, evaluate=False,
                                               load_id=args.load_id, load_element_id=args.load_element_id)
        train(args, train_dataset, model, criterion, tokenizer, load_id=args.load_id, load_element_id=args.load_element_id)

    if not args.do_train and args.do_eval:
        config = config_class.from_pretrained(args.output_dir)
        tokenizer = get_tokenizer(args.model_type, args.output_dir, do_lower_case=False)
        criterion = get_criterion(args.model_type, tokenizer)
        criterion.to(args.device)

        ckpt_paths = [f for f in os.listdir(args.output_dir) if f.startswith('step_') and f.endswith('.bin')]
        ckpt_paths = sorted(ckpt_paths, key=lambda x: int(x.strip('step_').strip('.bin')))
        eval_results = []
        scores= []

        for ckpt_path in ckpt_paths:
            model = torch.load(os.path.join(args.output_dir, ckpt_path))
            model.to(args.device)
            ckpt_step = int(ckpt_path.strip('step_').strip('.bin'))
            step_eval_results = evaluate(args, model, criterion, tokenizer,
                                         eval_subtask=args.eval_subtask_name,
                                         load_id=args.load_id,
                                         load_element_id=args.load_element_id)

            if args.model_type == 'bertpath-lmnsp':
                eval_loss = step_eval_results[args.task_name+'_loss']
                eval_loss_mm = step_eval_results[args.task_name+'_masked_lm']
                eval_loss_nsp = step_eval_results[args.task_name+'_next_sentence']

                acc_nsp = step_eval_results[args.task_name+'_next_sentence_acc']
                example_id = step_eval_results[args.task_name+'_example_id']
                score_seq_relationship = step_eval_results[args.task_name+'_seq_relationship_score']
                # scores.append(dict(zip(example_id, score_seq_relationship)))
                json.dump(dict(zip(example_id, score_seq_relationship)),
                          open(os.path.join(args.output_dir, f"test_scores_step_{global_step}.json"), 'w'),
                          ensure_ascii=False, indent=2)
                eval_results.append([ckpt_step, eval_loss, eval_loss_mm, eval_loss_nsp, acc_nsp])
                print(f'\nCheckpoint = {ckpt_path}, eval_loss = {eval_loss:.2f}, eval_loss_mm = {eval_loss_mm:.2f}, '
                      f'eval_loss_nsp = {eval_loss_nsp:.2f}, acc_nsp = {acc_nsp:.2f}\n')
            elif args.model_type == 'bertpath-mlmnspsp':
                # step_eval_results = evaluate(args, model, criterion, tokenizer,
                #                              load_id=load_id, load_element_id=load_element_id,
                #                              eval_subtask='mlmnspsp')
                # train_loss = np.mean(step_loss)
                eval_loss = step_eval_results[args.task_name + '_loss']
                if 'mlmnsp' in args.eval_subtask_name:
                    eval_loss_mm = step_eval_results[args.task_name + '_masked_lm']
                    eval_loss_nsp = step_eval_results[args.task_name + '_next_sentence']
                    eval_acc_nsp = step_eval_results[args.task_name + '_next_sentence_acc']
                    example_id_nsp = step_eval_results[args.task_name + '_example_id_nsp']
                    score_seq_relationship = step_eval_results[args.task_name + '_seq_relationship_score']
                    print('example_id_nsp', len(example_id_nsp))
                    json.dump(dict(zip(example_id_nsp, score_seq_relationship)),  # [:, 1]
                              open(os.path.join(args.output_dir, f"eval_scores_nsp_step_{ckpt_path}.json"), 'w'),
                              ensure_ascii=False,
                              indent=2)
                    eval_results.append(
                        [ckpt_step, eval_loss, eval_loss_mm, eval_loss_nsp, eval_acc_nsp])
                    print(
                        f'\nCheckpoint = {ckpt_path}, eval_loss = {eval_loss:.2f}, eval_mm_loss = {eval_loss_mm:.2f}, '
                        f'eval_loss_nsp = {eval_loss_nsp:.2f}, '
                        f'eval_acc_nsp = {eval_acc_nsp:.2f}\n')
                if 'sp' in args.eval_subtask_name:
                    eval_loss_sp = step_eval_results[args.task_name + '_sentence']
                    eval_acc_sp = step_eval_results[args.task_name + '_sentence_acc']
                    example_id_sp = step_eval_results[args.task_name + '_example_id_sp']
                    print('example_id_sp', len(example_id_sp))
                    score_seq = step_eval_results[args.task_name + '_seq_score']
                    json.dump(dict(zip(example_id_sp, score_seq)),
                              open(os.path.join(args.output_dir, f"eval_scores_sp_step_{ckpt_path}.json"), 'w'),
                              ensure_ascii=False,
                              indent=2)

                    eval_results.append(
                        [ckpt_step, eval_loss, eval_loss_sp, eval_acc_sp])
                    print(
                        f'\nCheckpoint = {ckpt_path}, eval_loss = {eval_loss:.2f}, '
                        f'eval_loss_sp = {eval_loss_sp:.2f}, '
                        f'eval_acc_sp = {eval_acc_sp:.2f}\n')
            elif args.model_type == 'xlnetpath-clm':
                # step_eval_results = evaluate(args, model, criterion, tokenizer,
                #                              load_id=load_id, load_element_id=load_element_id)
                # train_loss = np.mean(step_loss)
                example_id = step_eval_results[args.task_name + '_example_id']
                eval_loss = step_eval_results[args.task_name + '_loss']
                eval_perplexity = step_eval_results[args.task_name + '_perplexity']
                eval_loss_list = step_eval_results[args.task_name + '_loss_score']
                eval_results.append(
                    [ckpt_step, eval_loss, eval_perplexity])
                json.dump(dict(zip(example_id, eval_loss_list)),
                          open(os.path.join(args.output_dir, f"train_loss_step_{ckpt_step}.json"), 'w'),
                          ensure_ascii=False,
                          indent=2)
                print(
                    f'\nCheckpoint = {ckpt_path}, eval_loss = {eval_loss:.2f}, perplexity = {eval_perplexity:.2f}')
            elif args.model_type == 'xlnetpath-clmnsp':
                # step_eval_results = evaluate(args, model, criterion, tokenizer,
                #                              load_id=load_id, load_element_id=load_element_id,
                #                              eval_subtask='clmnsp')
                # train_loss = np.mean(step_loss)
                example_id_nsp = step_eval_results[args.task_name + '_example_id_nsp']
                example_id_lm = step_eval_results[args.task_name + '_example_id_lm']
                eval_loss = step_eval_results[args.task_name + '_loss']
                eval_loss_nsp = step_eval_results[args.task_name + '_next_sentence']
                eval_loss_lm = step_eval_results[args.task_name + '_lm']

                eval_acc_nsp = step_eval_results[args.task_name + '_next_sentence_acc']
                score_seq_relationship = step_eval_results[args.task_name + '_seq_relationship_score']
                score_lm = step_eval_results[args.task_name + '_loss_score']
                json.dump(dict(zip(example_id_nsp, score_seq_relationship)),  # [:, 1]
                          open(os.path.join(args.output_dir, f"train_scores_nsp_step_{ckpt_step}.json"), 'w'),
                          ensure_ascii=False,
                          indent=2)
                json.dump(dict(zip(example_id_lm, score_lm)),
                          open(os.path.join(args.output_dir, f"train_loss_lm_step_{ckpt_step}.json"), 'w'),
                          ensure_ascii=False,
                          indent=2)
                eval_results.append(
                    [ckpt_step, eval_loss, eval_loss_lm, eval_loss_nsp,
                     eval_acc_nsp])
                print(
                    f'\nSaving model checkpoint to {ckpt_path}, eval_lm_loss = {eval_loss_lm:.2f}, '
                    f'eval_loss_nsp = {eval_loss_nsp:.2f}, '
                    f'eval_acc_nsp = {eval_acc_nsp:.2f}\n')

        if len(eval_results[0]) > 2:
          header = ['step', 'eval_loss', 'eval_mm_loss', 'eval_nsp_loss', 'nsp_acc']
        else:
          header = ['step', 'eval_loss']
        best_results = report_results(header, eval_results, 1)
        best_step = best_results[0]
        # best_scores = scores[best_step]
        # json.dump(best_scores, open(os.path.join(args.output_dir, 'best_simscores.json'), 'w'), ensure_ascii=False, indent=2)
        print(f'best_ckpt = {os.path.join(args.output_dir, f"step_{best_step}.bin")}\n')


