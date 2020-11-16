import pandas as pd
from collections import defaultdict
import math
import random
from sklearn.model_selection import train_test_split
import os
import copy
import ujson as json


def list2path(path_list):
    return ' '.join(path_list)


def load_paths(path_file):
    '''
    load ace paths from Ying
    :param path_file:
    :return:
    '''
    # paths_df = pd.read_json(path_file, lines=True)
    paths_data = json.load(open(path_file))

    vocab = set()
    path_dict = defaultdict(lambda: defaultdict(list))
    path_pairs = defaultdict(lambda : defaultdict(int))
    path_group = defaultdict(lambda : defaultdict(set))
    # for index, row in paths_df.iterrows():
    #     start_evt = row["path"][0]
    #     end_evt = row["path"][-1]
    #     path = row['path']
    #     path_id = row['id']
    #     count = row['count']
    #     cooccur = row['cooccur']
    for event_type_pair in paths_data:
        for path_info in paths_data[event_type_pair]:
            start_evt, end_evt = event_type_pair.split('__')
            path_id = path_info['id']
            path = path_info['path']
            count = path_info['count']
            cooccur = path_info['cooccur']

            path_group[start_evt][end_evt].add(path_id)
            path_dict[path_id]['path'] = list2path(path)
            path_dict[path_id]['start'] = start_evt
            path_dict[path_id]['end'] = end_evt
            path_dict[path_id]['count'] = int(count)
            vocab.update(path)
            for path_cooccur in cooccur:
                path_cooccur_id = path_cooccur['id']  #path_cooccur[0]
                # probability = path_cooccur[1]
                # cooccur_count = round(probability * count) #math.ceil(probability * count)
                cooccur_count = path_cooccur['freq']
                path_pairs[path_id][path_cooccur_id] = int(cooccur_count)

    return path_pairs, path_dict, path_group, vocab


def sample_negative_sentence_idx(path_seq_idxs, all_path_seq_idxs, vocab_size, rng):

    # replace only one token idx
    ## randomly select the position of the sequence
    path_seq_neg_idx = copy.copy(path_seq_idxs)
    cand_indexes = list(range(0, len(path_seq_idxs) - 1))
    rng.shuffle(cand_indexes)
    while True:
        for index in cand_indexes:
            neg_token_idx = random.randint(0, vocab_size-1)
            path_seq_neg_idx[index] = str(neg_token_idx)

            if list2path(path_seq_neg_idx) not in all_path_seq_idxs:
                return path_seq_neg_idx

    # # use the way of masked language model, replace each token randomely, which is not good
    # masked_lm_prob = 0.15
    # max_predictions_per_seq = 20
    # cand_indexes = list(range(0, len(path_seq)-1))
    # rng.shuffle(cand_indexes)
    # len_cand = len(cand_indexes)
    #
    # path_seq_neg = copy.copy(path_seq)
    # # num_to_predict = min(max_predictions_per_seq,
    # #                      max(1, int(round(input_len * masked_lm_prob))))
    #
    # # masked_lm_labels = [-100] * args.max_seq_length
    # # masked_lms_pos = []
    # # covered_indexes = set()
    # for index in cand_indexes:
    #     # if len(masked_lms_pos) >= num_to_predict:
    #     #     break
    #     # if index in covered_indexes:
    #     #     continue
    #     # covered_indexes.add(index)
    #
    #     neg_token = None
    #     if rng.random() < 0.8:
    #         # replace to one word in the remaining sentence
    #         neg_token = path_seq_neg[cand_indexes[rng.randint(0, vocab_size - 1)]] #len_cand - 1)]]
    #
    #     # masked_lm_labels[index] = init_ids[index]
    #     path_seq_neg[index] = neg_token
    #     # masked_lms_pos.append(index)


def generate_tsv(path_pairs, path_dict, path_group, vocab,
                 save_dir,
                 save_vocab_bert=None,
                 save_vocab_xlnet=None,
                 save_file_lm_tsv=None,
                 save_file_sp_tsv=None,
                 save_file_nsp_tsv=None,
                 save_file_spnsp_tsv=None,
                 threshold=9, test_split=0.1, seed=111, test=False):
    print('vocab len', len(vocab))

    # LM
    write_lm_tsv(save_file_lm_tsv)

    # SP, generate negative samples
    vocab_id2word, vocab_size, all_path_seq_idxs, rng = preprocess_sp_tsv(path_dict, vocab, seed)
    write_sp_tsv(vocab_id2word, vocab_size, all_path_seq_idxs, rng, save_file_sp_tsv)

    # generate NSP negative samples
    path_pairs_neg = sample_neg_pairs(path_pairs, path_group, path_dict, threshold=threshold)
    save_nsp_tsv(path_pairs, path_pairs_neg, path_dict, save_dir, save_file_nsp_tsv=save_file_nsp_tsv,
                 test_split=test_split, test=test)
    print('all positive nsp pairs', sum(path_pairs[path_1][path_2] for path_1 in path_pairs for path_2 in path_pairs[path_1]))
    print('all negative nsp pairs', sum(len(path_pairs_neg[path_1]) for path_1 in path_pairs_neg))

    # generate SPNSP tsv
    save_sp_nsp_tsv(path_pairs, path_pairs_neg, path_dict,
                    vocab_id2word, vocab_size, all_path_seq_idxs, rng,
                    save_file_spnsp_tsv)

    # # vocab
    # if save_vocab_bert:
    #     write_vocab(vocab, save_vocab_bert, model='bert')
    # if save_vocab_xlnet:
    #     write_vocab(vocab, save_vocab_xlnet, model='xlnet')


def preprocess_sp_tsv(path_dict, vocab, seed):
    # vocab id
    vocab_word2id = {vocab_word: str(vocab_idx) for vocab_idx, vocab_word in enumerate(vocab)}
    vocab_id2word = {str(vocab_idx): vocab_word for vocab_word, vocab_idx in vocab_word2id.items()}
    vocab_size = len(vocab_word2id)
    print('vocab_size', vocab_size)

    all_path_seq_idxs = set()
    for path_id in path_dict:
        path_content = path_dict[path_id]['path']
        path_content = path_content.split(' ')
        path_content_idxs = [str(vocab_word2id[element]) for element in path_content]
        path_dict[path_id]['path_idxs'] = path_content_idxs
        all_path_seq_idxs.add(list2path(path_content_idxs))

    rng = random.Random(seed)

    return vocab_id2word, vocab_size, all_path_seq_idxs, rng


def sample_negative_sentence(path_id, vocab_id2word, vocab_size, all_path_seq_idxs, rng):
    path_content_idxs = path_dict[path_id]['path_idxs']
    path_content_neg_idx = sample_negative_sentence_idx(path_content_idxs, all_path_seq_idxs, vocab_size, rng)
    path_content_neg = [vocab_id2word[element] for element in path_content_neg_idx]

    return path_content_neg


def write_sp_tsv(vocab_id2word, vocab_size, all_path_seq_idxs, rng, save_file_sp_tsv):
    # generate SeqP negative samples
    with open(save_file_sp_tsv, 'w') as writer:
        writer.write('index	pathID	event_group	sentence1	gold_label\n')
        index = 0
        for path_id in path_dict:
            # write multiple times according to frequency
            for _ in range(path_dict[path_id]['count']):
                # positive
                path_eventtype = '%s__%s' % (path_dict[path_id]['start'], path_dict[path_id]['end'])
                writer.write('%d\t%s\t%s\t%s\t1\n' % (index, path_id, path_eventtype, path_dict[path_id]['path']))
                index += 1

                # negative
                path_content_neg = sample_negative_sentence(path_id, vocab_id2word, vocab_size, all_path_seq_idxs, rng)
                path_eventtype_neg = '%s__%s' % (path_content_neg[0], path_content_neg[-1])
                writer.write('%d\t-%s\t%s\t%s\t0\n' % (index, path_id, path_eventtype_neg, list2path(path_content_neg)))
                index += 1


def write_lm_tsv(save_file_lm_tsv):
    with open(save_file_lm_tsv, 'w') as writer:
        writer.write('index	pathID	event_group	sentence1	gold_label\n')
        index = 0
        for path_id in path_dict:
            path_eventtype = '%s__%s' % (path_dict[path_id]['start'], path_dict[path_id]['end'])
            # write multiple times according to frequency
            for _ in range(path_dict[path_id]['count']):
                writer.write('%d\t%s\t%s\t%s\t1\n' % (index, path_id, path_eventtype, path_dict[path_id]['path']))
                index += 1


def write_vocab(vocab, save_file, model='bert'):
    with open(save_file, 'w') as fout:
        if model == 'bert':
            fout.write('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n[UNUSED0]\n[UNUSED1]\n[UNUSED2]\n[UNUSED3]\n')
        elif model == 'xlnet':
            fout.write('<pad>\n<unk>\n<cls>\n<sep>\n<mask>\n<s>\n</s>\n<eop>\n<eod>\n')
        for word in vocab:
            if len(word) > 0:
                fout.write('%s\n' % word)


def save_nsp_tsv(path_pairs, path_pairs_neg, path_dict, output_dir, save_file_nsp_tsv=None,
                 test_split=0.1, test=False):
    # index	pairID	event_group	sentence1	sentence2	gold_label
    lines = list()

    for path_1 in path_pairs:
        start_evt = path_dict[path_1]['start']
        end_evt = path_dict[path_1]['end']
        for path_2 in path_pairs[path_1]:
            start_evt_2 = path_dict[path_2]['start']
            end_evt_2 = path_dict[path_2]['end']
            # delete the pairs with different event types
            if start_evt != start_evt_2 or end_evt != end_evt_2:
                continue
            # write multiple times according to frequency
            if test:
                # test only write once
                lines.append('%s_%s\t%s_%s\t%s\t%s\t1' %
                             (path_1, path_2, start_evt, end_evt,
                              path_dict[path_1]['path'], path_dict[path_2]['path']))
                # negative all, do not need sample
                if len(path_pairs_neg[path_1]) > 0:
                    sampled_neg_pairs = list(path_pairs_neg[path_1].keys())
                    for path_2 in sampled_neg_pairs:
                        lines.append('%s_%s\t%s_%s\t%s\t%s\t0' %
                                     (path_1, path_2, start_evt, end_evt,
                                      path_dict[path_1]['path'], path_dict[path_2]['path']))
            else:
                for _ in range(path_pairs[path_1][path_2]):
                    lines.append('%s_%s\t%s_%s\t%s\t%s\t1' %
                               (path_1, path_2, start_evt, end_evt,
                                path_dict[path_1]['path'], path_dict[path_2]['path']))
                # sample negative
                if len(path_pairs_neg[path_1]) > 0:
                    sampled_neg_pairs = random.choices(list(path_pairs_neg[path_1].keys()),
                                   weights=list(path_pairs_neg[path_1].values()), k=path_pairs[path_1][path_2])
                    for path_2 in sampled_neg_pairs:
                        lines.append('%s_%s\t%s_%s\t%s\t%s\t0' %
                                     (path_1, path_2, start_evt, end_evt,
                                      path_dict[path_1]['path'], path_dict[path_2]['path']))

        # # generate all negative pairs, which ignores the single-path frequency
        # for path_2 in path_pairs_neg[path_1]:
        #     lines.append('%s_%s\t%s_%s\t%s\t%s\t0' %
        #                    (path_1, path_2, start_evt, end_evt,
        #                     path_dict[path_1]['path'], path_dict[path_2]['path']))

    # # generate those paths not in path_pairs:
    # for

    if test_split > 0:
        # random.shuffle(lines)
        lines_train, lines_test = train_test_split(lines, test_size=test_split, random_state=1)
        val_size = test_split / (1-test_split)
        lines_train, lines_val = train_test_split(lines_train, test_size=val_size, random_state=1)  # 0.25 x 0.8 = 0.2

        write_tsv(lines_train, os.path.join(output_dir, 'train.tsv'))
        write_tsv(lines_test, os.path.join(output_dir, 'test.tsv'))
        write_tsv(lines_val, os.path.join(output_dir, 'dev.tsv'))
    else:
        write_tsv(lines, save_file_nsp_tsv)


def write_tsv(lines, save_file):
    index = 0
    with open(save_file, 'w') as fout:
        fout.write('index	pathID	event_group	sentence1	sentence2	gold_label\n')
        for line in lines:
            fout.write('%d\t%s\n' % (index, line))
            index += 1


def save_sp_nsp_tsv(path_pairs, path_pairs_neg, path_dict,
                    vocab_id2word, vocab_size, all_path_seq_idxs, rng,
                    save_file):
    # index	pairID	event_group	sentence1	sentence2	sentence1_neg	sentence2_neg	same_node1	same_node2	gold_nsp_label
    index = 0
    with open(save_file, 'w') as fout:
        fout.write('index	pairID	event_group	sentence1	sentence2	sentence1_neg	sentence2_neg	same_node1	same_node2	gold_nsp_label\n')
        for path_1 in path_pairs:
            start_evt = path_dict[path_1]['start']
            end_evt = path_dict[path_1]['end']
            # positive
            for path_2 in path_pairs[path_1]:
                start_evt_2 = path_dict[path_2]['start']
                end_evt_2 = path_dict[path_2]['end']
                # delete the pairs with different event types
                if start_evt != start_evt_2 or end_evt != end_evt_2:
                    continue
                # write multiple times according to frequency
                for _ in range(path_pairs[path_1][path_2]):
                    path1_neg = sample_negative_sentence(path_1, vocab_id2word, vocab_size, all_path_seq_idxs, rng)
                    path2_neg = sample_negative_sentence(path_2, vocab_id2word, vocab_size, all_path_seq_idxs, rng)
                    same_node1 = list()
                    same_node2 = list()
                    line = '%s_%s\t%s_%s\t%s\t%s\t%s\t%s\t%s\t%s\t1' % \
                                 (path_1, path_2, start_evt, end_evt,
                                  path_dict[path_1]['path'], path_dict[path_2]['path'],
                                  list2path(path1_neg), list2path(path2_neg),
                                  list2path(same_node1), list2path(same_node2)
                                  )
                    fout.write('%d\t%s\n' % (index, line))
                    index += 1

                # sample negative
                if len(path_pairs_neg[path_1]) > 0:
                    sampled_neg_pairs = random.choices(list(path_pairs_neg[path_1].keys()),
                                                       weights=list(path_pairs_neg[path_1].values()),
                                                       k=path_pairs[path_1][path_2])
                    for path_2 in sampled_neg_pairs:
                        path1_neg = sample_negative_sentence(path_1, vocab_id2word, vocab_size, all_path_seq_idxs, rng)
                        path2_neg = sample_negative_sentence(path_2, vocab_id2word, vocab_size, all_path_seq_idxs, rng)
                        same_node1 = list()
                        same_node2 = list()
                        line = '%s_%s\t%s_%s\t%s\t%s\t%s\t%s\t%s\t%s\t0' % \
                                     (path_1, path_2, start_evt, end_evt,
                                      path_dict[path_1]['path'], path_dict[path_2]['path'],
                                      list2path(path1_neg), list2path(path2_neg),
                                      list2path(same_node1), list2path(same_node2)
                                      )
                        fout.write('%d\t%s\n' % (index, line))
                        index += 1


def sample_neg_pairs(path_pairs, path_group, path_dict, threshold=9):
    path_pairs_neg = defaultdict(lambda : defaultdict(int))

    for path_id in path_pairs:
        # for path_id_2 in path_pairs:
        start_evt = path_dict[path_id]['start']
        end_evt = path_dict[path_id]['end']
        for path_id_2 in path_group[start_evt][end_evt]:
            # if path_dict[path_id_2]['count'] > threshold:
            #     # use the in-frequent ones as negative
            #     continue
            if path_id_2 != path_id and path_id_2 not in path_pairs[path_id]:
                path_pairs_neg[path_id][path_id_2] = path_dict[path_id_2]['count']

    return path_pairs_neg


if __name__ == '__main__':
    seed = 111
    vocab_only_train = False
    vocab_all = set()
    ace_dir = 'ace.system'
    save_file_dir = '../../../data/%s' % ace_dir
    save_vocab_bert = '../../../conf/%s/bertpath-mlmnspsp-ace-vocab.txt' % ace_dir
    save_vocab_xlnet = '../../../conf/%s/xlnetpath-clm-ace-vocab.txt' % ace_dir
    save_vocab_entity = '../../../conf/%s/vocab-entity.txt' % ace_dir
    save_vocab_event = '../../../conf/%s/vocab-event.txt' % ace_dir
    if not os.path.exists(os.path.join(save_file_dir, 'lm')):
        os.makedirs(os.path.join(save_file_dir, 'lm'), exist_ok=True)
    if not os.path.exists(os.path.join(save_file_dir, 'sp')):
        os.makedirs(os.path.join(save_file_dir, 'sp'), exist_ok=True)
    if not os.path.exists(os.path.join(save_file_dir, 'nsp')):
        os.makedirs(os.path.join(save_file_dir, 'nsp'), exist_ok=True)
    if not os.path.exists(os.path.join(save_file_dir, 'spnsp')):
        os.makedirs(os.path.join(save_file_dir, 'spnsp'), exist_ok=True)


    for mode in ['train']: #, 'test', 'dev']:
        if vocab_only_train:
            if mode == 'train':
                save_vocab_bert = '../../../conf/%s/bertpath-mlmnspsp-ace-vocab.txt' % ace_dir
                save_vocab_xlnet = '../../../conf/%s/xlnetpath-clm-ace-vocab.txt' % ace_dir
            else:
                save_vocab_bert = None
                save_vocab_xlnet = None
        else:
            save_vocab_bert = None
            save_vocab_xlnet = None

        path_file = os.path.join(save_file_dir, '%s.paths.direct.json' % mode) #'../../../data/ace/id.paths.all.json'
        save_file_lm_tsv = os.path.join(save_file_dir, 'lm', '%s_lm.tsv' % mode)
        save_file_sp_tsv = os.path.join(save_file_dir, 'sp', '%s_sp.tsv' % mode)
        save_file_nsp_tsv = os.path.join(save_file_dir, 'nsp', '%s_nsp.tsv' % mode)
        save_file_spnsp_tsv = os.path.join(save_file_dir, 'spnsp', '%s_spnsp.tsv' % mode)
        save_file_nsp_txt = os.path.join(save_file_dir, 'nsp', '%s.txt' % mode)

        if mode == 'train':
            test = False
        else:
            test = True

        path_pairs, path_dict, path_group, vocab = load_paths(path_file)
        with open(save_file_nsp_txt, 'w') as writer:
            for path_id_1 in path_pairs:
                for path_id_2 in path_pairs[path_id_1]:
                    co_occur_count = path_pairs[path_id_1][path_id_2]
                    writer.write('%s\t%s\t%s\t%s\t%d\n' % (
                        path_dict[path_id_1]['start'],
                        path_dict[path_id_1]['end'],
                        path_dict[path_id_1]['path'],
                        path_dict[path_id_2]['path'],
                        co_occur_count
                    ))

        generate_tsv(path_pairs, path_dict, path_group, vocab,
                     save_file_dir,
                     # save_vocab_bert=save_vocab_bert,
                     # save_vocab_xlnet=save_vocab_xlnet,
                     save_file_lm_tsv=save_file_lm_tsv,
                     save_file_sp_tsv=save_file_sp_tsv,
                     save_file_nsp_tsv=save_file_nsp_tsv,
                     save_file_spnsp_tsv=save_file_spnsp_tsv,
                     threshold=9, test_split=0,
                     seed=seed,
                     test=test)

        if mode == 'train':
            vocab_all.update(vocab)
        else:
            if not vocab_only_train:
                vocab_all.update(vocab)

    if not vocab_only_train:
        # vocab
        if save_vocab_bert:
            write_vocab(vocab_all, save_vocab_bert, model='bert')
        if save_vocab_xlnet:
            write_vocab(vocab_all, save_vocab_xlnet, model='xlnet')