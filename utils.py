import random
import numpy as np
import torch

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    BertTokenizer,
    GPT2Tokenizer,
    BERTPathTokenizer,
    XLNetPathTokenizer
)


def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if hasattr(args, 'n_gpu') and args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)


def align_column(row):
  row_str = ''
  for i, item in enumerate(row):
    if 'float' in item.__class__.__name__:
      item = f'{item:.2f}'
    if i == 0:
      row_str += f'{item:>12}'
    else:
      row_str += f'{item:>10}'
  return row_str


def report_results(header, results, axis):
  n_column = len(header)
  metric = header[axis].split('_')[-1]
  if metric in {'acc', 'f1'}:
    cmp = lambda x1, x2: x1 < x2
    best_row = [0] * n_column
  elif metric in {'loss', 'ppl'}:
    cmp = lambda x1, x2: x1 > x2
    best_row = [10000] * n_column
  else:
    raise NotImplementedError
  print()
  print(align_column(header))
  if results[0][0] == 'before':
    before_row = results[0]
    results = results[1:]
    print(align_column(before_row))
  else:
    before_row = None
  print('-' * (n_column * 10 + 2))
  for row in results:
    print(align_column(row))
    if cmp(best_row[axis], row[axis]):
      best_row = row
  print('-' * (n_column * 10 + 2))
  if metric in {'acc', 'f1'}:
    overfit = results[-1][axis] < best_row[axis] - 0.01
  elif metric in {'loss', 'ppl'}:
    overfit = best_row[axis] + 0.01 < results[-1][axis]
  else:
    raise NotImplementedError
  print(align_column([f'best: {best_row[0]}'] + best_row[1:] + (['(overfit)'] if overfit else [])))
  if before_row is not None:
    print(align_column(['gain'] + [best - before for (best, before) in zip(best_row[1:], before_row[1:])]))
  return best_row


def get_adamw(model, num_train_steps, num_warmup_steps, learning_rate, weight_decay=0.01):
  param_optimizer = list(model.named_parameters())
  no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
    {
      'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
      'weight_decay': weight_decay
    },
    {
      'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
      'weight_decay': 0.0
    }
  ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
  scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps)
  return optimizer, scheduler


def get_tokenizer(model_type, model_name_or_path, do_lower_case=True):
  if model_type == 'bert':
    tokenizer_class = BertTokenizer
    pad_token = '[PAD]'
  elif model_type == 'gpt2':
    tokenizer_class = GPT2Tokenizer
    pad_token = '<|endoftext|>'
  elif model_type.startswith('bertpath'):
    tokenizer_class = BERTPathTokenizer
    pad_token = '[PAD]'
  elif model_type.startswith('xlnetpath'):
    tokenizer_class = XLNetPathTokenizer
    pad_token = '<pad>'
  else:
    raise NotImplementedError
  tokenizer = tokenizer_class.from_pretrained(
    model_name_or_path,
    do_lower_case=do_lower_case,
    pad_token=pad_token,
  )
  return tokenizer


def get_criterion(model_type, tokenizer):
  if model_type == 'bert':
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    return criterion
  elif model_type == 'gpt2':
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.eos_token_id)
    return criterion
  elif model_type.startswith('bertpath'):
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    return criterion
  elif model_type.startswith('xlnetpath'):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)#tokenizer.eos_token_id)
    return criterion
  else:
    raise NotImplementedError
