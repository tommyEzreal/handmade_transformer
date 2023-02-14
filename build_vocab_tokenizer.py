from torchtext.data import Field
import os
import pickle
import argparse

import pandas as pd
from pathlib import Path

from torchtext import data 
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer

import spacy
import en_core_web_sm


class BuildVocaToken():
  def __init__(self, corpus_path, train_path, src_vocab_size, trg_vocab_size):
    self.corpus_path = corpus_path
    self.train_path = train_path
    self.spacy_en = spacy.load('en_core_web_sm')
    self.src_vocab_size = src_vocab_size
    self.trg_vocab_size = trg_vocab_size
    
    # empty Field 객체 생성 
    self.SRC = Field(init_token='<SOS>', eos_token='<EOS>',lower=True, batch_first=True)
    self.TRG = Field(init_token='<SOS>', eos_token='<EOS>', lower=True, batch_first=True)

  # 영어(English) 문장을 토큰화 하는 함수 using spacy
  def tokenize_en(self,text):
    spacy_en = self.spacy_en
    return [token.text for token in spacy_en.tokenizer(text)]

  # 한국어 토크나이저 빌드 using soynlp
  def build_tokenizer(self):
    print("Building soynlp tokenizer from corpus data..")

    corpus_path = self.corpus_path
    df = pd.read_csv(corpus_path, encoding='utf-8')

    # skip the non-text rows
    kor_lines = [row.korean for _, row in df.iterrows() if type(row.korean) == str]

    word_extractor = WordExtractor(min_frequency = 5) # 5번이상 출현단어 
    word_extractor.train(kor_lines)

    word_scores = word_extractor.extract()
    cohesion_scores = {word: score.cohesion_forward
                      for word, score in word_scores.items()}

    return LTokenizer(scores = cohesion_scores)

  def build_vocab(self):
    tokenizer = self.build_tokenizer()
    train_data = convert_to_dataset(self.train_path, self.SRC, self.TRG)
    
    self.SRC.tokenize = tokenizer.tokenize
    self.TRG.tokenize = self.tokenize_en

    print('Build vocab from train_data..')

    self.SRC.build_vocab(train_data, max_size=self.src_vocab_size, min_freq=2)
    self.TRG.build_vocab(train_data, max_size=self.trg_vocab_size, min_freq=2)

    print(f'Unique tokens in Source vocab: {len(self.SRC.vocab)}')
    print(f'Unique tokens in Target vocab: {len(self.TRG.vocab)}')

    return self.SRC, self.TRG
