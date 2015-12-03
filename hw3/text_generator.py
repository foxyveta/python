# coding=utf-8
"""Homework 2. Text generator."""
from __future__ import division

import marshal
import os
import random
import re
import sys
from collections import Counter
from optparse import OptionParser

import porter
import porter2

TYPE_WORD = 0
TYPE_PUNCT_MARK = 1
TYPE_SENTENCE_END = 2

SENTENCE_END = '.!?'
PUNCTUATIONS = {'.', '!', '?', ',', ';', ':', '...'}
CYRILLIC = re.compile(u'[\u0400-\u04ff]')


class Markov(object):
    _PATTERN = re.compile(ur"""
            (\w+(?:[-'’]\w+)?
            (?:[!?](?=\s-))?)
            |
            (\.\.\.|[.!?,:])
            """, re.UNICODE + re.VERBOSE)

    def __init__(self, encoding='utf-8'):
        self.encoding = encoding
        self.word_count = 0

    def _generate_files(self, folder):
        for root, sub_folders, files in os.walk(folder):
            for file_name in files:
                if file_name.endswith('.txt'):
                    path = os.path.join(root, file_name)
                    print u'processing file {}...'.format(path)
                    yield path

    def _generate_lines(self, corpus):
        for file_name in corpus:
            with open(file_name) as text:
                for line in text:
                    yield line.decode(self.encoding)

    def _generate_tokens(self, lines):
        for line in lines:
            for word, stop in self._PATTERN.findall(line):
                self.word_count += 1
                if word:
                    if word.isupper() and len(word) > 1:
                        word = word.lower()
                    start = word[:-1]
                    end = word[-1]
                    if end in SENTENCE_END:
                        yield start, TYPE_WORD
                        yield end, TYPE_PUNCT_MARK
                    else:
                        yield word, TYPE_WORD
                else:
                    yield stop, TYPE_SENTENCE_END \
                        if stop in SENTENCE_END else TYPE_PUNCT_MARK

    def _generate_tuples(self, tokens):
        t0, t1 = None, None
        for t2, t2_type in tokens:
            if t0 or t1 or t2_type != TYPE_SENTENCE_END:
                yield t0, t1, t2
            if t2_type == TYPE_SENTENCE_END:
                yield t1, t2, None
                t0, t1 = None, None
            else:
                t0, t1 = t1, t2

    def train(self, corpus_dir):
        def to_id(word):
            if word is None:
                return None
            size = len(words)
            index = word_map.setdefault(word, size)
            if index == size:
                words.append(word)
            return index

        tuples = self._generate_tuples(self._generate_tokens(
            self._generate_lines(self._generate_files(corpus_dir))))

        words = []
        word_map = {}

        pairs, triples = Counter(), Counter()
        for t0, t1, t2 in tuples:
            id0, id1, id2 = to_id(t0), to_id(t1), to_id(t2)
            pairs[id0, id1] += 1
            triples[id0, id1, id2] += 1

        print 'calculating corpus statistics...'
        frequencies = {}
        for (t0, t1, t2), count in triples.iteritems():
            next_tokens = frequencies.setdefault((t0, t1), [])
            next_tokens.append((t2, count / pairs[t0, t1]))
        print '{} words processed.'.format(self.word_count)
        return Model(frequencies, words, word_map)


class Model(object):
    def __init__(self, frequencies, words, word_map):
        self.frequencies = frequencies
        self.words = words
        self.word_map = word_map

    def generate(self, length):
        current_length = 0
        sentences = []
        while current_length < length:
            sentence_length, sentence = self.generate_sentence()
            current_length += sentence_length
            sentences.append(sentence)

        text = self._make_paragraphs(sentences)
        print u'\n'.join(text).encode('utf-8')

    def generate_sentence(self):
        length = 0
        while length == 0:
            length = 0
            sentence = []
            t0, t1 = None, None
            while True:
                t0, t1 = t1, self._random_token(self.frequencies[t0, t1])
                if not t1:
                    break
                word = self.words[t1]
                if word not in PUNCTUATIONS:
                    length += 1
                    if sentence:
                        sentence.append(u' ')
                sentence.append(word)
                if word in SENTENCE_END:
                    break

            if length < 3:
                length = 0
        sentence[0] = sentence[0].capitalize()
        return length, sentence

    def _random_token(self, tokens):
        random_value = random.uniform(0, 1)
        value = 0
        for token, frequency in tokens:
            value += frequency
            if value > random_value:
                return token

    def _detect_language_stemmer(self):
        sample_size = 1000

        for i in xrange(sample_size):
            word = self.words[i]
            if CYRILLIC.search(word) is not None:
                return porter
        return porter2

    def _make_paragraphs(self, sentences):
        indexes = self._reorder_sentences(sentences)
        size = len(indexes)
        index = 0
        text = []
        while index < size:
            para = []
            par_size = random.randint(4, 10)
            if par_size + index > size:
                par_size = size - index
            for i in xrange(index, index + par_size):
                para.append(sentences[indexes[i]])
            para.append([u'\n'])
            text.append(u' '.join(u''.join(sentence) for sentence in para))
            index += par_size
        return text

    def _reorder_sentences(self, sentences):
        stemming_alg = self._detect_language_stemmer()
        size = len(sentences)
        inf = 10 ** 6
        graph = [[inf for _ in xrange(size)]
                 for _ in xrange(size)]
        vectors = []
        for s in sentences:
            words = {stemming_alg.stem(word) for word in s if
                     word[0].isalpha()}
            vectors.append(words)
        for i, v1 in enumerate(vectors):
            for j in xrange(i + 1, size):
                v2 = vectors[j]
                count = len(v1 & v2)
                graph[i][j] = -count / (len(v1) + len(v2) - count)
                graph[j][i] = graph[i][j]
        path = self._make_path(graph)
        return path

    def _make_path(self, graph):
        def dfs(start):
            visited[start] = True
            min_edge = infinity
            to = infinity
            for v, w in enumerate(graph[start]):
                if not visited[v]:
                    if w < min_edge:
                        min_edge = w
                        to = v
            if to != infinity:
                path.append(to)
                dfs(to)

        infinity = 10 ** 6
        size = len(graph)
        visited = [False] * size
        start = random.randint(0, size - 1)
        path = [start]
        dfs(start)
        return path

    @staticmethod
    def serialize(f, model):
        # cPickle.dump(model, f)
        marshal.dump((model.frequencies, model.words, model.word_map), f)

    @staticmethod
    def deserialize(f):
        # model = cPickle.load(f)
        model = Model(*marshal.load(f))
        return model


def main():
    parser = OptionParser(
        usage='usage: %prog [options] [train|generate [count]]')
    parser.add_option('-d', '--corpus-dir',
                      dest='corpus', default=u'corpus',
                      help='corpus directory to train from')
    parser.add_option('-e', '--corpus-encoding',
                      dest='encoding', default='utf-8',
                      help='encoding of corpus files')
    parser.add_option('-m', '--model',
                      dest='model', default='model.ser',
                      help='model file')
    (options, args) = parser.parse_args()
    if len(args) > 2:
        parser.error('incorrect number of arguments')
    command = args[0] if args else 'generate'
    if command not in ('train', 'generate'):
        parser.error('incorrect argument {}'.format(command))

    if command == 'train':
        markov = Markov(encoding=options.encoding)
        model = markov.train(unicode(options.corpus))
        with open(options.model, 'wb') as f:
            print 'saving statistics to "{}"...'.format(f.name)
            Model.serialize(f, model)
    else:
        try:
            count = int(args[1]) if len(args) == 2 else 10000
        except ValueError:
            sys.exit('Parameter count must be an integer number ')
        try:
            with open(options.model, 'rb') as f:
                model = Model.deserialize(f)
        except IOError:
            sys.exit('Could not open file "{}"'.format(options.model))
        model.generate(count)


if __name__ == '__main__':
    main()
