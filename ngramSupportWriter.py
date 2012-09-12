from nltk.corpus.reader import TaggedCorpusReader
from nltk.tokenize import LineTokenizer

from itertools import izip, chain
from enviroment_vars import ReportEnviroments


class NGramSupportWriter(object):

    def take_ngrams_by_topic_from_file(self, 
		                               ngram_directory, 
		                               ngram_file):
        corpus = \
            TaggedCorpusReader(ngram_directory, 
                               ngram_file, 
                               sent_tokenizer=LineTokenizer(blanklines='discard'), 
                               encoding='utf-8')
        corpus_paras = corpus.paras()[:]
        k = corpus_paras[::2]
        for i in range(2):
            k = list(chain(*k))
        v = corpus_paras[1::2]
        ngrams_by_topic_from_file = \
            {k.encode('utf-8'): list(set(chain(*v))) 
               for k, v in dict(izip(k, v)).items()}
        return ngrams_by_topic_from_file


    def merge_run_time_and_ngrams_from_file(self, 
                                            ngrams_by_topic_from_file, 
                                            run_time_ngrams_by_topic):
        merged_run_time_and_ngrams_from_file = \
            dict((k, list(set(chain(*[ngrams_by_topic_from_file[k], 
                                      run_time_ngrams_by_topic[k]])))) 
                             for k in ngrams_by_topic_from_file.keys())
        return merged_run_time_and_ngrams_from_file

    def write_ngrams_in_a_file(self, 
    	                       ngram_directory, 
		                       ngram_file,
		                       ngram_content):
    	with open(ngram_directory+ngram_file, 'w') as f:
            for k, v in ngram_content.items():
                f.write('\n\n' + k + '\n\n')
                for unigram in v:
                    f.write(unigram.encode('utf-8') + '\n')
            f.close()