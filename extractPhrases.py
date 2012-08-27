from nltk.corpus.reader import TaggedCorpusReader
from nltk.tokenize import LineTokenizer
from nltk.corpus import mac_morpho
from nltk.tag import UnigramTagger


from os import environ
from commands import getstatusoutput

from nltk.util import bigrams
from enviroment_vars import ReportEnviroments


class PhrasesRequirementProcessor(object):

    def __init__(self):
        object.__init__(self)

    @property   
    def generate_corpus_from_segmented_reports(self):
        re = ReportEnviroments()
        new_corpus_of_segmented_reports = TaggedCorpusReader(re.segmented_reports_corpus_path, '.*',
                                                             sent_tokenizer=LineTokenizer(blanklines='discard'),
                                                             encoding='utf-8')
        raw_segmented_reports = []
        for i in range(len(new_corpus_of_segmented_reports.fileids())):
            raw_segmented_reports.append(new_corpus_of_segmented_reports.sents(fileids=new_corpus_of_segmented_reports.fileids()[i]))
        cut_of_segmented_reports = []
        topics = ['DISCENTE', 'DOCENTE', 'INFRAESTRUTURA', 'UNCATEGORIZED']
        for i in range(len(raw_segmented_reports)):
            cut_of_segmented_reports.append(raw_segmented_reports[i][raw_segmented_reports[i].index([topics[0].decode('utf-8')]):raw_segmented_reports[i].index([topics[-1].decode('utf-8')])+1])
        return cut_of_segmented_reports, topics

    def aggregate_topics_of_segmented_reports(self, cut_of_segmented_reports, topics):
        aggregated_topics = []
        bigrams_of_topics = bigrams(map(lambda x: [x.decode('utf-8')], topics))
        for i in range(len(bigrams_of_topics)):
            for j in range(len(cut_of_segmented_reports)):
                aggregated_topics.extend(cut_of_segmented_reports[j][cut_of_segmented_reports[j].index(bigrams_of_topics[i][0]):cut_of_segmented_reports[j].index(bigrams_of_topics[i][1])])
        return aggregated_topics
        
    def organize_aggregated_topics_by_dict(self, aggregated_topics, topics):
        aggregated_topics.append([None])
        topics.pop()
        topics.append(None)
        modified_topics = map(lambda x: [x.decode('utf-8')], topics[0:-1])
        modified_topics.append([None])
        modified_bigrams_of_topics = bigrams(modified_topics)
        aggregated_list_of_tuple = []
        for i in range(len(modified_bigrams_of_topics)):
            aggregated_list_of_tuple.append(tuple([aggregated_topics[aggregated_topics.index(modified_bigrams_of_topics[i][0])][0].encode('utf-8'), 
                                                   aggregated_topics[aggregated_topics.index(modified_bigrams_of_topics[i][0]):aggregated_topics.index(modified_bigrams_of_topics[i][1])]]))
        dict_of_sentences_by_topic = dict(aggregated_list_of_tuple)
        for k, v in dict_of_sentences_by_topic.items():
            dict_of_sentences_by_topic[k] = [s for s in dict_of_sentences_by_topic[k] 
                                                          if s != [k.decode('utf-8')]]
        for k, v in dict_of_sentences_by_topic.items():
            for i in range(len(dict_of_sentences_by_topic[k])):
                dict_of_sentences_by_topic[k][i] = map(lambda w: w.lower(), 
                                                       dict_of_sentences_by_topic[k][i])
        return dict_of_sentences_by_topic
                        
    def tag_unigrams_by_topic(self, dict_of_sentences_by_topic):
        tagged_unigrams_by_topic = {}
        train_sents = mac_morpho.tagged_sents()[:5000]
        tagger = UnigramTagger(train_sents)
        for k, v in dict_of_sentences_by_topic.items():
            tagged_unigrams_by_topic[k] = tagger.batch_tag(dict_of_sentences_by_topic[k])
        return tagged_unigrams_by_topic
    
    def generate_nouns_unigrams_by_topic(self, tagged_unigrams_by_topic):
        nouns_unigrams_by_topic = {}
        for k, v in tagged_unigrams_by_topic.items():
            nouns_unigrams_by_topic[k] = [[w for (w, t) in sent 
                                             if t=='N'.decode('utf-8')] 
                                             for sent in tagged_unigrams_by_topic[k]]
        for k, v in nouns_unigrams_by_topic.items():
            nouns_unigrams_by_topic[k] = [w for w in nouns_unigrams_by_topic[k] 
                                            if w != []]
        return nouns_unigrams_by_topic
        
    def create_a_dict_model_for_test_accuracy(self, tagged_unigrams_by_topic):
        pre_model = {k: map(dict, v) for k, v in tagged_unigrams_by_topic.items()}
        for k, v in pre_model.items():
            reference_model_by_topic = {}
            for i in v:
                reference_model_by_topic.update(i)
            pre_model[k] = reference_model_by_topic
        dict_model_by_topic = pre_model
        test_sents = mac_morpho.tagged_sents()[:5000]
        tagger_accuracy_by_topic = {}
        for k, v in pre_model.items():
            tagger_accuracy_by_topic[k] = UnigramTagger(model=pre_model[k]).evaluate(test_sents)
        return dict_model_by_topic, tagger_accuracy_by_topic 
        
    def create_wordtypes_of_nouns_unigrams_by_topic(self, nouns_unigrams_by_topic):
        for k, v in nouns_unigrams_by_topic.items():
            for i in range(1, len(nouns_unigrams_by_topic[k])):
                nouns_unigrams_by_topic[k][0].extend(nouns_unigrams_by_topic[k][i])
        wordtypes_of_nouns_unigrams_by_topic = {}  
        for k, v in nouns_unigrams_by_topic.items():
            wordtypes_of_nouns_unigrams_by_topic[k] = list(set(nouns_unigrams_by_topic[k][0]))
        for k, v in wordtypes_of_nouns_unigrams_by_topic.items():
            wordtypes_of_nouns_unigrams_by_topic[k] = map(lambda x: x.encode('utf-8'), 
                                                          wordtypes_of_nouns_unigrams_by_topic[k])
        return wordtypes_of_nouns_unigrams_by_topic      

    def create_unigrams_list(self, wordtypes_of_nouns_unigrams_by_topic):
        re = ReportEnviroments()
        with open(re.unigrams_list+"unigrams.txt", 'w') as f:
            for k, v in wordtypes_of_nouns_unigrams_by_topic.items():
                f.write('\n\n' + k + '\n\n')
                for unigram in v:
                    f.write(unigram + '\n')
            f.close()
        
    def show_accuracy_by_topic(self, tagger_accuracy_by_topic):
        print '\n'
        print '{0:15}  {1:5}'.format('topic', 'accuracy')
        for k, v in tagger_accuracy_by_topic.items():
            print '{0:15}  {1:6.2f} %'.format(k, v*100.0)
            
    @property        
    def remove_pyc_and_zombie_files(self):
        re = ReportEnviroments()
        commands_set = ['rm -f *.pyc ', 'rm -f *~ ']
        for i in range(len(commands_set)):
            getstatusoutput(commands_set[i] + re.nltkphrasedetector_fold)
        getstatusoutput(commands_set[1] + re.segmented_reports_corpus_path)
        getstatusoutput(commands_set[1] + re.unigrams_folder)
        
