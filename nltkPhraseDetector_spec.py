from unittest import TestCase
from should_dsl import should

from extractPhrases import PhrasesRequirementProcessor
from specvar import Variables


class TestPhrasesRequirementProcessor(TestCase):

    def setUp(self):
        self.v = Variables()
        self.prp = PhrasesRequirementProcessor()
        
    def it_generates_new_corpus_from_segmented_reports(self):
        self.prp.generate_corpus_from_segmented_reports |should| equal_to((self.v.cut_of_segmented_reports, self.v.topics))
        
    def it_aggregates_sentences_of_topics_on_segmented_reports(self):
        self.prp.aggregate_topics_of_segmented_reports(self.v.cut_of_segmented_reports, self.v.topics) \
        |should| equal_to(self.v.aggregated_topics)
        
    def it_organizes_aggregated_topics_by_dictionary(self):
        self.prp.organize_aggregated_topics_by_dict(self.v.aggregated_topics, self.v.topics) \
        |should| equal_to(self.v.dict_of_sentences_by_topic)
        
    def it_tags_unigrams_by_topic(self):
        self.prp.tag_unigrams_by_topic(self.v.dict_of_sentences_by_topic) \
        |should| equal_to(self.v.tagged_unigrams_by_topic)
       
    def it_generates_nouns_unigrams_by_topic(self):
        self.prp.generate_nouns_unigrams_by_topic(self.v.tagged_unigrams_by_topic) \
        |should| equal_to(self.v.nouns_unigrams_by_topic)
            
    def it_creates_a_dictionary_model_for_test_accuracy_of_tagger_by_topic(self):
        self.prp.create_a_dict_model_for_test_accuracy(self.v.tagged_unigrams_by_topic) \
        |should| equal_to((self.v.dict_model_by_topic, self.v.tagger_accuracy_by_topic))
        
    def it_creates_wordtypes_of_nouns_unigrams_by_topic(self):
        self.prp.create_wordtypes_of_nouns_unigrams_by_topic(self.v.nouns_unigrams_by_topic) \
        |should| equal_to(self.v.wordtypes_of_nouns_unigrams_by_topic)    
        
