from extractPhrases import PhrasesRequirementProcessor


class runPhraseDetector(object):

    def detect_unigrams(self):
        prp = PhrasesRequirementProcessor()
        cut_of_segmented_reports, \
        topics = prp.generate_corpus_from_segmented_reports
        aggregated_topics = \
            prp.aggregate_topics_of_segmented_reports(cut_of_segmented_reports, 
                                                      topics)
        dict_of_sentences_by_topic = \
            prp.organize_aggregated_topics_by_dict(aggregated_topics, 
                                                   topics)
        tagged_unigrams_by_topic = prp.tag_unigrams_by_topic(dict_of_sentences_by_topic)
        nouns_unigrams_by_topic = \
            prp.generate_nouns_unigrams_by_topic(tagged_unigrams_by_topic)
        none_unigrams_by_topic = \
            prp.generate_none_unigrams_by_topic(tagged_unigrams_by_topic)
        dict_model_by_topic, \
        tagger_accuracy_by_topic = \
            prp.create_a_dict_model_for_test_accuracy(tagged_unigrams_by_topic)
        run_time_most_frequent_nouns_unigrams_by_topic = \
            prp.create_most_frequent_nouns_unigrams_by_topic(nouns_unigrams_by_topic)
        run_time_wordtypes_of_none_unigrams_by_topic = prp.create_wordtypes_of_none_unigrams_by_topic(none_unigrams_by_topic)
        prp.create_unigram_set_of_nouns_and_nones(run_time_most_frequent_nouns_unigrams_by_topic, run_time_wordtypes_of_none_unigrams_by_topic)
        prp.show_accuracy_by_topic(tagger_accuracy_by_topic)
        prp.remove_pyc_and_zombie_files

rpd = runPhraseDetector()
rpd.detect_unigrams()
