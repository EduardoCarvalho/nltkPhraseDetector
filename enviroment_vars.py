from os import environ

class ReportEnviroments(object):

    def __init__(self):
        self.segmented_reports_corpus_path = environ["HOME"]+'/nltkPhraseDetector/segmented_reports/segmented_reports_with_empty_topic'
        self.unigrams_list = environ["HOME"]+'/nltkPhraseDetector/Phrases/Unigrams/'
        self.unigrams_folder = environ["HOME"]+'/nltkPhraseDetector/Phrases/Unigrams'
        self.nltkphrasedetector_fold = environ["HOME"]+'/nltkPhraseDetector'

