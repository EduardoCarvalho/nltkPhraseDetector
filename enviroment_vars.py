from os import environ

class ReportEnviroments(object):

    def __init__(self):
        self.segmented_reports_corpus_path = environ["HOME"]+'/nltkPhraseDetector/segmented_reports'
        self.unigrams_directory = environ["HOME"]+'/nltkPhraseDetector/Phrases/Unigrams/'
        self.nouns_unigrams_fileid = 'nouns_unigrams.txt'
        self.none_unigrams_fileid = 'none_unigrams.txt'
        self.unigrams_folder = environ["HOME"]+'/nltkPhraseDetector/Phrases/Unigrams'
        self.nltkphrasedetector_fold = environ["HOME"]+'/nltkPhraseDetector'

