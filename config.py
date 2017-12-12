class Config:
    def __init__(self):
        self.ner_model_path = './stanford/classifiers/english.all.3class.distsim.crf.ser.gz'
        self.ner_jar_path = './stanford/stanford-ner.jar'
        self.WH_words = ['how', 'what', 'where', 'when', 'who', 'which']
        self.actions = []
        self.word2vec_model_path = './models/pruned.word2vec.txt'
        self.embedding_size = 300
        self.dep_jar_path = './stanford/parser/stanford-parser.jar'
        self.dep_model_path = './stanford/parser/stanford-parser-3.7.0-models.jar'
        self.pos_jar_path = './stanford/postagger/stanford-postagger.jar'
        self.pos_model_path = './stanford/postagger/models/english-bidirectional-distsim.tagger'