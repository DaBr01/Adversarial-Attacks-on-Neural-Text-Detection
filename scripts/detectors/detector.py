class Detector:
    def __init__(self):
        self.name = "Detector"
        pass

    # returns individual score
    def detect(self, text):
        pass

    # returns all scores for texts in the dictionary.
    # argument: content = {"ID", "text"}
    # return: {"ID", "score"}
    # if an error occured, the score will be set to None.
    def detect_batch(self, content, threads):
        pass