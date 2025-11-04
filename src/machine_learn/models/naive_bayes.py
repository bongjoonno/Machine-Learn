from math import log

class NaiveBayes:
  def __init__(self):
    pass

  def train(self, class_labels: list, texts_by_class: list):
    total_number_of_texts = sum([len(text) for text in texts_by_class])
    
    self.prior_probabilities = [len(text)/total_number_of_texts for text in texts_by_class]

    self.word_counts_by_class = {label: {} for label in class_labels}
    vocab_set = set()
    
    for i, texts in enumerate(texts_by_class):
      cur_class = class_labels[i]

      for sentence in texts:
        words_in_sentence = sentence.lower().split()

        for word in words_in_sentence:
          self.word_counts_by_class[cur_class][word] = self.word_counts_by_class[cur_class].get(word, 0) + 1
          
          vocab_set.add(word)

    self.vocab_size  = len(vocab_set)

    self.total_words_by_class = {cur_class: sum(word_counts.values()) for cur_class, word_counts in self.word_counts_by_class.items()}


  def predict(self, sentence: str, alpha=1):
    words_in_sentence = sentence.lower().split()

    scores_by_class = {class_name: 0 for class_name, _ in self.word_counts_by_class.items()}
    
    vocab_size_laplace_smoothing = self.vocab_size * alpha

    for i, (cur_class, word_counts) in enumerate(self.word_counts_by_class.items()):
      total_words = self.total_words_by_class[cur_class]
      log_probs = log(self.prior_probabilities[i])

      total_words_with_laplace_smoothing = (total_words + vocab_size_laplace_smoothing)
      
      for word in words_in_sentence:
        count = word_counts.get(word, 0) + alpha
        
        laplace_smoothing = count / total_words_with_laplace_smoothing
        
        log_probs += log(laplace_smoothing)
      
      scores_by_class[cur_class] = log_probs
    
    return scores_by_class