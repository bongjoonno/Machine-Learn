from math import log

def naive_bayes_prep(class_labels, texts_by_class):
  word_counts_by_class = {label: {} for label in class_labels}
  vocab_set = set()
  
  for i, texts in enumerate(texts_by_class):
    cur_class = class_labels[i]
    #iterate through each row getting word counts
    for sentence in texts:
      words_in_sentence = sentence.lower().split()

      for word in words_in_sentence:
        word_counts_by_class[cur_class][word] = word_counts_by_class[cur_class].get(word, 0) + 1

        #all unique words
        vocab_set.add(word)

  vocab_size  = len(vocab_set)

  total_words_by_class = {cur_class: sum(word_counts.values()) for cur_class, word_counts in word_counts_by_class.items()}

  return word_counts_by_class, total_words_by_class, vocab_size


def naive_bayes_predict(sentence, word_counts_by_class, total_words_by_class, vocab_size, prior_probs=[], alpha=1):
  if not prior_probs:
    num_classes = len(word_counts_by_class)
    prior_probs = [1 / num_classes for _ in range(num_classes)]

  words_in_sentence = sentence.lower().split()

  scores_by_class = {class_name: 0 for class_name, _ in word_counts_by_class.items()}

  for i, (cur_class, word_counts) in enumerate(word_counts_by_class.items()):
    total_words = total_words_by_class[cur_class]
    log_probs = log(prior_probs[i])

    for word in words_in_sentence:
      count = word_counts.get(word, 0) + alpha
      
      laplace_smoothing = count / (total_words + (vocab_size * alpha))
      
      log_probs += log(laplace_smoothing)
    
    scores_by_class[cur_class] = log_probs
  
  return scores_by_class