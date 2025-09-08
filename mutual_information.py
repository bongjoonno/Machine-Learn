def mutual_information(df, independent_class, dependent_class):
  class_counts_by_x = df[[independent_class, dependent_class]].value_counts()
  class_counts_by_y = df[[dependent_class, independent_class]].value_counts()

  total = class_counts_by_x.sum()


  unique_x = df[independent_class].unique()
  unique_y = df[dependent_class].unique()

  mutual_information_total = 0

  for cat1 in unique_x:
    marginal_prob_x = class_counts_by_x[cat1].sum() / total

    for cat2 in unique_y:
      try:
        joint_prob = class_counts_by_x[cat1, cat2] / total

        marginal_prob_y = class_counts_by_y[cat2].sum() / total
        
        mutual_information = joint_prob * log(joint_prob / (marginal_prob_x*marginal_prob_y))
        mutual_information_total += mutual_information
      except KeyError: #permutation may not be present
        break

  return mutual_information_total