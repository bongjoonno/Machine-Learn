def train_test_split(x, y, train_split):
    n = len(x)
    
    training_border = int(train_split * n)
    
    x_train = x[:training_border]
    y_train = y[:training_border]
    
    x_test = x[training_border:]
    y_test = y[training_border:]

x = [1,2,3,4,5,6,7,8,9]
y = [1,2,3,4,5,6,7,8,9]

print(x, y)