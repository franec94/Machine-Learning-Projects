from keras import models
from keras import layers


def compile_model(model, n_classes, optimizer='rmsprop'):
    if n_classes == 0:
        model.add(layers.Dense(1))
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    else:
        if n_classes > 2:
            model.add(layers.Dense(n_classes, activation='softmax'))
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_model(n_features, n_classes = 0, num_units = 64, activation = 'relu', optimizer = 'rmsprop'):
    model = models.Sequential()

    model.add(layers.Dense(num_units, activation='relu',
        input_shape=(n_features,))
    )
    model.add(layers.Dense(num_units, activation='relu')
    )

    compile_model(model, n_classes, optimizer=optimizer)
    return model
