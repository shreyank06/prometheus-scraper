import tensorflow as tf
from .baseline import Baseline
from .autoregressive_model.feedback import FeedBack
import os
import keras_tuner as kt
# from kerastuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
import pandas as pd
tf.compat.v1.experimental.output_all_intermediates(True)
tf.get_logger().setLevel('ERROR')

class Models:
    MAX_EPOCHS = 700

    def __init__(self, column_indices, window_size, num_features, config, component):
        self.column_indices = column_indices
        self.window_size = window_size
        self.num_features = num_features
        self.config = config
        self.hp = HyperParameters()
        self.model = None
        self.component = component
        
        
    def create_baseline_model(self):
        baseline = Baseline(label_index=self.column_indices['phoenix_memory_used_cm_sessionP_smf'])
        baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                         metrics=[tf.keras.metrics.MeanAbsoluteError()])
        
        val_performance = {}
        performance = {}
        val_performance['Baseline'] = baseline.evaluate(self.window_size.val)
        performance['Baseline'] = baseline.evaluate(self.window_size.test, verbose=0)

        self.window_size.plot(baseline)

    
    def linear_model(self):
        linear = tf.keras.Sequential([
            tf.keras.layers.Dense(units=1)
        ])
        return linear
    
    def densed_model(self):
        dense = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])
        return dense
    
    def multi_step_densed_model(self, wide_window, num_features):

        multi_step_dense_model = tf.keras.Sequential([
            # Take the last time step.
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            # Shape => [batch, 1, dense_units]
            tf.keras.layers.Dense(512, activation='relu'),
            # Shape => [batch, out_steps*features]
            tf.keras.layers.Dense(wide_window.label_width*num_features,
                                kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([wide_window.label_width, num_features])
        ])

        return multi_step_dense_model
    
    def convolutional_model(self, wide_conv_window, num_features):
        conv_model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=32,
                                kernel_size=(wide_conv_window.input_width,),
                                activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=24),  # Adjust units to match the desired output shape
            tf.keras.layers.Reshape([wide_conv_window.label_width, num_features])  # Reshape to (batch_size, 24, 1)
        ])
        return conv_model
    
    def lstm_model(self):
        lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
        ])
        return lstm_model

    def multi_step_linear_single_shot(self, wide_window, num_features):
        multi_linear_model = tf.keras.Sequential([
        # Take the last time-step.
        # Shape [batch, time, features] => [batch, 1, features]
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        # Shape => [batch, 1, out_steps*features]
        tf.keras.layers.Dense(wide_window.label_width*num_features,
                            kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([wide_window.label_width, num_features])
    ])
        return multi_linear_model

    def multi_step_convolutional_model(self, wide_window, num_features):
        CONV_WIDTH = 3
        multi_conv_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
        tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
        # Shape => [batch, 1, conv_units]
        tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
        # Shape => [batch, 1,  out_steps*features]
        tf.keras.layers.Dense(wide_window.label_width*num_features,
                            kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([wide_window.label_width, num_features])
    ])
        return multi_conv_model
    
    def multi_step_lstm_model(self, wide_window, num_features):
        units = self.hp.Int('units', min_value=16, max_value=128, step=16) 
        multi_lstm_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, lstm_units].
        # Adding more `lstm_units` just overfits more quickly.
        tf.keras.layers.LSTM(units, return_sequences=False),
        # Shape => [batch, out_steps*features].
        tf.keras.layers.Dense(self.window_size.label_width*self.num_features,
                            kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features].
        tf.keras.layers.Reshape([self.window_size.label_width, self.num_features])
    ])
        return multi_lstm_model

    def autoregressive_lstm(self, wide_window, num_features):

        autoregressive_feedback_lstm = FeedBack(units=32, out_steps=wide_window.input_width, num_features=num_features)
        return autoregressive_feedback_lstm
    
    def build_model(self, hp):
        
        optimizer = self.hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'])
        
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(),
                      optimizer=optimizer, #tf.keras.optimizers.legacy.Adam(),
                      metrics=[tf.keras.metrics.MeanAbsoluteError()])
        return self.model


   # def hyperparameter_tuning(self, model_fn, model_type):
    def compile_and_fit(self, model, model_type,model_path, patience=10):
        
        # MAX_TRIALS = 20
        # EXECUTIONS_PER_TRIAL = 5
        # RANDOM_SEED = 42
        # tuner = kt.RandomSearch(
        #         self.build_model,
        #         objective='val_accuracy',
        #         max_trials=MAX_TRIALS,
        #         executions_per_trial=EXECUTIONS_PER_TRIAL,
        #         directory='test_dir',
        #         project_name='tune_optimizer',
        #         seed=RANDOM_SEED
        #         )
        # tuner.search(self.window_size.train,
        #              validation_data=self.window_size.val,
        #              epochs=self.MAX_EPOCHS)
        # #tuner.results_summary()
        # sys.exit()
        # best_hyperparameters = tuner.get_best_hyperparameters()[0]
        # best_model = tuner.hypermodel.build(best_hyperparameters)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                           patience=patience,
                                                           mode='min')
        
        hp_learning_rate = self.hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        optimizer = self.hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'])

        
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                      optimizer=optimizer, #tf.keras.optimizers.legacy.Adam(),
                      metrics=[tf.keras.metrics.MeanAbsoluteError()])

        history = model.fit(self.window_size.train, epochs=self.MAX_EPOCHS,
                            validation_data=self.window_size.val,
                            callbacks=[early_stopping])
        
        # Retrieve MAE values from training and validation
        # train_mae = history.history['mean_absolute_error']
        # val_mae = history.history['val_mean_absolute_error']

        # Example: Print the last MAE values for decision-making
        # print(f'Final Training MAE: {train_mae[-1]}')
        # print(f'Final Validation MAE: {val_mae[-1]}')
        # sys.exit()
        # Save the trained model with the model type as the filename
        save_dir = os.path.join(os.getcwd(), model_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model.save(os.path.join(save_dir, f'{model_type}.h5'))

        return history

    def performance_evaluation(self, model_type, wide_window):
        val_performance = {}
        performance = {}
        model = None
        model_filename = f"{model_type}_{self.component}.h5"
        model_path = os.path.join("trained_models", self.component, model_filename)
        if os.path.exists(model_path):
            # If the model file already exists, load it
            model = tf.keras.models.load_model(os.path.join(model_path, f'{model_type}.h5'))
            #sys.exit()
            if self.config['retrain_model']:
                if model.layers[-1].output_shape[1] == self.window_size.train.element_spec[0].shape[1]:
                    history = self.compile_and_fit(model, model_type, model_path)
                    #model = tf.keras.models.load_model(model_path)
                    results = self.window_size.plot(dataset='val', model=model)
                    val_performance[model_type] = model.evaluate(self.window_size.train)
                    performance[model_type] = model.evaluate(self.window_size.val, verbose=0)
                    # print("Plotting accuracy")
                    # history_df = pd.DataFrame(history.history)
                    # history_df.loc[:, ['mean_absolute_error', 'val_mean_absolute_error']].plot()
                    #plt.show()

                else:
                    print('Model is not trained for desired output shape, creating new model and training again')
                    # Continue with creating a new model
                    model = self.create_model(model_type, wide_window)
                    history = self.compile_and_fit(model, model_type)
                    model.save(model_path)
                    results = self.window_size.plot(dataset='test', model=model)
                    #self.performance_evaluation(model_type, wide_window)
                    history_df = pd.DataFrame(history.history)
                    history_df.loc[:, ['loss', 'val_loss']].plot()
                    #plt.show()
            else:
                if model.layers[-1].output_shape[1] == self.window_size.train.element_spec[0].shape[1]:
                    #model = tf.keras.models.load_model(model_path)
                    #sys.exit()
                    results = self.window_size.plot(dataset='train', model=model)
                    val_performance[model_type] = model.evaluate(self.window_size.train)
                    performance[model_type] = model.evaluate(self.window_size.val, verbose=0)
                else:
                    print('Model is not trained for desired output shape, creating new model and training again')
                    # Continue with creating a new model
                    model = self.create_model(model_type, wide_window)
                    history = self.compile_and_fit(model, model_type)
                    model.save(model_path)
                    self.performance_evaluation(model_type, wide_window)
                    history_df = pd.DataFrame(history.history)
                    history_df.loc[:, ['loss', 'val_loss']].plot()
                    #plt.show()
        else:
            # Otherwise, create a new model based on the model type
            print("The model doesnt exist, creating new model")
            model = self.create_model(model_type, wide_window)
            #best_model = self.hyperparameter_tuning(lambda hp: self.create_model(model_type, wide_window, hp=hp), model_type)
            history = self.compile_and_fit(model, model_type, model_path)
            history_df = pd.DataFrame(history.history)
            history_df.loc[:, ['loss', 'val_loss']].plot()
            #history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
            #plt.show()

            #val_performance[model_type] = model.evaluate(self.window_size.val)
            #performance[model_type] = model.evaluate(self.window_size.test, verbose=0)
            results = self.window_size.plot(dataset='val', model=model)
        #return val_performance, performance
        return val_performance, performance, results
    
    def create_model(self, model_type, wide_window, hp = None):
        if model_type == 'linear':
            model = self.linear_model()
        elif model_type == 'densed':
            model = self.densed_model()
        elif model_type == 'convolutional_model':
            model = self.convolutional_model(wide_window, self.num_features)
        elif model_type == 'lstm_model':
            model = self.lstm_model()
        elif model_type == 'single_shot_linear':
            model = self.multi_step_linear_single_shot(wide_window, self.num_features)
        elif model_type == 'multi_step_densed':
            model = self.multi_step_densed_model(wide_window, self.num_features)
        elif model_type == 'multi_step_conv':
            model = self.multi_step_convolutional_model(wide_window, self.num_features)
        elif model_type == 'multi_step_lstm':
            model = self.multi_step_lstm_model(wide_window, self.num_features)
        elif model_type == 'autoregressive_lstm':
            model = self.autoregressive_lstm(wide_window, self.num_features)
        return model
