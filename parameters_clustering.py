train_loss_func = 'mse'
test_loss_func = 'mse'

# introduce a bit of dirty training data to investigate the effect of anomalous training data 
dirty_data_percentage = 0

epochs = 15

network = 'dense'

encoder_hidden_neuron_sizes=[512, 256, 128, 64, 32, 16, 8]