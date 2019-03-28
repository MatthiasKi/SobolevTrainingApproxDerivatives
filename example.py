import tensorflow as tf

# Local Imports
from Data_Handler import Data_Handler
from training import Training

tf.enable_eager_execution()

# IMPLEMENTED FUNCTIONS
# Bowl-Shaped: ['Sphere','SumOfDifferentPowers','SumSquares','Trid']
# Plate-Shaped: ['Booth','McCormick','Matyas','Powersum']
# Valley-Shaped: ['Rosenbrock','ThreeHumpCamel','SixHumpCamel','DixonPrice']
# Steep Ridges/Drops: ['Easom','Michalewicz']
# Others: ['StyblinskiTang','Beale','Branin','GoldsteinPrice']

# Hyperparameters
NUM_TRAINING_EXAMPLES = 1000
NUM_VALIDATION_EXAMPLES = 4000
fun_name = 'SumSquares'
# ---

# Create Data for Training
dh = Data_Handler(fun_name=fun_name)
dh.create_inputs_and_outputs(nb_train_samples=NUM_TRAINING_EXAMPLES,nb_val_samples=NUM_VALIDATION_EXAMPLES,nb_test_samples=NUM_VALIDATION_EXAMPLES)
dh.create_dataset()

# Start Training
trainer = Training(dh)
trainer.train_vt()
trainer.train_stad()
trainer.train_stfd()

# Plot the Data
trainer.plot_results()