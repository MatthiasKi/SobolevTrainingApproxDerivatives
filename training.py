import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

class Training():

  def __init__(self,data_handler):
    self.rho_up = 0.95
    self.activation = "relu"
    self.loss_function = 'mse'
    self.target_batches_amount = 10
    self.neuron_amount = 200
    self.shuffle = True
    self.patience = 10
    self.data_handler = data_handler

  def get_model(self,act,inputs,neuron_amount):
    model = tf.keras.Sequential([
      tf.keras.layers.Dense(neuron_amount, input_shape=(inputs,),activation=act),
      tf.keras.layers.Dense(neuron_amount,activation=act), 
      tf.keras.layers.Dense(1)])
    return model

  def grad(self, model, inputs, targets):
    with tf.GradientTape() as tape:
      prediction = model(inputs)
      loss_value = tf.losses.mean_squared_error(targets,prediction)
    return tape.gradient(loss_value, model.variables)

  def sobolev_loss(self,target_output,model,inputs,jac,target_jac,rho_factor):
    targets = target_output.copy()
    outputs =  model(inputs)

    tmp_shape = (len(target_jac[:,0]),1)
    weights = np.array([1.0 for i in range(tmp_shape[0])])
    for i in range(np.shape(inputs.numpy())[1]):
      targets = np.append(targets,np.reshape(target_jac[:,i],tmp_shape))
      outputs = tf.concat((outputs,tf.reshape(jac[:,i],tmp_shape)),axis=0)
      weights = np.append(weights,np.array([rho_factor for i in range(tmp_shape[0])]))
    targets = np.reshape(targets,np.shape(outputs.numpy()))
    weights = np.reshape(weights,np.shape(outputs.numpy()))
    return tf.losses.mean_squared_error(targets,outputs,weights=weights)

  def get_jacobian_tf(self,fun,inp):
    with tf.GradientTape() as tape: 
      output = fun(inp)
    return tape.gradient(output,inp)

  def get_jacobian_with_dirs_tf(self,model,inputs,dirs):
    jac = self.get_jacobian_tf(model,inputs)
    # NOTE: This step does only work with single-step directions!! I.e. a direction vector like [1,1] DOES NOT WORK!
    dir_jac = tf.expand_dims(jac,-1) * dirs
    dir_jac = tf.reduce_sum(dir_jac,1)
    return dir_jac
    
  def stad_grad(self,model,inputs,targets,jac,rho_factor):
    inp = tf.contrib.eager.Variable(inputs)
    targets = targets.numpy()
    target_jac = jac.numpy()

    with tf.GradientTape() as tape:
      curr_jac = self.get_jacobian_tf(model,inp)
      loss_value = self.sobolev_loss(targets,model,inp,curr_jac,target_jac,rho_factor)
    return tape.gradient(loss_value,model.variables)

  def stfd_grad(self,model,inputs,targets,jac,rho_factor,dirs):
    inp = tf.contrib.eager.Variable(inputs)
    targets = targets.numpy()
    target_jac = jac.numpy()

    with tf.GradientTape() as tape:
      curr_jac = self.get_jacobian_with_dirs_tf(model,inp,dirs)
      loss_value = self.sobolev_loss(targets,model,inp,curr_jac,target_jac,rho_factor)
    return tape.gradient(loss_value,model.variables)

  def backscale(self,y,scaler_tuple):
    if scaler_tuple[0] == '0':
      return y
    elif scaler_tuple[0] == 'Scaler':
      return scaler_tuple[1].inverse_transform(y)
    elif scaler_tuple[0] == 'SobolevScaler':
      return scaler_tuple[1].inverse_transform(y) + scaler_tuple[2]
    else:
      raise Exception("Scaling mode not recognized")
      
  def backscaled_loss(self,model,x,raw_y,scaler_tuple):
    if scaler_tuple[0] == '0':
      prediction = model(x)
      return tf.losses.mean_squared_error(raw_y,prediction)
    else:
      prediction = self.backscale(model(x),scaler_tuple)
      return tf.losses.mean_squared_error(raw_y,prediction)

  # Training using Value Training
  def train_vt(self):
    # Calculate the needed batch size
    batch_size = int(math.ceil(self.data_handler.nb_train_samples / self.target_batches_amount))

    # Create list used for storing the results
    test_loss_backscaled = []
    val_loss_backscaled = []

    # Set up the model and optimizer
    model = self.get_model(self.activation,self.data_handler.nb_inputs,self.neuron_amount)
    optimizer = tf.train.AdamOptimizer()

    # Create the unbatched training dataset
    training_dataset_tuple = (self.data_handler.vt_training_inputs.copy(),self.data_handler.vt_training_outputs.copy())
    training_dataset_unbatched = tf.data.Dataset.from_tensor_slices(training_dataset_tuple)

    # Train until stopping criterion is met
    continue_training = True
    while continue_training:
      if self.shuffle == True:
        training_dataset = training_dataset_unbatched.shuffle(self.data_handler.nb_train_samples)
        training_dataset = training_dataset.batch(batch_size)
      else:
        training_dataset = training_dataset_unbatched.batch(batch_size)
      # Start Epoch
      for dataset_slice in training_dataset:
        x = dataset_slice[0]
        y = dataset_slice[1]
        # Optimize the model
        grads = self.grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.variables),global_step=tf.train.get_or_create_global_step())
      # End Epoch

      # Store some stats
      val_loss_backscaled.append(self.backscaled_loss(model,self.data_handler.vt_val_inputs.copy(),self.data_handler.raw_val_outputs.copy(),self.data_handler.vt_scaler_tuple).numpy())
      test_loss_backscaled.append(self.backscaled_loss(model,self.data_handler.vt_test_inputs.copy(),self.data_handler.raw_test_outputs.copy(),self.data_handler.vt_scaler_tuple).numpy())

      # Check if Stopping criterion is met:
      if np.argmin(np.array(val_loss_backscaled)) < len(val_loss_backscaled) - self.patience:
        continue_training = False

    # Save the results
    self.vt_val_loss_backscaled = np.array(val_loss_backscaled)
    self.vt_test_loss_backscaled = np.array(test_loss_backscaled)

  # Training Function using Sobolev Training with Least Squares Estimated Derivatives
  def train_stad(self):
    # Calculate the needed batch size
    batch_size = int(math.ceil(self.data_handler.nb_train_samples / self.target_batches_amount))
    rho_factor = 1.0

    # Create list used for storing the results
    test_loss_backscaled = []
    val_loss_backscaled = []

    # Set up the model and optimizer
    model = self.get_model(self.activation,self.data_handler.nb_inputs,self.neuron_amount)
    optimizer = tf.train.AdamOptimizer()

    # Create the unbatched training dataset
    training_dataset_tuple = (self.data_handler.stad_training_inputs.copy(),self.data_handler.stad_training_outputs.copy(),self.data_handler.stad_jac.copy())
    training_dataset_unbatched = tf.data.Dataset.from_tensor_slices(training_dataset_tuple)

    # Train until stopping criterion is met
    continue_training = True
    while continue_training:
      if self.shuffle == True:
        training_dataset = training_dataset_unbatched.shuffle(self.data_handler.nb_train_samples)
        training_dataset = training_dataset.batch(batch_size)
      else:
        training_dataset = training_dataset_unbatched.batch(batch_size)
      # Start Epoch
      for dataset_slice in training_dataset:
        x = dataset_slice[0]
        y = dataset_slice[1]
        jac = dataset_slice[2]
        # Optimize the model
        grads = self.stad_grad(model,x,y,jac,rho_factor)
        optimizer.apply_gradients(zip(grads, model.variables),global_step=tf.train.get_or_create_global_step())
      # End Epoch
      # Adapt the rho factor
      rho_factor = rho_factor*self.rho_up

      # Store some stats
      val_loss_backscaled.append(self.backscaled_loss(model,self.data_handler.stad_val_inputs.copy(),self.data_handler.raw_val_outputs.copy(),self.data_handler.stad_scaler_tuple).numpy())
      test_loss_backscaled.append(self.backscaled_loss(model,self.data_handler.stad_test_inputs.copy(),self.data_handler.raw_test_outputs.copy(),self.data_handler.stad_scaler_tuple).numpy())

      # Check if Stopping criterion is met:
      if np.argmin(np.array(val_loss_backscaled)) < len(val_loss_backscaled) - self.patience:
        continue_training = False

    # Save the results
    self.stad_val_loss_backscaled = np.array(val_loss_backscaled)
    self.stad_test_loss_backscaled = np.array(test_loss_backscaled)

  # Training Function using Sobolev Training with Finite Differences Derivatives
  def train_stfd(self):
    # Calculate the needed batch size
    batch_size = int(math.ceil(self.data_handler.nb_train_samples / self.target_batches_amount))
    rho_factor = 1.0

    # Create list used for storing the results
    test_loss_backscaled = []
    val_loss_backscaled = []

    # Set up the model and optimizer
    model = self.get_model(self.activation,self.data_handler.nb_inputs,self.neuron_amount)
    optimizer = tf.train.AdamOptimizer()

    # Create the unbatched training dataset
    training_dataset_tuple = (self.data_handler.stfd_training_inputs.copy(),self.data_handler.stfd_training_outputs.copy(),self.data_handler.stfd_jac.copy(),self.data_handler.stfd_dirs.copy())
    training_dataset_unbatched = tf.data.Dataset.from_tensor_slices(training_dataset_tuple)

    # Train until stopping criterion is met
    continue_training = True
    while continue_training:
      if self.shuffle == True:
        training_dataset = training_dataset_unbatched.shuffle(self.data_handler.nb_train_samples)
        training_dataset = training_dataset.batch(batch_size)
      else:
        training_dataset = training_dataset_unbatched.batch(batch_size)
      # Start Epoch
      for dataset_slice in training_dataset:
        x = dataset_slice[0]
        y = dataset_slice[1]
        jac = dataset_slice[2]
        dirs = dataset_slice[3]
        # Optimize the model
        grads = self.stfd_grad(model,x,y,jac,rho_factor,dirs)
        optimizer.apply_gradients(zip(grads, model.variables),global_step=tf.train.get_or_create_global_step())
      # End Epoch
      # Adapt the rho factor
      rho_factor = rho_factor*self.rho_up

      # Store some stats
      val_loss_backscaled.append(self.backscaled_loss(model,self.data_handler.stfd_val_inputs.copy(),self.data_handler.raw_val_outputs.copy(),self.data_handler.stfd_scaler_tuple).numpy())
      test_loss_backscaled.append(self.backscaled_loss(model,self.data_handler.stfd_test_inputs.copy(),self.data_handler.raw_test_outputs.copy(),self.data_handler.stfd_scaler_tuple).numpy())

      # Check if Stopping criterion is met:
      if np.argmin(np.array(val_loss_backscaled)) < len(val_loss_backscaled) - self.patience:
        continue_training = False

    # Save the results
    self.stfd_val_loss_backscaled = np.array(val_loss_backscaled)
    self.stfd_test_loss_backscaled = np.array(test_loss_backscaled)

  def plot_results(self):
    if not hasattr(self,'vt_test_loss_backscaled') or not hasattr(self,'stad_test_loss_backscaled') or not hasattr(self,'stfd_test_loss_backscaled'):
      print("The training functions must be called before results can be plotted!")
    else:
      plt.figure()
      plt.title(self.data_handler.fun_name)
      plt.semilogy(range(len(self.vt_test_loss_backscaled)),self.vt_test_loss_backscaled,label="Value Training Test Loss")
      plt.semilogy(range(len(self.stad_test_loss_backscaled)),self.stad_test_loss_backscaled,label="Sobolev Training (Least Squares Derivatives)")
      plt.semilogy(range(len(self.stfd_test_loss_backscaled)),self.stfd_test_loss_backscaled,label="Sobolev Training (Finite Differences)")
      plt.legend(loc='best')
      plt.show()