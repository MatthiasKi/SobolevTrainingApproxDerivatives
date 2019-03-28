import tensorflow as tf
import numpy as np
import numpy.matlib as matlib
import math
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KDTree

# Local imports
from function_definitions import *

class Data_Handler():
  
  def __init__(self,fun_name):
    self.fun_name = fun_name
    # Get information about the function
    self.fun = get_regression_function(self.fun_name)
    self.sample_space = get_sample_space(self.fun_name)
    self.nb_inputs = get_amount_of_inputs(self.fun_name)

  def create_inputs_and_outputs(self,nb_train_samples,nb_val_samples,nb_test_samples):
    self.nb_train_samples = nb_train_samples
    self.nb_val_samples = nb_val_samples
    self.nb_test_samples = nb_test_samples
    # Create lists describing the boundaries of the sample space
    lows = []
    highs = []
    for i in range(self.nb_inputs):
      lows.append(self.sample_space[i][0])
      highs.append(self.sample_space[i][1])
    # Generate the inputs
    self.raw_training_inputs = np.float32(np.random.uniform(low=lows,high=highs,size=(self.nb_train_samples,self.nb_inputs)))
    self.raw_val_inputs = np.float32(np.random.uniform(low=lows,high=highs,size=(self.nb_val_samples,self.nb_inputs)))
    self.raw_test_inputs = np.float32(np.random.uniform(low=lows,high=highs,size=(self.nb_test_samples,self.nb_inputs)))
    # Generate the outputs
    self.raw_training_outputs = np.reshape(np.float32(np.array(self.fun(self.raw_training_inputs))),(self.nb_train_samples,1))
    self.raw_val_outputs = np.reshape(np.float32(np.array(self.fun(self.raw_val_inputs))),(self.nb_val_samples,1))
    self.raw_test_outputs = np.reshape(np.float32(np.array(self.fun(self.raw_test_inputs))),(self.nb_test_samples,1))

  def get_approx_jac_col_with_dir(self,x,dirs,epsilon):
    return ((self.fun(x+dirs*epsilon) - self.fun(x)) / np.linalg.norm(dirs*epsilon,axis=1)).numpy()

  def approximate_derivatives_stfd(self,inpu):
    inp = inpu.copy()
    raw_training_inputs = inp.copy()
    epsilon = np.float32(1e-4*np.ones(inp.shape))

    dirs = (np.zeros((inp.shape[0]*(1+inp.shape[1]),inp.shape[1],inp.shape[1]))).astype('float32')
    for dim in range(inp.shape[1]):
      dirs[0:len(inp),dim,dim] = 1.0
      raw_training_inputs = np.concatenate((raw_training_inputs,inp[0:len(inp)]+epsilon[0:len(inp)]*dirs[0:len(inp),dim]),axis=0)
      epsilon = np.concatenate((epsilon,epsilon[0:len(inp)]),axis=0)
      dirs[inp.shape[0]+int(dim*len(inp)):inp.shape[0]+int((dim+1)*len(inp)),dim,dim] = -1.
    jac = np.zeros((inp.shape[0]+(len(inp)*inp.shape[1]),inp.shape[1]))

    for dim in range(inp.shape[1]):
      mask = ~np.all(dirs[:,dim] == 0,axis=1)
      jac[mask,dim] = self.get_approx_jac_col_with_dir(raw_training_inputs[mask],dirs[mask,dim,:],epsilon[mask])

    return jac, dirs, raw_training_inputs

  def approximate_derivatives(self,inp,out):
    # This is currently only implemented for number of input dimensions equals 2!!!
    threshhold = 0.2
    number_neighbors = 5
    use_weights = True
    use_threshhold = False

    inp_np = inp.copy()
    out_np = out.copy()
    number_samples = len(inp_np)
    kdt = KDTree(inp_np,metric='euclidean')
    distances, neighbors = kdt.query(inp_np,k=number_neighbors,return_distance=True)

    approx_jac = np.zeros((number_samples,2))
    for sample_i in range(number_samples):
      if use_threshhold:
        mask = distances[sample_i,1:] < threshhold
      else:
        mask = distances[sample_i,1:] > -1. # This is always true due to the metric definition
      if use_weights:
        w = np.exp(-1.*np.linalg.norm(inp_np[neighbors[sample_i,1:]] - inp_np[sample_i],axis=1))[mask]
        W = np.diag(w / np.sum(w))
      else:
        W = np.eye(np.sum(mask))
      A = (inp_np[neighbors[sample_i,1:]] - inp_np[sample_i])[mask]
      b = (out_np[neighbors[sample_i,1:]] - out_np[sample_i])[mask]
      row = np.linalg.lstsq(a=np.matmul(W,A),b=np.matmul(W,b),rcond=-1)[0]
      approx_jac[sample_i,:] = row.reshape((len(row),))

    return approx_jac

  def get_x_scales_from_scaler(self,scaler):
    return [1.0 / s for s in scaler.scale_]

  def get_y_scale_from_scaler(self,scaler):
    return 1.0 / scaler.scale_[0]

  def create_dataset(self):
    #amt_train_samples_with_derivatives,amt_train_samples_without_derivatives,amt_val_samples,amt_test_samples,fun_name,epsilon,approximation_level='None',raw_data=(),use_sobolev_scaler=True,augment=False,der_approx_fraction=1.0
    # Check if the inputs and outputs are created
    if self.nb_train_samples is None:
      raise Exception("Please call the create_inputs_and_outputs function before creating the dataset")

    # TODO: Need to check that the handling of Numpy-Arrays versus Tensorflow-Arrays are handled correctly!

    # 1) Create the dataset for Value Training
    # Scale the input values
    self.vt_scaler_x = StandardScaler()
    self.vt_training_inputs = self.vt_scaler_x.fit_transform(self.raw_training_inputs.copy())
    self.vt_val_inputs = self.vt_scaler_x.transform(self.raw_val_inputs.copy())
    self.vt_test_inputs = self.vt_scaler_x.transform(self.raw_test_inputs.copy())

    # Scale the output values
    self.vt_scaler_y = StandardScaler()
    self.vt_training_outputs = self.vt_scaler_y.fit_transform(self.raw_training_outputs.copy())
    self.vt_val_outputs = self.vt_scaler_y.transform(self.raw_val_outputs.copy())
    self.vt_test_outputs = self.vt_scaler_y.transform(self.raw_test_outputs.copy())
    self.vt_scaler_tuple = ("Scaler",self.vt_scaler_y)

    # 2) Create the dataset for Sobolev Training with approximated derivatives (as introduced in the paper)
    # Approximate the derivatives
    self.stad_jac = self.approximate_derivatives(self.raw_training_inputs.copy(),self.raw_training_outputs.copy())
    
    # Scale the input values
    self.stad_scaler_x = StandardScaler()
    self.stad_training_inputs = self.stad_scaler_x.fit_transform(self.raw_training_inputs.copy())
    self.stad_val_inputs = self.stad_scaler_x.transform(self.raw_val_inputs.copy())
    self.stad_test_inputs = self.stad_scaler_x.transform(self.raw_test_inputs.copy())

    # Scale the output values and derivatives
    x_scales = self.get_x_scales_from_scaler(self.stad_scaler_x)
    for i, scale in enumerate(x_scales):
      self.stad_jac[:,i] = self.stad_jac[:,i] / scale 
    jac_values = self.stad_jac.flatten()
    output_shift = np.mean(self.raw_training_outputs) - np.mean(jac_values)
    shifted_raw_training_outputs = self.raw_training_outputs.copy() - output_shift
    shifted_raw_val_outputs = self.raw_val_outputs.copy() - output_shift
    shifted_raw_test_outputs = self.raw_test_outputs.copy() - output_shift
    output_jac_combined = np.expand_dims(np.append(shifted_raw_training_outputs,jac_values),1)
    self.stad_scaler_y = StandardScaler(with_mean=False).fit(output_jac_combined)
    self.stad_training_outputs = self.stad_scaler_y.fit_transform(shifted_raw_training_outputs)
    self.stad_val_outputs = self.stad_scaler_y.transform(shifted_raw_val_outputs)
    self.stad_test_outputs = self.stad_scaler_y.transform(shifted_raw_test_outputs)
    y_scale = self.get_y_scale_from_scaler(self.stad_scaler_y)
    self.stad_jac = self.stad_jac * y_scale
    self.stad_scaler_tuple = ("SobolevScaler",self.stad_scaler_y,output_shift)

    # 3) Create the dataset for Sobolev Training with approximated derivatives based on finite differences
    # As explained in the paper, we will use only one third of the original training data for this approach (as new training data is created during the approximation of derivatives)
    stfd_nb_train_samples = int(np.floor(len(self.raw_training_inputs) / (1. + self.nb_inputs)))
    stfd_raw_training_inputs = self.raw_training_inputs[0:stfd_nb_train_samples].copy()
    stfd_raw_training_outputs = self.raw_training_outputs[0:stfd_nb_train_samples].copy()
    
    # Approximate the derivatives
    self.stfd_jac, self.stfd_dirs, stfd_raw_training_inputs = self.approximate_derivatives_stfd(stfd_raw_training_inputs)
    stfd_raw_training_outputs = np.reshape(np.float32(np.array(self.fun(stfd_raw_training_inputs))),(len(stfd_raw_training_inputs),1))
    
    # Scale the input values
    self.stfd_scaler_x = StandardScaler()
    self.stfd_training_inputs = self.stfd_scaler_x.fit_transform(stfd_raw_training_inputs)
    self.stfd_val_inputs = self.stfd_scaler_x.transform(self.raw_val_inputs.copy())
    self.stfd_test_inputs = self.stfd_scaler_x.transform(self.raw_test_inputs.copy())

    # Scale the output values and derivatives
    x_scales = self.get_x_scales_from_scaler(self.stfd_scaler_x)
    mask = ~np.all(self.stfd_dirs==0,axis=2)
    for i, scale in enumerate(x_scales):
      self.stfd_jac[:,i] = self.stfd_jac[:,i] / scale
    jac_values = self.stfd_jac[mask]
    output_shift = np.mean(stfd_raw_training_outputs) - np.mean(jac_values)
    shifted_raw_training_outputs = stfd_raw_training_outputs - output_shift 
    shifted_raw_val_outputs = self.raw_val_outputs.copy() - output_shift
    shifted_raw_test_outputs = self.raw_test_outputs.copy() - output_shift
    output_jac_combined = np.expand_dims(np.append(shifted_raw_training_outputs,jac_values),1)
    self.stfd_scaler_y = StandardScaler(with_mean=False).fit(output_jac_combined)
    self.stfd_training_outputs = self.stfd_scaler_y.transform(shifted_raw_training_outputs)
    self.stfd_val_outputs = self.stfd_scaler_y.transform(shifted_raw_val_outputs)
    self.stfd_test_outputs = self.stfd_scaler_y.transform(shifted_raw_test_outputs)
    y_scale = self.get_y_scale_from_scaler(self.stfd_scaler_y)
    self.stfd_jac = self.stfd_jac * y_scale
    self.stfd_scaler_tuple = ("SobolevScaler",self.stfd_scaler_y,output_shift)