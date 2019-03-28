# IMPLEMENTED FUNCTIONS
# Definitions of the Functions: https://www.sfu.ca/~ssurjano/optimization.html
# Bowl-Shaped: ['Sphere','SumOfDifferentPowers','SumSquares','Trid']
# Plate-Shaped: ['Booth','McCormick','Matyas','Powersum']
# Valley-Shaped: ['Rosenbrock','ThreeHumpCamel','SixHumpCamel','DixonPrice']
# Steep Ridges/Drops: ['Easom','Michalewicz']
# Others: ['StyblinskiTang','Beale','Branin','GoldsteinPrice']

import tensorflow as tf
import numpy as np
import numpy.matlib as matlib
import math

def Easom(x):
  # orig: -cos(x_1)*cos(x_2)*exp(-(x_1-pi)^2-(x_2-pi)^2)
  x = tf.to_double(x)
  in_exp = - tf.pow(x[:,0] - np.pi,2) - tf.pow(x[:,1] - np.pi,2)
  return -tf.cos(x[:,0])*tf.cos(x[:,1])*tf.exp(in_exp)

def Michalewicz(x):
  # orig: - sum_i^n (sin(x_i)sin^20(i*x_i^2/pi))
  x = tf.to_double(x)
  out = tf.to_double(tf.constant(0,shape=[np.array(tf.shape(x)[0]),]))
  n = tf.to_double(tf.shape(x)[1])
  for i in range(n):
    out += tf.sin(x[:,i])*tf.pow(tf.sin((i+1)/np.pi*tf.pow(x[:,i],2)),20)
  return -1*out

def ThreeHumpCamel(x):
  # orig: 2*x_1^2 -1.05*x_1^4 + x_1^6 /6.0 + x_1*x_2 + x_2^2
  x = tf.to_double(x)
  power_2 = tf.to_double(tf.constant(2,shape=[np.array(tf.shape(x)[0]),]))
  power_4 = tf.to_double(tf.constant(4,shape=[np.array(tf.shape(x)[0]),]))
  power_6 = tf.to_double(tf.constant(6,shape=[np.array(tf.shape(x)[0]),]))
  return 2.0*tf.pow(x[:,0],power_2) - 1.05*tf.pow(x[:,0],power_4) + tf.divide(tf.pow(x[:,0],power_6),6.0) + x[:,0]*x[:,1] + tf.pow(x[:,1],power_2)

def SixHumpCamel(x):
  # orig: (4-2.1*x_1^2+x_1^4/3)*x_1^2 + x_1*x_2 + (-4+4*x_2^2)*x_2^2
  x = tf.to_double(x)
  power_2 = tf.to_double(tf.constant(2,shape=[np.array(tf.shape(x)[0]),]))
  power_4 = tf.to_double(tf.constant(4,shape=[np.array(tf.shape(x)[0]),]))
  add1 = tf.add(tf.subtract(tf.divide(tf.pow(x[:,0],power_4),3),2.1*tf.pow(x[:,0],power_2)),4.0)*tf.pow(x[:,0],power_2)
  add2 = (tf.add(4.0*tf.pow(x[:,1],power_2),-4.0))*tf.pow(x[:,1],power_2)
  return add1 + x[:,0]*x[:,1] + add2

def DixonPrice(x):
  # orig: (x_1)^2 + sum_i=2^n[i*(2*x_i^2-x_(i-1))^2]
  x = tf.to_double(x)
  power_2 = tf.to_double(tf.constant(2,shape=[np.array(tf.shape(x)[0]),]))
  add1 = tf.pow(x[:,0],power_2)
  add2 = tf.to_double(tf.constant(0,shape=[np.array(tf.shape(x)[0]),]))
  for i in range(1,tf.shape(x).numpy()[1]):
    add2 = tf.add(add2,i*tf.pow(2*tf.pow(x[:,i],power_2)-x[:,i-1],power_2))
  return add1 + add2

def Sphere(x):
  # orig: sum(x_i^2)
  x = tf.to_double(x)
  power_2 = tf.to_double(tf.constant(2,shape=tf.shape(x).numpy()))
  return tf.reduce_sum(tf.pow(x,power_2),1)

def SumOfDifferentPowers(x):
  # orig: sum_i^n |x_i|^(i+1)
  x = tf.to_double(x)
  power = np.matlib.repmat([(i+2) for i in range(np.shape(x)[1])],np.shape(x)[0],1)
  powered = tf.pow(tf.abs(x),power)
  return tf.reduce_sum(powered,1)

def SumSquares(x):
  # orig: sum_i^n i*x_i^2
  x = tf.to_double(x)
  power_2 = tf.to_double(tf.constant(2,shape=tf.shape(x).numpy()))
  powered = tf.pow(x,power_2)
  multi = np.matlib.repmat([(i+1) for i in range(np.shape(x)[1])],np.shape(x)[0],1)
  return tf.reduce_sum(powered*multi,1)

def Trid(x):
  # orig: sum_i^n [(x_i-1)^2] - sum_i^n [x_i*x_(i-1)]
  x = tf.to_double(x)
  sub = tf.to_double(tf.constant(1,shape=tf.shape(x).numpy()))
  power_2 = tf.to_double(tf.constant(2,shape=tf.shape(x).numpy()))
  add1 = tf.pow(x-sub,power_2)
  add2 = tf.to_double(tf.constant(0,shape=[np.array(tf.shape(x)[0]),]))
  for i in range(1,tf.shape(x).numpy()[1]):
    add2 = tf.add(add2,x[:,i]*x[:,i-1])
  return tf.reduce_sum(add1,1) - add2

def Beale(x):
  # orig: square(1.5 - x_1 + x_1*x_2) + square(2.25 - x_1 + x_1*square(x_2)) + square(2.625 - x_1 + x_1*cube(x_2))
  x = tf.to_double(x) 
  power1 = tf.to_double(tf.constant(2,shape=[np.array(tf.shape(x)[0]),]))
  add1 = tf.pow(tf.add(-1.0*x[:,0] + x[:,0]*x[:,1],1.5),power1)
  square1 = tf.pow(x[:,1],power1)
  add2 = tf.pow(tf.add(-1.0*x[:,0] + x[:,0]*square1,2.25),power1)
  power2 = tf.to_double(tf.constant(3,shape=[np.array(tf.shape(x)[0]),]))
  cube1 = tf.pow(x[:,1],power2)
  add3 = tf.pow(tf.add(-1.0*x[:,0] + x[:,0] * cube1,2.625),power1)
  return add1 + add2 + add3

def Branin(x):
  #orig: a(x_2-b*x_1^2+c*x_1-r)^2 + s(1-t)*cos(x_1)+s
  # with:
  a = 1.0
  b = 5.1 / (4*np.power(np.pi,2))
  c = 5.0 / np.pi
  r = 6.0
  s = 10.0
  t = 1.0 / (8*np.pi)
  x = tf.to_double(x) 
  power2 = tf.to_double(tf.constant(2,shape=[np.array(tf.shape(x)[0]),]))
  r_vec = tf.to_double(tf.constant(r,shape=[np.array(tf.shape(x)[0]),]))
  s_vec = tf.to_double(tf.constant(s,shape=[np.array(tf.shape(x)[0]),]))
  return a*tf.pow(x[:,1]-b*tf.pow(x[:,0],power2)+c*x[:,0]-r_vec,power2) + s*(1-t)*tf.cos(x[:,0]) + s_vec

def GoldsteinPrice(x):
  #orig: [1+(x_1+x_2+1)^2*(19-14x_1+3x_1^2-14x_2+6x_1x_2+3x_2^2)]*[30+(2x_1-3x_2)^2*(18-32x_1+12x_1^2+48x_2-36x_1x_2+27x_2^2)]
  x = tf.to_double(x)
  ones = tf.to_double(tf.constant(1,shape=[np.array(tf.shape(x)[0]),]))
  mul1 =  ones + tf.pow(x[:,0]+x[:,1]+ones,2)*(19*ones-14*x[:,0]+3*tf.pow(x[:,0],2)-14*x[:,1]+6*x[:,0]*x[:,1]+3*tf.pow(x[:,1],2))
  mul2 = 30*ones + tf.pow(2*x[:,0]-3*x[:,1],2) * (18*ones - 32 *x[:,0] + 12*tf.pow(x[:,0],2) + 48*x[:,1] - 36*x[:,0]*x[:,1] + 27*tf.pow(x[:,1],2)) 
  return mul1*mul2

def StyblinskiTang(x):
  # orig: 0.5*sum(pow(x_i,4)-16*square(x_i) + 5*x_i)
  x = tf.to_double(x)
  power1 = tf.to_double(tf.constant(4,shape=(tf.shape(x)).numpy()))
  mod1 = tf.pow(x,power1)
  power2 = tf.to_double(tf.constant(2,shape=(tf.shape(x)).numpy()))
  mod2 = tf.pow(x,power2)
  raw = mod1 - 16.0 * mod2 + 5.0 * x
  summed = tf.reduce_sum(raw,1)
  return 0.5*summed

def Booth(x):
  # orig: math.pow(x[0]+2*x[1]-7,2)+math.pow(2*x[0]+x[1]-5,2)
  x = tf.to_double(x)
  power = tf.to_double(tf.constant(2,shape=[np.array(tf.shape(x)[0]),]))
  add1 = tf.subtract(x[:,0]+2*x[:,1],7.2)
  add1 = tf.pow(add1,power)
  add2 = tf.subtract(2*x[:,0]+x[:,1],5.2)
  add2 = tf.pow(add2,power)
  return add1+add2

def Matyas(x):
  # orig: 0.26(x_1^2+x_2^2)-0.48*x_1*x_2
  x = tf.to_double(x)
  power = tf.to_double(tf.constant(2,shape=[np.array(tf.shape(x)[0]),]))
  add1 = 0.26*tf.add(tf.pow(x[:,0],power),tf.pow(x[:,1],power))
  add2 = 0.48*x[:,0]*x[:,1]
  return add1 - add2

def Powersum(x):
  # orig: sum_i(sum_j(x_j^i)-b_i)^2 with b = (8,18,44,114)
  x = tf.to_double(x)
  b =  [8,18,44,114]
  result =  tf.to_double(tf.constant(0,shape=[np.array(tf.shape(x)[0]),]))
  power_2 = tf.to_double(tf.constant(2,shape=[np.array(tf.shape(x)[0]),]))
  for i in range(get_amount_of_inputs("Powersum")):
    tmp = tf.to_double(tf.constant(0,shape=[np.array(tf.shape(x)[0]),]))
    power_tmp = tf.to_double(tf.constant(float(i+1),shape=[np.array(tf.shape(x)[0]),]))
    for j in range(get_amount_of_inputs("Powersum")):
      tmp = tf.add(tmp,tf.pow(x[:,j],power_tmp))
    tmp = tf.subtract(tmp,b[i])
    result = tf.add(result,tf.pow(tmp,power_2))
  return result

def McCormick(x):
  # orig: math.sin(x[0]+x[1]) + math.pow(x[0]-x[1],2) - 1.5*x[0] + 2.5*x[1] + 1
  x = tf.to_double(x)
  power = tf.to_double(tf.constant(2,shape=[np.array(tf.shape(x)[0]),]))
  add1 = tf.sin(x[:,0]+x[:,1]) 
  add2 = tf.pow(x[:,0]+x[:,1],power)
  add3 = tf.add(-1.5*x[:,0] + 2.5*x[:,1],1.0)
  return add1 + add2 + add3

def Rosenbrock(x):
  # orig: 100*math.pow(x[1]-math.pow(x[0],2),2) + math.pow(x[0]-1,2)
  x = tf.to_double(x)
  power = tf.to_double(tf.constant(2,shape=[np.array(tf.shape(x)[0]),]))
  pow1 = tf.pow(x[:,0],power)
  add1 = 100.0*tf.pow(x[:,1]-pow1,power)
  pow2 = tf.pow(tf.subtract(x[:,0],1.2),power)
  return add1 - pow2

def get_regression_function(fun):
  if fun == 'Booth':
    return Booth
  elif fun == 'Matyas':
    return Matyas
  elif fun == 'Powersum':
    return Powersum
  elif fun == 'ThreeHumpCamel':
    return ThreeHumpCamel
  elif fun == 'SixHumpCamel':
    return SixHumpCamel
  elif fun == 'Sphere':
    return Sphere
  elif fun == 'SumOfDifferentPowers':
    return SumOfDifferentPowers
  elif fun == 'SumSquares':
    return SumSquares
  elif fun == 'Trid':
    return Trid
  elif fun == 'McCormick':
    return McCormick
  elif fun == 'Rosenbrock':
    return Rosenbrock
  elif fun == 'DixonPrice':
    return DixonPrice
  elif fun == 'StyblinskiTang':
    return StyblinskiTang
  elif fun == 'Beale':
    return Beale
  elif fun == 'Branin':
    return Branin
  elif fun == 'GoldsteinPrice':
    return GoldsteinPrice
  elif fun == 'Easom':
    return Easom
  elif fun == 'Michalewicz':
    return Michalewicz
  else:
    raise Exception("ERROR: Function not found")

def get_amount_of_inputs(fun):
  if fun == 'Booth':
    return 2
  elif fun == 'Matyas':
    return 2
  elif fun == 'Powersum':
    return 2 # Works with 1-4
  elif fun == 'McCormick':
    return 2
  elif fun == 'ThreeHumpCamel':
    return 2
  elif fun == 'SixHumpCamel':
    return 2
  elif fun == 'Sphere':
    return 2 # Note: variable input dimension
  elif fun == 'SumOfDifferentPowers':
    return 2 # Note: variable input dimension
  elif fun == 'SumSquares':
    return 2 # Note: variable input dimension
  elif fun == 'Trid':
    return 2 # Note: variable input dimension
  elif fun == 'Rosenbrock':
    return 2 # Note: variable input dimension (not implemented yet)
  elif fun == 'DixonPrice':
    return 2 # Note: variable input dimension
  elif fun == 'StyblinskiTang':
    return 2    # Note: variable input dimension
  elif fun == 'Beale':
    return 2
  elif fun == 'Branin':
    return 2
  elif fun == 'GoldsteinPrice':
    return 2
  elif fun == 'Easom':
    return 2
  elif fun == 'Michalewicz':
    return 2 # Note: variable input dimension
  else:
    raise Exception("ERROR: Function not found")

def get_sample_space(fun):
  if fun == 'Booth':
    return [[-10,10],[-10,10]]
  elif fun == 'Matyas':
    return [[-10,10],[-10,10]]
  elif fun == 'Powersum':
    return [[0,get_amount_of_inputs(fun)] for i in range(get_amount_of_inputs(fun))]
  elif fun == 'McCormick':
    return [[-1.5,4],[-3,4]]
  elif fun == 'ThreeHumpCamel':
    return [[-5,5],[-5,5]]
  elif fun == 'SixHumpCamel':
    return [[-3,3],[-2,2]]
  elif fun == 'Sphere':
    return [[-5.12,5.12] for i in range(get_amount_of_inputs(fun))]
  elif fun == 'SumOfDifferentPowers':
    return [[-1,1] for i in range(get_amount_of_inputs(fun))]
  elif fun == 'SumSquares':
    return [[-10,10] for i in range(get_amount_of_inputs(fun))]
  elif fun == 'Trid':
    n = get_amount_of_inputs(fun)
    return [[-np.power(n,2),np.power(n,2)] for i in range(n)]
  elif fun == 'Rosenbrock':
    return [[-5,10] for i in range(get_amount_of_inputs(fun))]
  elif fun == 'DixonPrice':
    return [[-10,10] for i in range(get_amount_of_inputs(fun))]
  elif fun == 'StyblinskiTang':
    return [[-5,5] for i in range(get_amount_of_inputs(fun))]
  elif fun == 'Beale':
    return [[-4.5,4.5],[-4.5,4.5]]
  elif fun == 'Branin':
    return [[-5,10],[0,15]]
  elif fun == 'GoldsteinPrice':
    return [[-2,2],[-2,2]]
  elif fun == 'Easom':
    return [[-5,5] for i in range(get_amount_of_inputs(fun))]
  elif fun == 'Michalewicz':
    return [[0,np.pi] for i in range(get_amount_of_inputs(fun))]
  else:
    raise Exception("ERROR: Function not found")