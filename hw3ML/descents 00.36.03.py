from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """
    A base class and templates for all functions
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        :param dimension: feature space dimension
        :param lambda_: learning rate parameter
        :param loss_function: optimized loss function
        """
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function
        self.delta: int = 1

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Template for update_weights function
        Update weights with respect to gradient
        :param gradient: gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Template for calc_gradient function
        Calculate gradient of loss function with respect to weights
        :param x: features array
        :param y: targets array
        :return: gradient: np.ndarray
        """
        
        if self.loss_function.name == 'MSE':
            return (2 * x.transpose() @ x @ self.w - 2 * x.transpose() @ y)/x.shape[0]

        elif self.loss_function.name == 'MAE':
            return (np.sign(x.transpose() @ x @ self.w - x.transpose() @ y))/x.shape[0]
        
        elif self.loss_function.name == 'Huber':
            diff = y - self.predict(x)
            mse = np.abs(diff)/x.shape[0] < self.delta
            
            return np.sum((~mse) * (0.5 * diff ** 2) - (mse) * self.delta * (0.5 * self.delta - diff))

        elif self.loss_function.name == 'LogCosh':
            return (np.tanh(x @ self.w - y).transpose() @ x) /x.shape[0]

        else:
            raise NameError('Unknown Loss')


    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate loss for x and y with our weights
        :param x: features array
        :param y: targets array
        :return: loss: float
        """

        if self.loss_function.name == 'MSE':
            return np.sum((y - self.predict(x))**2)/x.shape[0]

        elif self.loss_function.name == 'MAE':
            return np.sum(np.abs(y - self.predict(x)))/x.shape[0]
        
        elif self.loss_function.name == 'Huber':

            diff = y - self.predict(x)
            mse = np.abs(diff)/x.shape[0] < self.delta
            
            return np.sum((~mse) * self.delta * (np.abs(diff)-1/2*self.delta) - (mse) * 1/2*(diff**2))

        elif self.loss_function.name == 'LogCosh':
            return np.sum(np.log(np.cosh(y - self.predict(x))))/x.shape[0]

        else:
            raise NameError('Unknown Loss')

  


    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate predictions for x
        :param x: features array
        :return: prediction: np.ndarray
        """
        # # TODO: implement prediction function
        # raise NotImplementedError('BaseDescent predict function not implemented')

        return x @ self.w


class VanillaGradientDescent(BaseDescent):
    """
    Full gradient descent class
    """

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        step = -self.lr() * gradient
        self.w += step
        return step
        
    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # # TODO: implement calculating gradient function
        # raise NotImplementedError('VanillaGradientDescent calc_gradient function not implemented')
        return super().calc_gradient(x, y)


class StochasticDescent(VanillaGradientDescent):
    """
    Stochastic gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
        
        loss_function: LossFunction = LossFunction.MSE):
        """
        :param batch_size: batch size (int)
        """
        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # # TODO: implement calculating gradient function
        # raise NotImplementedError('StochasticDescent calc_gradient function not implemented')
        
        indx = np.random.randint(0, x.shape[0], self.batch_size)
        x_sample = x[indx]
        y_sample = y[indx]
        return super().calc_gradient(x_sample, y_sample)


class MomentumDescent(VanillaGradientDescent):
    """
    Momentum gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9

        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.h = self.alpha * self.h + self.lr() * gradient
        self.w -= self.h 

        return -self.h


class Adam(VanillaGradientDescent):
    """
    Adaptive Moment Estimation gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """

        self.iteration += 1
        self.m = self.beta_1 * self.m + (1-self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1-self.beta_2) * (gradient**2)
        m_hat = self.m/(1 - self.beta_1 ** self.iteration)
        v_hat = self.v/(1 - self.beta_2 ** self.iteration)
        step_adam = self.lr()/(np.sqrt(v_hat)+ self.eps) * m_hat
        self.w -= step_adam

        return -step_adam
        
class AdamMax(VanillaGradientDescent):
    """
    Another Adaptive Moment Estimation gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999
        self.alpha: float = 0.002

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """

        self.iteration += 1
        self.m = self.beta_1 * self.m + (1-self.beta_1) * gradient
        self.v = np.max((self.beta_2 * self.v, np.abs(gradient)), axis = 0)
        m_hat = self.m/(1 - self.beta_1 ** self.iteration)
        
        step_adam = self.lr()*self.alpha  * m_hat /(self.v+self.eps)
        self.w -= step_adam

        return -step_adam
        

class BaseDescentReg(BaseDescent):
    """
    A base class with regularization
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        :param mu: regularization coefficient (float)
        """
        super().__init__(*args, **kwargs)

        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of loss function and L2 regularization with respect to weights
        """
        l2_gradient: np.ndarray = self.w  # TODO: replace with L2 gradient calculation
        l2_gradient[-1] = 0

        return super().calc_gradient(x, y) + l2_gradient * self.mu

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of loss function and L2 regularization with respect to weights
        """

        return super().calc_loss(x, y) + np.sum(self.mu * (self.w)**2)/x.shape[0]


class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Full gradient descent with regularization class
    """


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Stochastic gradient descent with regularization class
    """


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Momentum gradient descent with regularization class
    """


class AdamReg(BaseDescentReg, Adam):
    """
    Adaptive gradient algorithm with regularization class
    """

class AdamMaxReg(BaseDescentReg, AdamMax):
    """
    Adaptive gradient algorithm with regularization class
    """


def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg,
        'adammax': AdamMax if not regularized else AdamMaxReg
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))
