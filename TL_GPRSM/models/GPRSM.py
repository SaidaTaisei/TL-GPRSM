import GPy
import numpy as np
import os
import sys
import typing
import h5py

class GPRSM:
    def __init__(self, train_x:np.array, train_y:np.array, kernel_name:str="Matern52", is_ard:bool=True, normalizer="standardization") -> None:
        """initialize GPR surrogate model

        Args:
            train_x (numpy.ndarray): training data x (number of samples x number of features)
            train_y (numpy.ndarray): training data x (number of samples x 1)
            kernel_name (str): kernel name (Matern52, Matern32, RBF, etc.)
            is_ard (bool): whether ARD is used or not
            normalizer (str): normalizer (standardization, minmax, none)
        """
        if train_x.shape[0] != train_y.shape[0]:
            print("Error: train_x and train_y have different number of samples", file=sys.stderr)
            return
        self.train_x = train_x.copy()
        self.train_y = train_y.copy()
        if normalizer is not None:
            inp_x, inp_y = self.normalize(normalizer)
        else:
            inp_x, inp_y = train_x, train_y
        self.is_ard = is_ard
        self.kernel_name = kernel_name
        num_dim = train_x.shape[1]
        kernel_func = self.select_kernel_func(kernel_name)
        self.kernel = kernel_func(input_dim=num_dim, ARD=is_ard)
        self.model = GPy.models.GPRegression(X=inp_x, Y=inp_y, kernel=self.kernel, normalizer=False)
        self.is_transfer = False
        self.is_optimized = False
    
    def normalize(self, normalizer):
        if normalizer=="standardization":
            self.mean_x = np.mean(self.train_x, axis=0)
            self.std_x = np.std(self.train_x, axis=0)
            self.mean_y = np.mean(self.train_y, axis=0)
            self.std_y = np.std(self.train_y, axis=0)
            train_x = (self.train_x - self.mean_x) / self.std_x
            train_y = (self.train_y - self.mean_y) / self.std_y
        elif normalizer=="minmax":
            self.min_x = np.min(self.train_x, axis=0)
            self.max_x = np.max(self.train_x, axis=0)
            self.min_y = np.min(self.train_y, axis=0)
            self.max_y = np.max(self.train_y, axis=0)
            train_x = (train_x - self.min_x) / (self.max_x - self.min_x)
            train_y = (train_y - self.min_y) / (self.max_y - self.min_y)
        else:
            print("Error: normalizer is not defined", file=sys.stderr)
            return
        self.normalizer = normalizer
        return train_x, train_y
    
    def normalize_x(self, inp_x):
        if self.normalizer=="standardization":
            inp_x = (inp_x - self.mean_x) / self.std_x
        elif self.normalizer=="minmax":
            inp_x = (inp_x - self.min_x) / (self.max_x - self.min_x)
        else:
            print("Error: normalizer is not defined", file=sys.stderr)
            return
        return inp_x
    
    def inverse_normalize(self, pred_y):
        if self.normalizer=="standardization":
            pred_y = pred_y * self.std_y + self.mean_y
        elif self.normalizer=="minmax":
            pred_y = pred_y * (self.max_y - self.min_y) + self.min_y
        else:
            print("Error: normalizer is not defined", file=sys.stderr)
            return
        return pred_y
        
    def select_kernel_func(self, kernel_name):
        """select kernel function

        Args:
            kernel_name (str): kernel name (Matern52, Matern32, RBF, etc.)

        Returns:
            GPy.kern: kernel function
        """
        if kernel_name=="Matern52":
            return GPy.kern.Matern52
        elif kernel_name=="Matern32":
            return GPy.kern.Matern32
        elif kernel_name=="RBF":
            return GPy.kern.RBF
        elif kernel_name=="Exp" or kernel_name=="Exponential":
            return GPy.kern.Exponential
        elif kernel_name=="RQ" or kernel_name=="RatQuad":
            return GPy.kern.RatQuad
        else:
            print("Error: kernel name is not defined", file=sys.stderr)
            return None
    
    def set_transfer_learning(self, source_x, source_y):
        """set transfer learning

        Args:
            source_x (numpy.ndarray): source data x of transfer learning (number of samples x number of features)
            source_y (numpy.ndarray): source data y of transfer learning (number of samples x 1)
        """        
        if self.is_transfer:
            print("Reset transfer learning")
        self.source_x = source_x.copy()
        self.source_y = source_y.copy()
        min_dim = min(self.train_x.shape[1], self.source_x.shape[1])
        num_dim = self.train_x.shape[1] + self.source_x.shape[1] + min_dim
        self.kernel = GPy.kern.Matern52(input_dim=num_dim, ARD=self.kernel.ARD)
        if self.train_x.shape[1] <= self.source_x.shape[1]:
            train_x1 = np.hstack((self.source_x[:,:self.train_x.shape[1]], self.source_x, np.zeros(shape=(self.source_x.shape[0], self.train_x.shape[1]))))
            train_x2 = np.hstack((self.train_x, np.zeros(shape=(self.train_x.shape[0], self.source_x.shape[1])), self.train_x))
        else:
            train_x1 = np.hstack((self.source_x, self.source_x, np.zeros(shape=(self.source_x.shape[0], self.train_x.shape[1]))))
            train_x2 = np.hstack((self.train_x[:,:self.source_x.shape[1]], np.zeros(shape=(self.train_x.shape[0], self.source_x.shape[1])), self.train_x))
        self.target_x = self.train_x.copy()
        self.target_y = self.train_y.copy()
        self.train_x = np.vstack((train_x1, train_x2))
        self.train_y = np.vstack((self.source_y, self.train_y))
        if self.normalizer is not None:
            inp_x, inp_y = self.normalize(self.normalizer)
        else:
            inp_x, inp_y = train_x, train_y
        self.model = GPy.models.GPRegression(X=inp_x, Y=inp_y, kernel=self.kernel, normalizer=False)
        self.common_part_dim = min_dim
        self.target_part_dim = self.target_x.shape[1]
        self.source_part_dim = self.source_x.shape[1]
        self.is_transfer = True
    
    def optimize(self, max_iter:int, messages:bool=False, num_restarts:int=10, parallel:bool=True) -> None:
        """optimize GPR surrogate model

        Args:
            max_iter (int): max iteration number
            messages (bool, optional): print messages. Defaults to False.
            num_restarts (int, optional): number of restarts. Defaults to 10.
            parallel (bool, optional): is parallel. Defaults to True.
        """
        self.model.optimize(max_iters=max_iter, messages=messages)
        if num_restarts > 0:
            num_cpu = os.cpu_count()
            self.model.optimize_restarts(num_restarts=num_restarts, parallel=parallel, num_processes=num_cpu, messages=messages)
        self.is_optimized = True
    
    def predict(self, test_x:np.array) -> typing.Tuple[np.array, np.array]:
        """predict by GPR surrogate model

        Args:
            test_x (np.array): test data x (number of samples x number of features)

        Returns:
            typing.Tuple[np.array, np.array]: predicted mean and variance
        """
        if self.is_transfer:
            if self.target_x.shape[1] != test_x.shape[1]:
                print("Error: dimension of test data is different from dimension of training data", file=sys.stderr)
                return None, None
        else:
            if self.train_x.shape[1] != test_x.shape[1]:
                print("Error: dimension of test data is different from dimension of training data", file=sys.stderr)
                return None, None
        
        if not self.is_optimized:
            print("Warning: GPR surrogate model is not optimized", file=sys.stderr)
            return None, None
        self.test_x = test_x.copy()
        if self.is_transfer:
            if self.target_part_dim <= self.source_part_dim:
                self.test_x = np.hstack((test_x, np.zeros(shape=(self.test_x.shape[0], self.source_part_dim)), test_x))
            else:
                self.test_x = np.hstack((test_x[:, self.source_part_dim], np.zeros(shape=(self.test_x.shape[0], self.source_part_dim)), test_x))
        if self.normalizer is not None:
            inp_x = self.normalize_x(self.test_x)
        else:
            inp_x = self.test_x
        pred_mean, pred_var = self.model.predict(inp_x)
        if self.normalizer is not None:
            _tmp = self.inverse_normalize(pred_mean+pred_var)
            pred_mean = self.inverse_normalize(pred_mean)
            pred_var = _tmp - pred_mean
        return pred_mean, pred_var
    
    def get_ard_contribution(self, normalized=True) -> np.array:
        """get ARD contribution

        Args:
            normalized (bool, optional): whether normalized or not. Defaults to True.

        Returns:
            np.array: ARD contributions
        """
        if not self.is_optimized:
            print("Error: model is not optimized", file=sys.stderr)
            return None
        if not self.is_ard:
            print("Error: kernel is not ARD", file=sys.stderr)
            return None
        length_scales = self.model.kern.lengthscale
        contribution = [1.0/l for l in length_scales]
        if normalized:
            contribution = [100.0*c/sum(contribution) for c in contribution]
        return np.array(contribution)
    
    def get_transfer_learning_effect(self) -> float:
        """get transfer learning effect

        Returns:
            float: effect of transfer learning (%)
        """
        if not self.is_optimized:
            print("Error: model is not optimized", file=sys.stderr)
            return None
        if not self.is_ard:
            print("Error: kernel is not ARD", file=sys.stderr)
            return None
        if not self.is_transfer:
            print("Error: model is not transfer lerning", file=sys.stderr)
            return None
        contribution = self.get_ard_contribution(normalized=True)
        contribution_common = np.sum(contribution[:self.common_part_dim])
        contribution_target = np.sum(contribution[self.common_part_dim+self.source_part_dim:])
        effect = contribution_common / contribution_target
        return effect
    
    def get_part_contribution(self) -> typing.Tuple[float, float, float]:
        """get contribution of each part

        Returns:
            typing.Tuple[float, float, float]: contribution of common part, source part, target part (%)
        """
        if not self.is_optimized:
            print("Error: model is not optimized", file=sys.stderr)
            return None
        if not self.is_ard:
            print("Error: kernel is not ARD", file=sys.stderr)
            return None
        if not self.is_transfer:
            print("Error: model is not transfer lerning", file=sys.stderr)
            return None
        contribution = self.get_ard_contribution(normalized=True)
        contribution_common = np.sum(contribution[:self.common_part_dim])
        contribution_source = np.sum(contribution[self.common_part_dim:self.common_part_dim+self.source_part_dim])
        contribution_target = np.sum(contribution[self.common_part_dim+self.source_part_dim:])
        return contribution_common, contribution_source, contribution_target
    
    def save_model(self, path:str) -> None:
        """save model

        Args:
            path (str): path to save model
        """
        if not self.is_optimized:
            print("Error: model is not optimized", file=sys.stderr)
            return None
        if not ".h5" in path:
            print("Error: path must be .h5 file", file=sys.stderr)
            return None
        with h5py.File(path, "w") as h5:
            h5.create_dataset("param_array", data=self.model.param_array)
            h5.create_dataset("is_ard", data=np.array([self.is_ard]), dtype=np.bool)
            h5.create_dataset("is_transfer", data=np.array([self.is_transfer]), dtype=np.bool)
            h5.create_dataset("is_optimized", data=np.array([self.is_optimized]), dtype=np.bool)
            kernel_name = h5.create_dataset("kernel_name", shape=(1,), dtype=h5py.special_dtype(vlen=str))
            kernel_name[0] = self.kernel_name
            h5.create_dataset("train_x", data=self.train_x)
            h5.create_dataset("train_y", data=self.train_y)
            if self.is_transfer:
                h5.create_dataset("common_part_dim", data=np.array([self.common_part_dim]), dtype=np.int32)
                h5.create_dataset("source_part_dim", data=np.array([self.source_part_dim]), dtype=np.int32)
                h5.create_dataset("target_part_dim", data=np.array([self.target_part_dim]), dtype=np.int32)
                h5.create_dataset("target_x", data=self.target_x)
                h5.create_dataset("target_y", data=self.target_y)
                h5.create_dataset("source_x", data=self.source_x)
                h5.create_dataset("source_y", data=self.source_y)
    
    @staticmethod
    def load_model(path:str):
        """load model

        Args:
            path (str): path to load model
        Returns:
            GPRSM: GPRSM model
        """
        if not ".h5" in path:
            print("Error: path must be .h5 file", file=sys.stderr)
            return None
        with h5py.File(path, "r") as h5:
            param_array = h5["param_array"][:]
            train_x = h5["train_x"][:]
            train_y = h5["train_y"][:]
            kernel_name = h5["kernel_name"][0].decode()
            is_ard = h5["is_ard"][0]
        gprsm = GPRSM(train_x, train_y, kernel_name=kernel_name, is_ard=is_ard)
        with h5py.File(path, "r") as h5:
            gprsm.is_transfer = h5["is_transfer"][0]
            gprsm.is_optimized = h5["is_optimized"][0]
            if gprsm.is_transfer:
                gprsm.common_part_dim = h5["common_part_dim"][0]
                gprsm.source_part_dim = h5["source_part_dim"][0]
                gprsm.target_part_dim = h5["target_part_dim"][0]
                gprsm.target_x = h5["target_x"][:]
                gprsm.target_y = h5["target_y"][:]
                gprsm.source_x = h5["source_x"][:]
                gprsm.source_y = h5["source_y"][:]
        gprsm.model = GPy.models.GPRegression(gprsm.train_x, gprsm.train_y, gprsm.kernel, initialize=False)
        gprsm.model.update_model(False)
        gprsm.model.initialize_parameter()
        gprsm.model[:] = param_array
        gprsm.model.update_model(True)
        return gprsm
