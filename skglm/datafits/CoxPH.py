import numpy as np
from numba import float64


class Cox_PH():
    """Base class for datafits."""
    def __init__(self):
        pass
    
    def get_spec(self):
        """Specify the numba types of the class attributes.

        Returns
        -------
        spec: Tuple of (attribute_name, dtype)
            spec to be passed to Numba jitc
            lass to compile the class.
        """
        spec = (
            ('Xty', float64[:]),
            ('lipschitz', float64[:]),
            ('global_lipschitz', float64),
        )
        return spec
    
    def params_to_dict(self):
        """Get the parameters to initialize an instance of the class.

        Returns
        -------
        dict_of_params : dict
            The parameters to instantiate an object of the class.
        """
        return dict()
    
    def get_risk_times(self, y):
    
        Times = []
        Risk = []
        E = []
        
        for label in y:
            time = label[0]
            
            if time not in Times:
                Times.append(time)
        
        print(" :: Times :: ", Times)
        sorted_Times = np.sort(Times)
        
        for time in sorted_Times:
            risk_time = []
            Et = []
            for i in range(len(y)):
                
                if y[i][0] >= time:
                    risk_time.append(i)
                if y[i][0] == time:
                    Et.append(i)
                    
            E.append(Et)
                    
            Risk.append(risk_time)
            
        return Risk, sorted_Times, E
    
    def initialize(self, X, y):
        """Pre-computations before fitting on X and y.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Design matrix.

        y : array, shape (n_samples,)
            Target vector.
        """
        
        n_features = len(X[0])
        self.global_lipschitz = 1
        self.lipschitz = np.ones(n_features, dtype=X.dtype) 
        
    def initialize_sparse(
            self, X_data, X_indptr, X_indices, y):
        """Pre-computations before fitting on X and y when X is a sparse matrix.

        Parameters
        ----------
        X_data : array, shape (n_elements,)
            `data` attribute of the sparse CSC matrix X.

        X_indptr : array, shape (n_features + 1,)
            `indptr` attribute of the sparse CSC matrix X.

        X_indices : array, shape (n_elements,)
            `indices` attribute of the sparse CSC matrix X.

        y : array, shape (n_samples,)
            Target vector.
        """
        n_features = len(X_indptr) - 1
        self.lipschitz = np.ones(n_features, dtype=X_data.dtype)
    
        
    def value(self, y, w, Xw):
        """Value of datafit at vector w.

        Parameters
        ----------
        y : array_like, shape (n_samples,)
            Target vector.

        w : array_like, shape (n_features,)
            Coefficient vector.

        Xw: array_like, shape (n_samples,)
            Model fit.

        Returns
        -------
        value : float
            The datafit value at vector w.
        """
        Risk, Times, E = self.get_risk_times(y)
        
        n_times = len(Times)
        
        res = 0
        
        for j in range(n_times):
            Ej = E[j]
            sum_events = sum([Xw[k] for k in Ej])
            sum_risk = sum([np.exp(Xw[k]) for k in Risk[j]])
            res += sum_events - len(Ej) * np.log(sum_risk)
            
        return -1 * res
    
    def gradient_scalar(self, X, y, w, Xw, j):
        """Gradient with respect to j-th coordinate of w.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Design matrix.

        y : array, shape (n_samples,)
            Target vector.

        w : array, shape (n_features,)
            Coefficient vector.

        Xw : array, shape (n_samples,)
            Model fit.

        j : int
            The coordinate at which the gradient is evaluated.

        Returns
        -------
        gradient : float
            The gradient of the datafit with respect to the j-th coordinate of w.
        """
        Risk, Times, E = self.get_risk_times(y)
        
        n_times = len(Times)
        
        grad_j = 0
        
        for i in range(n_times):
            Ei = E[i]
            sum_events = sum([X[k][j] for k in Ei])
            sum_risk = (sum([np.exp(Xw[a]) * X[a][j] for a in Risk[i]])) / (sum([np.exp(Xw[a]) for a in Risk[i]]))
            grad_j += sum_events - len(Ei) * sum_risk
            
        return -1 * grad_j
    
    def gradient_scalar_sparse(self, X_data, X_indptr, X_indices, y, Xw, j):
        """Gradient with respect to j-th coordinate of w when X is sparse.

        Parameters
        ----------
        X_data : array, shape (n_elements,)
            `data` attribute of the sparse CSC matrix X.

        X_indptr : array, shape (n_features + 1,)
            `indptr` attribute of the sparse CSC matrix X.

        X_indices : array, shape (n_elements,)
            `indices` attribute of the sparse CSC matrix X.

        y : array, shape (n_samples,)
            Target vector.

        Xw: array, shape (n_samples,)
            Model fit.

        j : int
            The dimension along which the gradient is evaluated.

        Returns
        -------
        gradient : float
            The gradient of the datafit with respect to the j-th coordinate of w.
        """
        Risk, Times, E = self.get_risk_times(y)
        
        n_times = len(Times)
        
        grad_j = 0
        
        for i in range(n_times):
            Ei = E[i]
            sum_events = 0
            num_risk = 0
            for k in range(X_indptr[j], X_indptr[j + 1]):
                if k in Ei:
                    sum_events += X_data[k]
                if k in Risk[i]:
                    idx_k = X_indices[k]
                    num_risk += X_data[k] * np.exp(Xw[idx_k])
            
            sum_risk = num_risk / (sum([np.exp(Xw[a]) for a in Risk[i]]))
            grad_j += sum_events - len(Ei) * sum_risk
            
        return -1 * grad_j
