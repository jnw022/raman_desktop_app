import numpy as np
import pandas as pd
from pyspectra.transformers.spectral_correction import msc
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression


class MSC():
    """Multiplicative Scatter Correction"""
    
    def __init__(self):
        self.mean_spectrum = None  # reference spectrum--i.e. the mean of all calibration spectra.
        
    def fit(self, spc):
        self.mean_spectrum = np.array(spc.mean(axis=0))
        
    def transform(self, spc):
        """Applies the correction to each spectrum provided"""
        
        def transform_per_spc(spc, mean_spec):
            """Fits a linear regression between the reference and sample spectra to identify additive
               and multiplicative scattering effects.
               Then, it backs the regression out of the sample spectrum to correct for those effects."""
            
            m, b = np.polyfit(mean_spec, spc, 1)
            
            # y=mx+b --> (y-b)/m = x
            return (spc - b) / m  
        
        # apply the transformation to each spectrum
        try:
            # spc is a Dataframe
            output = spc.apply(transform_per_spc, args=(self.mean_spectrum,), axis=1)
        except TypeError:
            # spc is a Series
            output = transform_per_spc(spc, self.mean_spectrum)
        
        return output
    
    def fit_transform(self, spc):
        # fit the spectra
        self.fit(spc)
        # make the transformation
        return self.transform(spc)


class Product_Classifier():
    """This class encapsulates all models and transformers associated with the product classifier"""
    
    def __init__(self, svm, scaler, pca, msc):
        self.svm = svm
        self.scaler = scaler
        self.pca = pca
        self.msc = msc
    
    def identify_product(self, spec):
        """Identify which product a spectrum represents"""
        
        # MSC correction
        spec_pp = self.msc.transform(spec)
        # reshape spectrum to 2d array
        spec_rs = spec_pp.values.reshape(1, -1)
        # center/scale the spectrum
        spec_scaled = self.scaler.transform(spec_rs)
        # run PCA on scaled spectrum
        spec_pca = self.pca.transform(spec_scaled)
        # run SVM on PCA outputs (returns a 1D vector with a single element)
        output = self.svm.predict(spec_pca)[0]
        
        return output 


class PLS_Model():
    """This class wraps PLS model objects, plus all other data and behaviors necessary to make them work."""
    
    def __init__(self, pls, scaler, msc, roi_limits, cal_scores, output_types):
        # the PLS model itself
        self.model = pls
        # pretrained scaler
        self.scaler = scaler
        # pretrained MSC object
        self.msc = msc
        # tuple of lower and upper bounds, respectively, for the spectral region of interest
        self.roi_limits = roi_limits
        # matrix of PLS scores for the calibration set this model was trained on.  
        # it is used in the calculation of M-distance.
        self.cal_scores = cal_scores
        # a tuple of strings labeling the type of properties the model is predicting (i.e. (lights, heavies), etc)
        self.output_types = output_types


    def _set_roi(self, spec):
        """Utility method to trim wavenumbers that aren't in the region of interest"""
    
        low, high = self.roi_limits
        
        # array of 1.0 wavenumber steps from start (400 wavenumbers) to lower bound of ROI
        roi_low = np.arange(400, low, 1.0)
        # higher to end (2000 wavenumbers)
        roi_high = np.arange(high+1, 2001, 1.0)

        spec = spec.drop(labels=roi_low)
        spec = spec.drop(labels=roi_high)
        
        return spec
    
    def _mahalanobis(self, x):
        """
        Calculate Mahalanobis distance between a PLS vector and a set of calibration points.
        Mahalanobis distance definition: 
    
        Definition:  D^2 = (x - m) dot C^-1 dot (x - m).T
    
        Where x is a vector (i.e. PLS scores for a spectrum),
        Y is the matrix of PLS scores for the calibration set.
        and C is the covariance matrix for calibration data.
        """

        delta_x_m = x - np.mean(self.cal_scores, axis=0)
        cov = np.cov(self.cal_scores.T)
        inv_cov = np.linalg.inv(cov)
        mahal_squared = np.dot(np.dot(delta_x_m, inv_cov), delta_x_m.T)
    
        return np.sqrt(mahal_squared)
    
    def _spectral_residual(self, original, pls_scores):
        """
        Returns the normalized spectral residual for a given spectrum.
        The MSC correction is expected to have already been applied.
        """
        
        # get loading matrix from PLS model
        loadings = self.model.x_loadings_
        # dot product between PLS scores for the spectrum and the model's loadings.
        # also unsqueeze the array to 2D for the scaler.
        scaled_reconstruction = np.dot(pls_scores, loadings.T).reshape(1, -1)
        # unscale the reconstruction and squeeze back to 1D
        reconstructed_spectrum = self.scaler.inverse_transform(scaled_reconstruction).reshape(-1)
        # delta between original spectrum and reconstructed spectrum
        residual = original - reconstructed_spectrum
        # sum of residual normalized by the original spectrum
        norm_residual = abs(residual).sum() / original.sum() 
        
        return norm_residual
    
    def predict(self, spec):
        """Apply a PLS model to a given spectrum"""
    
        # extract region of interest from the spectrum
        spec_roi = self._set_roi(spec)
        # perform multiplicative scattering correction on spectrum
        spec_pp = self.msc.transform(spec_roi)
        # sklearn expects 2d arrays as inputs, so the spectrum must first be reshaped
        spec_rs = spec_pp.values.reshape(1, -1)
        # center/scale the spectrum 
        spec_scaled = self.scaler.transform(spec_rs)
        # run PLS on scaled spectrum (returns a 2D vector)
        output = self.model.predict(spec_scaled)[0]
        
        # calculate M-distance
        pls_scores = self.model.transform(spec_scaled)
        m_dist = self._mahalanobis(pls_scores)[0][0]
        
        # calculate Q-residual
        residual = self._spectral_residual(spec_pp, pls_scores)
        
        return output, self.output_types, m_dist, residual


class Model_Container():
    """This class encapsulates all the objects and behaviors required for running the Rapid Product Release model."""
    
    def __init__(self, classifier, c6, c8, c10, configs, version):
        # product classifier object
        self.classifier = classifier
        
        # PLS models
        self.c6_model = c6
        self.c8_model = c8
        self.c10_model = c10
        
        # JSON object containing spec limits and discrimination criteria
        self.configs = configs
        
        self.version = version
        
        # mapping between product classifications and the models that correspond to them
        self.model_mapping = {'N6': self.c6_model,
                              'N8': self.c8_model,
                              'N10': self.c10_model}
        
    def quantify(self, spec, model_type):
        """Apply a PLS model to a given spectrum"""
        
        if model_type in self.model_mapping.keys():
            # pick and run the correct model
            model = self.model_mapping[model_type]
            output = model.predict(spec)
        else:
            # no PLS model has been built for this product
            output = "Model Not Trained"
        
        return output
    
    def identify_product(self, spec):
        """Identify which product a spectrum represents"""
        
        output = self.classifier.identify_product(spec)
        
        return output  