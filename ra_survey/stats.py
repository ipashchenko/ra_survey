#!/usr/bin python
# -*- coding: utf-8 -*-

from __future__ import print_function
from utils import gauss_2d_anisotropic
import numpy as np
import warnings
import math
try:
    from scipy.stats import rice
    from scipy.special import erf, iv
    from scipy.optimize import leastsq
    is_scipy = True
except ImportError:
    warnings.warn('Install ``scipy`` python package to use least squares'
                  ' estimates.')
    is_scipy = False


# TODO: Implement additional arguments for ``model.__call__``. It will allow
# more flexibility (using functions not only classes for ``model``).
class LnPost(object):
    """
    Class that represents posterior probability density of parameters.

    :param x:
        Vector of explanatory variable.

    :param y:
        Vector of response variable.

    :param model:
        Model class. Class with ``__call__`` method that accepts
        vector of parameters and returns vector of model values
        for given explanatory variables (defined e.g. in it's
        constructor).

    :param sy: (optional)
        Std of estimates of response variable.

    :param x_limits: (optional)
        Vector of explanatory variable for limits.

    :param y_limits: (optional)
        Vector of the response variable for limits.

    :param sy_limits: (optional)
        Std of estimates of limits for response variable.

    :param lnpr:
        Instance of ``LnPrior`` class. Actually any callable that returns ln of
        the prior pdf for parameter vector.

    :param args: (optional)
        Positional arguments for prior density callable.

    :param kwargs: (optional)
        Keyword arguments for prior density callable.

    :param jitter: (optional)
        use jitter if data allows it. (default: ``False``)

    :param outliers: (optional)
        model outliers. (default: ``False``)

    """
    def __init__(self, x, y, model, sy=None, x_limits=None, y_limits=None,
                 sy_limits=None, lnpr=None, args=None, kwargs=None,
                 jitter=False, outliers=False, model_args=None,
                 model_kwargs=None, model_call_args=None,
                 model_call_kwargs=None):

        if lnpr is None:
            raise Exception('No prior information in ``lnpr`` argument!')

        self._lnpr = lnpr
        if args is None:
            args = list()
        if kwargs is None:
            kwargs = dict()
        self.args = args
        self.kwargs = kwargs
        self._lnlike = LnLike(x, y, model, sy=sy, x_limits=x_limits,
                              y_limits=y_limits, sy_limits=sy_limits,
                              jitter=jitter, outliers=outliers,
                              model_args=model_args, model_kwargs=model_kwargs,
                              model_call_args=model_call_args,
                              model_call_kwargs=model_call_kwargs)
        self.model_args = model_args
        self.model_kwargs = model_kwargs
        self.model_call_args = model_call_args
        self.model_call_kwargs = model_call_kwargs

    def lnpr(self, p):
        return self._lnpr(p, *self.args, **self.kwargs)

    def lnlike(self, p):
        return self._lnlike.__call__(p)

    def __call__(self, p):
        lnpr = self.lnpr(p)
        if np.isinf(lnpr):
            return lnpr
        return self.lnlike(p) + lnpr


# TODO: Add t-distribution noise model
# TODO: Account for outliers in data
class LnLike(object):
    """
    Class that represents Likelihood function.

    :param x:
        uv-radius (for 1D) or coordinates in uv-plane (for 2D models).

    :param y:
        Vector of response variable.

    :param model:
        Model class. Class with ``__call__`` method that accepts
        vector of parameters and returns vector of model values
        for given explanatory variables (defined e.g. in it's
        constructor).

    :param sy: (optional)
        Std of estimates of response variable.

    :param x_limits: (optional)
        Vector of explanatory variable for limits.

    :param y_limits: (optional)
        Vector of the response variable for limits.

    :param sy_limits: (optional)
        Std of estimates of limits for response variable.

    :param jitter: (optional)
        use jitter if data allows it. (default: ``False``)

    :param outliers: (optional)
        model outliers. (default: ``False``)

    """
    def __init__(self, x, y, model, sy=None, x_limits=None,
                 y_limits=None, sy_limits=None, jitter=False,
                 outliers=False, model_args=None, model_kwargs=None,
                 model_call_args=None, model_call_kwargs=None):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self._model = model
        if sy is not None:
            self.sy = np.asarray(sy)
        else:
            self.sy = None
        if x_limits is not None:
            self.x_limits = np.asarray(x_limits)
            self.y_limits = np.asarray(y_limits)
        else:
            self.x_limits = None
            self.y_limits = None
        if sy_limits is not None:
            self.sy_limits = np.asarray(sy_limits)
        else:
            self.sy_limits = None
        self.jitter = jitter
        self.outliers = outliers

        # TODO: Put in subclass
        # Various assertions on data consistency
        assert(len(self.x) == len(self.y))
        if self.sy is not None:
            assert(len(self.y) == len(self.sy))

        self._lnprob = list()
        self._lnprob.append(LnProbDetections(x, y, self._model, sy=sy,
                                             jitter=jitter, outliers=outliers,
                                             model_args=model_args,
                                             model_kwargs=model_kwargs,
                                             model_call_args=model_call_args,
                                             model_call_kwargs=model_call_kwargs))

        # If we have any data on upper limits
        if (self.x_limits is not None) or (self.y_limits is not None) or\
                (self.sy_limits is not None):
            # Then we must have at least x & y for upper limits
            assert(self.x_limits is not None and self.y_limits is not None)
            # Assert equal dimensionality for x of detections and limits
            assert(self.x.ndim == self.x_limits.ndim)

            self._lnprob.append(LnProbUlimits(x_limits, y_limits, self._model,
                                              sy=sy_limits, jitter=jitter,
                                              outliers=outliers,
                                              model_args=model_args,
                                              model_kwargs=model_kwargs,
                                              model_call_args=model_call_args,
                                              model_call_kwargs=model_call_kwargs))

# TODO: Should i try to use try/except?
    def __call__(self, p):
        return sum([lnprob(p) for lnprob in self._lnprob])


# TODO: ``x``` in constructor to use the same subclass of model for
# different xs.
class LnProb(object):
    """
    Basic class that calculates the probability of parameters given
    data (i.e. likelihood).

    :param x:
        Vector of explanatory variable.

    :param y:
        Vector of response variable.

    :param model:
        1) Model class. Class with ``__call__`` method that accepts
        vector of parameters and returns vector of model values
        for given explanatory variables (defined e.g. in it's
        constructor).
        2) Model function. ``model_args`` & ``model_kwargs`` are used when
        calling function.That is if ``model = foo`` then actually ``foo`` is
        called ``foo(p, *model_args, **model_kwargs)``.

    :param sy (optional):
        Std of estimates of responce variable.

    :param jitter (optional):
        Use jitter if data allows it.

    :param outliers (optional):
        Model outliers.
    """
    def __init__(self, x, y, model, model_args=None, model_kwargs=None, sy=None,
                 jitter=False, outliers=False, model_call_args=None,
                 model_call_kwargs=None):
        self.x = x
        self.y = y
        self.sy = sy
        if model_args is None:
            model_args = []
        if model_kwargs is None:
            model_kwargs = {}
        if model_call_args is None:
            model_call_args = []
        if model_call_kwargs is None:
            model_call_kwargs = {}
        self.model_args = model_args
        self.model_kwargs = model_kwargs
        self.model_call_args = model_call_args
        self.model_call_kwargs = model_call_kwargs
        # Initialize model class instance
        if isinstance(model, type):
            self.model = model(x, *self.model_args, **self.model_kwargs)
        else:
            self.model = model
        self.jitter = jitter
        self.outliers = outliers

        # TODO: Put logic to method?
        if self.sy is None:
            if not jitter and not outliers:
                self.lnprob = self._lnprob3
            elif not jitter and outliers:
                self.lnprob = self._lnprob5
            else:
                raise Exception("Your data doesn't support this setup.")
        else:
            if not jitter and not outliers:
                self.lnprob = self._lnprob1
            elif not jitter and outliers:
                self.lnprob = self._lnprob6
            elif jitter and not outliers:
                self.lnprob = self._lnprob2
            elif jitter and outliers:
                self.lnprob = self._lnprob4

    def __call__(self, p):
        return self.lnprob(p)

    def _lnprob1(self, p):
        """
        With estimated uncertainties.
        """
        raise NotImplementedError()

    def _lnprob2(self, p):
        """
        With estimated uncertainties plus jitter.
        """
        raise NotImplementedError()

    def _lnprob3(self, p):
        """
        Without estimated uncertainties.
        """
        raise NotImplementedError()

    def _lnprob4(self, p):
        """
        Without estimated uncertainties plus outliers.
        """
        raise NotImplementedError()

    def _lnprob5(self, p):
        """
        With estimated uncertainties plus outliers.
        """
        raise NotImplementedError()

    def _lnprob6(self, p):
        """
        With estimated uncertainties plus jitter plus outliers.
        """
        raise NotImplementedError()


class LnProbDetections(LnProb):
    """
    Likelihood calculation for detections.

    """
    def __init__(self, x, y, model, model_args=None, model_kwargs=None, sy=None,
                 noise=None, jitter=False, outliers=False, model_call_args=None,
                 model_call_kwargs=None):
        super(LnProbDetections, self).__init__(x, y, model,
                                               model_args=model_args,
                                               model_kwargs=model_kwargs, sy=sy,
                                               jitter=jitter, outliers=outliers,
                                               model_call_args=model_call_args,
                                               model_call_kwargs=model_call_kwargs)
        # TODO: Subclass detections with noise types
        self.noise = noise

    def _lnprob1(self, p):
        """
        With estimated uncertainties.

        :param p:
            Array-like of the parameters.

        """
        # Gaussian noise
        #return (-0.5 * np.log(2. * math.pi * self.sy ** 2) -
        #       (self.y - self.model(p)) ** 2. / (2. * self.sy ** 2)).sum()

        model = self.model(p, *self.model_call_args, **self.model_call_kwargs)
        # Rice distributed noise
        #return (np.log(self.y) - 2. * np.log(self.sy) -
        #        (self.y ** 2 + model) / (2. * self.sy ** 2.) +
        #        np.log(iv(0, self.y * model / self.sy ** 2.))).sum()
        # Using scipy.stats.rice
        return rice.logpdf(self.y, model, scale=self.sy).sum()

    def _lnprob2(self, p):
        """
        With estimated uncertainties plus jitter.

        :param p:
            Array-like of the parameters, where:
            p[-1] - variance of jitter.

        """
        # Gaussian noise
        #return (-0.5 * np.log(2. * math.pi * (p[-1] + self.sy ** 2)) -
        #        (self.y - self.model(p)) ** 2. / (2. * (p[-1] + self.sy **
        #                                                2))).sum()

        model = self.model(p, *self.model_call_args, **self.model_call_kwargs)
        # Rice distributed noise
        return (np.log(self.y) + np.log(self.sy ** 2. + p[-1]) -
                (self.y ** 2 + model) / (2. * (self.sy ** 2. + p[-1])) +
                np.log(iv(0, self.y * model / (self.sy ** 2. + p[-1])))).sum()
        # Using scipy.stats.rice
        return rice.logpdf(self.y, model,
                           scale=np.sqrt(self.sy ** 2. + p[-1])).sum()

    def _lnprob3(self, p):
        """
        Without estimated uncertainties.

        :param p:
            Array-like of the parameters, where:
            p[-1] - variance of data.
        """
        return (-0.5 * math.log(2. * math.pi * p[-1] ** 2) -
               (self.y - self.model(p)) ** 2. / (2. * p[-1])).sum()

    def _lnprob4(self, p):
        """
        Without estimated uncertainties plus outliers.

        :param p:
            Array-like of the parameters, where:
            p[-4] - variance of data,
            p[-3] - probability that data point comes from outlier distribution,
            p[-2], p[-1] - mean and variance of outliers distribution.
        """
        return (np.logaddexp(-(self.y - self.model(p)) ** 2 /
                             (2. * p[-4]) +
                             math.log((1. - p[-3]) /
                                      np.sqrt(2. * math.pi * p[-4])),
                             -(self.y - p[-2]) ** 2 / (2. * (p[-4] + p[-1])) +
                             math.log(p[-3] /
                                      math.sqrt(2. * math.pi *
                                                (p[-4] + p[-1]))))).sum(axis=0)

    def _lnprob5(self, p):
        """
        With estimated uncertainties plus outliers.

        :param p:
            Array-like of the parameters, where:
            p[-3] - probability that data point comes from outlier distribution,
            p[-2], p[-1] - mean and variance of outliers distribution.
        """
        return (np.logaddexp(-(self.y - self.model(p)) ** 2 /
                             (2. * self.sy ** 2.) +
                             np.log((1. - p[-3]) /
                                    np.sqrt(2. * math.pi * self.sy ** 2.)),
                             -(self.y - p[-2]) ** 2 /
                             (2. * (self.sy ** 2. + p[-1])) +
                             np.log(p[-3] / np.sqrt(2. * math.pi *
                                                    (self.sy ** 2. +
                                                     p[-1]))))).sum(axis=0)

    def _lnprob6(self, p):
        """
        With estimated uncertainties plus jitter plus outliers.

        :param p:
            Array-like of the parameters, where:
            p[-4] - variance of jitter,
            p[-3] - probability that data point comes from outlier distribution,
            p[-2], p[-1] - mean and variance of outliers distribution.
        """
        return (np.logaddexp(-(self.y - self.model(p)) ** 2 /
                             (2. * (p[-4] + self.sy ** 2.)) +
                             np.log((1. - p[-3]) /
                                    np.sqrt(2. * math.pi * (p[-4] +
                                                            self.sy ** 2.))),
                             -(self.y - p[-2]) ** 2 /
                             (2. * (self.sy ** 2. + p[-1] + p[-4])) +
                             np.log(p[-3] /
                                    np.sqrt(2. * math.pi *
                                            (self.sy ** 2. +
                                             p[-1] + p[-4]))))).sum(axis=0)


# TODO: Can i calculate upper lim. probability for t-distribution?
class LnProbUlimits(LnProb):
    """
    Lnlikelihood calculation for upper limits.

    """
    def __init__(self, x, y, model, model_args=None, model_kwargs=None, sy=None,
                 jitter=None, outliers=None, model_call_args=None,
                 model_call_kwargs=None):
        super(LnProbUlimits, self).__init__(x, y, model, model_args=model_args,
                                            model_kwargs=model_kwargs, sy=sy,
                                            jitter=jitter, outliers=outliers,
                                            model_call_args=model_call_args,
                                            model_call_kwargs=model_call_kwargs)
        if not is_scipy:
            raise ImportError("Install scipy to use " + str(self.__class__))

    def _lnprob1(self, p):
        """
        With estimated uncertainties.

        :param p:
            Array-like of the parameters.

        """
        # Gaussian noise
        #return (np.log(0.5 * (1. + erf((self.y - self.model(p)) /
        #        (math.sqrt(2.) * self.sy))))).sum()
        model = self.model(p, *self.model_call_args, **self.model_call_kwargs)
        # Using scipy.stats.rice
        return rice.logcdf(self.y, model, scale=self.sy).sum()

    def _lnprob2(self, p):
        """
        With estimated uncertainties plus jitter.

        :param p:
            Array-like of the parameters, where:
            p[-1] - variance of jitter.
        """
        # Gaussian noise
        #return (np.log(0.5 * (1. + erf((self.y - self.model(p)) /
        #        (np.sqrt(2. * (self.sy ** 2. + p[-1]))))))).sum()
        model = self.model(p, *self.model_call_args, **self.model_call_kwargs)

        # Using scipy.stats.rice
        return rice.logcdf(self.y, model,
                           scale=np.sqrt(self.sy ** 2. + p[-1])).sum()

    def _lnprob3(self, p):
        """
        Without estimated uncertainties.

        :param p:
            Array-like of the parameters, where:
            p[-1] - variance of data.
        """
        return (np.log(0.5 * (1. + erf((self.y - self.model(p)) /
                (math.sqrt(2.) * p[-1]))))).sum()


# TODO: First, do for gaussian errors.
class LnProbDetectionsOutliers(LnProb):
    """
    Lnlikelihood for detections with outliers.
    """
    def __init__(self, x, y, model, sy=None, noise=None):
        super(LnProbDetections, self).__init__(x, y, model, sy=sy)

    def _lnprob1(self, p):
        pass

    def _lnprob2(self, p):
        pass

    def _lnprob3(self, p):
        pass


class LnPrior(object):
    """
    Class that represents prior pdf for parameters.

    :param lnprs:
        Tuple of tuples (callable, args, kwargs,) where args & kwargs -
        additional arguments to callable. Each callable is called callable(p,
        *args, **kwargs).

        Example:
            ((scipy.stats.norm.logpdf, [mu, s], dict(),),
            (scipy.stats.beta.logpdf, [alpha, beta], dict(),),
            (scipy.stats.uniform.logpdf, [a, b - a], dict(),),
            (scipy.stats.lognorm.logpdf, [i don't know:)], dict,),)

        First tuple will result in calling: scipy.stats.norm.logpdf(x, mu, s)

    """
    def __init__(self, lnprs):
        self.lnprs = [_function_wrapper(func, args, kwargs) for func, args,
                      kwargs in lnprs]

    def __call__(self, p):
        """
        Returns ln of prior pdf for given parameter vector.
        """
        return sum([self.lnprs[i](p_) for i, p_ in enumerate(p)])


class _function_wrapper(object):
    """
    This is a hack to make the likelihood function pickleable when ``args``
    and ``kwargs``are also included.

    """
    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        try:
            return self.f(x, *self.args, **self.kwargs)
        except:
            import traceback
            print("uvmod: Exception while calling your prior pdf:")
            print("  params:", x)
            print("  args:", self.args)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise


class LS_estimates(object):
    """
    Class that implements estimates of parameters via least squares method.

    The algorithm uses the Levenberg-Marquardt algorithm through `leastsq`.
    Additional keyword arguments are passed directly to that algorithm. It is
    almost scipy.optimize.curve_fit but with different arguments of function.

    :param x:
        Vector of explanatory variable.

    :param y:
        Vector of response variable.

    :param model:
        Model class. Class with ``__call__`` method that accepts
        vector of parameters and returns vector of model values
        for given explanatory variables (defined e.g. in it's
        constructor).

    :param sy (optional):
        Vector of uncertainties. If not ``None`` then this vector will be used
        as relative weights in the least-squares problem. (default: ``None``)
    """

    def _weighted_residuals(cls, p, y, sy, model):
        return (y - model(p)) / sy

    def _residuals(cls, p, y, model):
        return (y - model(p))

    def __init__(self, x, y, model, sy=None, model_args=None,
                 model_kwargs=None):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.model = model
        if model_args is None:
            model_args = []
        if model_kwargs is None:
            model_kwargs = {}
        self.model_args = model_args
        self.model_kwargs = model_kwargs
        if sy is not None:
            self.sy = np.asarray(sy)
        else:
            self.sy = None

    def fit(self, p0=None):
        """
        LS for 2D data.

        :param p0:
            The starting estimate for the minimization.

        :return:
            Optimized vector of parameters and it's covariance matrix.
        """

        if not is_scipy:
            raise ImportError("Install ``scipy`` to use ``fit_1d`` method of "
                              + str(self.__class__))
        if p0 is None:
            raise Exception("Define starting estimate for minimization!")

        if self.sy is None:
            func, args = self._residuals, (self.y, self.model(self.x,
                                                              *self.model_args,
                                                              **self.model_kwargs),)
        else:
            func, args = self._weighted_residuals, (self.y, self.sy,
                                                    self.model(self.x,
                                                               *self.model_args,
                                                               **self.model_kwargs),)

        fit = leastsq(func, p0, args=args, full_output=True)
        (p, pcov, infodict, errmsg, ier) = fit

        if ier not in [1, 2, 3, 4]:
            msg = "Optimal parameters not found: " + errmsg
            raise RuntimeError(msg)

        if (len(self.y) > len(p0)) and pcov is not None:
            # Residual variance
            s_sq = (func(p, *args) ** 2.).sum() / (len(self.y) - len(p0))
            pcov *= s_sq
        else:
            pcov = np.nan

        return p, pcov
