import numpy as np
from joblib import wrap_non_picklable_objects
import numba as nb
import inspect
import numpy as np
import pandas as pd
from typing import Any
from copy import copy
from QuantGplearn import functions
from functools import wraps
import warnings

warnings.filterwarnings('ignore')

NoneType = type(None)

__all__ = ['make_function', 'raw_function_list']


class _Function(object):
    """
    函数对象，参数至少有一个为向量
    默认函数类型为，all，既可用于时序也可用于截面
    默认返回类型为数值，
    默认输入类型，数值向量或者标量

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    param_type : [{
                  'vector': {'category': (None, None), 'number': (None, None)},
                  'scalar': {'int': (int, int), 'float': (float, float)}
                  },]
    function_type : 'all', 'section', 'time_series‘
    return_type: 'number', 'category'

    """

    def __init__(self, function, name, arity, param_type=None, return_type='number', function_type='all'):
        self.function = function
        self.name = name
        self.arity = arity
        if param_type is None:
            # 默认不接受分类类型
            param_type = arity * [{'vector': {'number': (None, None)},
                                   'scalar': {'int': (None, None), 'float': (None, None)}}]
        else:
            # 防止长度不一
            if len(param_type) != arity:
                raise ValueError(
                    "length of param_type should be equal to arity, it should be {}, not {}"
                    .format(arity, len(param_type)))
        self.param_type = param_type
        if (return_type != 'number') and (return_type != 'category'):
            raise ValueError("return_type of function {} should be number or category, NOT {}"
                             .format(name, return_type))
        self.return_type = return_type
        self.function_type = function_type

    def __call__(self, *args):
        """
        调用函数特殊处理，
        参数仅接受标量，却传入向量
        则取向量第一个值为标量
        """
        for _param, _param_type in zip(args, self.param_type):
            if len(_param_type) == 1 and 'scalar' in _param_type and isinstance(_param, (list, np.ndarray)):
                _param = _param[0]
        return self.function(*args)

    def add_range(self, const_range):
        # 作用：替换掉参数中没有约束的范围，给所有标量限制范围
        # 若没有const_range, 则表明所有函数不接收常数， 去掉所有的const type
        if const_range is None:
            for i, _dict in enumerate(self.param_type):
                if 'vector' not in _dict:
                    raise ValueError("for None const range, vector type should in all function param")
                if 'scalar' in _dict:
                    self.param_type[i].pop('scalar')
            return
        if not isinstance(const_range, tuple):
            raise ValueError('const_range must be an tuple')
        _min, _max = const_range
        if not isinstance(_min, (int, float)):
            raise ValueError('const_range left must be an int, float')
        if not isinstance(_max, (int, float)):
            raise ValueError('const_range right must be an int, float')
        if _min > _max:
            raise ValueError('const_range left should le right')

        for i, _dict in enumerate(self.param_type):
            if 'scalar' in _dict:
                _scalar_range = _dict['scalar']
                if 'int' in _scalar_range:
                    _l = int(_min) if _scalar_range['int'][0] is None else int(_scalar_range['int'][0])
                    _r = int(_max) if _scalar_range['int'][1] is None else int(_scalar_range['int'][1])
                    self.param_type[i]['scalar']['int'] = (_l, _r)
                if 'float' in _scalar_range:
                    _l = float(_min) if _scalar_range['float'][0] is None else float(_scalar_range['float'][0])
                    _r = float(_max) if _scalar_range['float'][1] is None else float(_scalar_range['float'][1])
                    self.param_type[i]['scalar']['float'] = (_l, _r)

        return

    def is_point_mutation(self, candidate_func):
        # 检验某个待替换函数是否可以替换
        if not isinstance(candidate_func, _Function):
            raise ValueError("wrong type, it should be _Function style")
        # 带替换函数是否与该函数参数长度一致
        if len(candidate_func.param_type) != len(self.param_type):
            return False
        if self.return_type != candidate_func.return_type:
            return False

        # candidate函数的参数必须为待替换函数参数的子集
        # 要求替换和，函数的所有参数仍然合法
        for dict_self, dict_candi in zip(self.param_type, candidate_func.param_type):
            if len(dict_candi) <= len(dict_self):
                return False
            for upper_type in dict_self:
                if upper_type not in dict_candi:
                    return False
                else:
                    for lower_type in dict_self:
                        if lower_type not in dict_candi[upper_type]:
                            return False
                        else:
                            if upper_type == 'scalar':
                                if (dict_candi['scalar'][lower_type][0] > dict_self['scalar'][lower_type][0]) or (
                                        dict_candi['scalar'][lower_type][1] > dict_candi['scalar'][lower_type][1]):
                                    return False
        return True

def _groupby(gbx, func, *args, **kwargs):
    indices = np.argsort(gbx)
    gbx_sorted = gbx[indices]
    X = np.column_stack((np.arange(len(gbx)), gbx_sorted, *args))
    splits = np.split(X, np.unique(gbx_sorted, return_index=True)[1][1:])
    result_list = [func(*(split[:, 2:].T), **kwargs) for split in splits]
    result = np.hstack(result_list)
    return result[indices.argsort()]


# warp 用于多进程序列化，会降低进化效率
def make_function(*, function, name, arity, param_type=None, wrap=True, return_type='number', function_type='all'):
    """
       Parameters
       ----------
       function : callable

       name : str

       arity : int

       param_type : [{type: (, ), type: (, )}, ........]

       wrap : bool, optional (default=True)
       """

    if not isinstance(arity, int):
        raise ValueError('arity must be an int, got %s' % type(arity))
    if not isinstance(name, str):
        raise ValueError('name must be a string, got %s' % type(name))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))

    # check out param_type vector > scalar int > float
    if param_type is None:
        param_type = [None] * arity
    if not isinstance(param_type, list):
        raise ValueError('param_type must be list')
    if len(param_type) != arity:
        raise ValueError('len of param_type must be arity')
    # 保证函数中至少有一个向量
    vector_flag = False
    for i, _dict in enumerate(param_type):
        # 转换None type
        # 标记某一个参数是否可接受向量
        non_vector_param = True
        if _dict is None:
            param_type[i] = {'vector': {'category': (None, None), 'number': (None, None)},
                             'scalar': {'int': (None, None), 'float': (None, None)}}
        elif not isinstance(_dict, dict):
            raise ValueError('element in param_type {} must be dict'.format(i + 1))
        if len(_dict) > 2:
            raise ValueError('len of element in param_type {} must be 1, 2'.format(i + 1))
        for upper_type in _dict:
            if upper_type == 'vector':
                if not isinstance(_dict['vector'], dict):
                    raise ValueError('type of element in param_type {} must be {upper_type: {lower_type:( , )}}}'
                                     .format(i + 1))
                if len(_dict['vector']) == 0:
                    raise ValueError('length of upper_type dict in param_type {} should not be 0'.format(i + 1))
                vector_flag = True
                non_vector_param = False
                for lower_type in _dict['vector']:
                    if lower_type not in ['number', 'category']:
                        raise ValueError('key of vector in param_type {} must be number or category'.format(i + 1))
                    param_type[i]['vector'][lower_type] = (None, None)

            elif upper_type == 'scalar':
                if not isinstance(_dict['scalar'], dict):
                    raise ValueError('type of element in param_type {} must be {upper_type: {lower_type:( , )}}}'
                                     .format(i + 1))
                if len(_dict['scalar']) == 0:
                    raise ValueError('length of upper_type dict in param_type {} should not be 0'.format(i + 1))
                for lower_type in _dict['scalar']:
                    # print(lower_type)
                    if lower_type == 'int':
                        # print(111)
                        # continue
                        # if not isinstance(_dict['scalar']['int'], tuple):
                        #     raise ValueError('structure of lower_type in param_type {} must be {type: ( , )}}'
                        #                      .format(i + 1))
                        if not isinstance(_dict['scalar']['int'], list):
                            raise ValueError('structure of lower_type in param_type {} must be {type: ( , )}}'
                                             .format(i + 1))
                        # if len(_dict['scalar']['int']) != 2:
                        #     raise ValueError("len of lower_type's structure in param_type {} must be 2".format(i + 1))
                        # if not isinstance(_dict['scalar']['int'][0], (int, NoneType)):
                        #     raise ValueError("the first element in lower_type's structure in param_type {} "
                        #                      "must be None, int or float".format(i + 1))
                        # if not isinstance(_dict['scalar']['int'][1], (int, NoneType)):
                        #     raise ValueError("the second element in lower_type's structure in param_type {} "
                        #                      "must be None, int or float".format(i + 1))
                        # if isinstance(_dict['scalar']['int'][0], int) and isinstance(_dict['scalar']['int'][1], int) \
                        #         and _dict['scalar']['int'][1] < _dict['scalar']['int'][0]:
                        #     raise ValueError('the second element should ge the first element in param_type {}'
                        #                      .format(i + 1))

                    elif lower_type == 'float':
                        if not isinstance(_dict['scalar']['float'], tuple):
                            raise ValueError('structure of lower_type in param_type {} must be {type: ( , )}}'
                                             .format(i + 1))
                        if len(_dict['scalar']['float']) != 2:
                            raise ValueError("len of lower_type's structure in param_type {} must be 2".format(i + 1))
                        if not isinstance(_dict['scalar']['float'][0], (float, int, NoneType)):
                            raise ValueError("the first element in lower_type's structure in param_type {} "
                                             "must be None, int or float".format(i + 1))
                        if not isinstance(_dict['scalar']['float'][1], (float, int, NoneType)):
                            raise ValueError("the second element in lower_type's structure in param_type {} "
                                             "must be None, int or float".format(i + 1))
                        if isinstance(_dict['scalar']['float'][0], (int, float)) and \
                                isinstance(_dict['scalar']['float'][1], (int, float)) \
                                and _dict['scalar']['float'][1] < _dict['scalar']['float'][0]:
                            raise ValueError('the second element should ge the first element in param_type {}'
                                             .format(i + 1))
                    else:
                        # print(3333)
                        raise ValueError('key of scalar in param_type {} must be int or float'.format(i + 1))
            else:
                raise ValueError('key of element in param_type {} must be vector or scalar'.format(i + 1))

    if not vector_flag:
        raise ValueError('there is at least 1 vector in param_type {}'.format(i + 1))

    for i, _dict in enumerate(param_type):
        # 省略其它检查部分
        if 'scalar' in _dict:
            if 'int' in _dict['scalar']:
                # 检查参数类型为 int 的部分
                allowed_values = _dict['scalar'].get('allowed_values', None)
                if allowed_values is not None and not isinstance(allowed_values, list):
                    raise ValueError("allowed_values for scalar int must be a list")

                # 检查是否在允许的值范围内
                if allowed_values is not None and _dict['scalar']['int'] not in allowed_values:
                    raise ValueError(
                        f"scalar int value {_dict['scalar']['int']} is not in the allowed range {allowed_values}")

            elif 'float' in _dict['scalar']:
                # 检查参数类型为 float 的部分
                allowed_values = _dict['scalar'].get('allowed_values', None)
                if allowed_values is not None and not isinstance(allowed_values, list):
                    raise ValueError("allowed_values for scalar float must be a list")

                # 检查是否在允许的值范围内
                if allowed_values is not None and _dict['scalar']['float'] not in allowed_values:
                    raise ValueError(
                        f"scalar float value {_dict['scalar']['float']} is not in the allowed range {allowed_values}")

    # Check output shape
    # 生成测试数据
    args = []
    for _dict in param_type:
        if 'vector' in _dict:
            if 'number' in _dict['vector']:
                args.append(np.ones(10))
            else:
                args.append(np.array([1] * 10))
        elif 'scalar' in _dict:
            if 'int' in _dict['scalar']:
                args.append(((0 if _dict['scalar']['int'][1] is None else _dict['scalar']['int'][1]) +
                             (0 if _dict['scalar']['int'][0] is None else _dict['scalar']['int'][0])) // 2)
            else:
                args.append(((0 if _dict['scalar']['float'][1] is None else _dict['scalar']['float'][1]) +
                             (0 if _dict['scalar']['float'][0] is None else _dict['scalar']['float'][0])) // 2)

    try:
        function(*args)
    except (ValueError, TypeError):
        print(args)
        raise ValueError('supplied function %s does not support arity of %d.'
                         % (name, arity))
    if not hasattr(function(*args), 'shape'):
        raise ValueError('supplied function %s does not return a numpy array.'
                         % name)
    if function(*args).shape != (10,):
        raise ValueError('supplied function %s does not return same shape as '
                         'input vectors.' % name)
    if function(*args).dtype.type is np.float64 and return_type == 'category':
        raise ValueError('the return type should be category not {}'.format(function(*args).dtype.type))
    elif function(*args).dtype not in [float, int, np.int64] and return_type == 'number':
        raise ValueError('the return type should be category not {}'.format(function(*args).dtype.type))

    # Check closure for zero & negative input arguments
    args2 = []
    args3 = []
    for _dict in param_type:
        if 'vector' in _dict:
            # 兼容category向量
            args2.append(np.zeros(10))
            args3.append(-1 * np.ones(10))
        elif 'scalar' in _dict:
            if 'int' in _dict['scalar']:

                _temp = (((0 if _dict['scalar']['int'][1] is None else _dict['scalar']['int'][1]) +
                          (0 if _dict['scalar']['int'][0] is None else _dict['scalar']['int'][0])) // 2)
                args2.append(_temp)
                args3.append(_temp)
            else:
                _temp = (((0 if _dict['scalar']['float'][1] is None else _dict['scalar']['float'][1]) +
                          (0 if _dict['scalar']['float'][0] is None else _dict['scalar']['float'][0])) // 2)
                args2.append(_temp)
                args3.append(_temp)

    if not np.all(np.isnan(function(*args2)) | np.isfinite(function(*args2))):
        raise ValueError('supplied function %s does not have closure against '
                         'zeros in argument vectors.' % name)

    if not np.all(np.isnan(function(*args3)) | np.isfinite(function(*args3))):
        raise ValueError('supplied function %s does not have closure against '
                         'negatives in argument vectors.' % name)
    if wrap:
        return _Function(function=wrap_non_picklable_objects(function),
                         name=name,
                         arity=arity,
                         param_type=param_type,
                         return_type=return_type,
                         function_type=function_type)
    return _Function(function=function,
                     name=name,
                     arity=arity,
                     param_type=param_type,
                     return_type=return_type,
                     function_type=function_type)


def _protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        # 如果 x1 和 x2 都是列表或数组
        if isinstance(x1, (list, np.ndarray)) and isinstance(x2, (list, np.ndarray)):
            # 确保两个数组或列表的长度一致
            if len(x1) != len(x2):
                raise ValueError("x1 and x2 must have the same length")
            # 逐元素进行除法操作
            return np.array([np.divide(x1[i], x2[i]) if np.abs(x2[i]) > 0.001 else 1. for i in range(len(x1))])

        # 如果 x1 是列表或数组，x2 是单个值
        elif isinstance(x1, (list, np.ndarray)):
            return np.array([np.divide(x1[i], x2) if np.abs(x2) > 0.001 else 1. for i in range(len(x1))])

        # 如果 x2 是列表或数组，x1 是单个值
        elif isinstance(x2, (list, np.ndarray)):
            return np.array([np.divide(x1, x2[i]) if np.abs(x2[i]) > 0.001 else 1. for i in range(len(x2))])

        # 如果 x1 和 x2 都是单个值
        else:
            return np.divide(x1, x2) if np.abs(x2) > 0.001 else 1.


def _protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    if isinstance(x1, (list, np.ndarray)):
        # 如果是列表或数组，逐个元素应用 sqrt 操作
        return np.array([np.sqrt(np.abs(x)) for x in x1])
    else:
        # 如果是单个值，直接计算
        return np.sqrt(np.abs(x1))


def _protected_log(x1):
    """Closure of log for zero and negative arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        if isinstance(x1, (list, np.ndarray)):
            # 如果是列表或数组，逐个元素应用 log 操作
            return np.array([np.log(np.abs(x)) if np.abs(x) > 0.001 else 0 for x in x1])
        else:
            # 如果是单个值，直接计算
            return np.log(np.abs(x1)) if np.abs(x1) > 0.001 else 0.


def _protected_inverse(x1):
    """Closure of inverse for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        if isinstance(x1, (list, np.ndarray)):
            # 如果是列表或数组，逐个元素应用 inverse 操作
            return np.array([1. / x if np.abs(x) > 0.001 else 0 for x in x1])
        else:
            # 如果是单个值，直接计算
            return 1. / x1 if np.abs(x1) > 0.001 else 0.


def _sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    if isinstance(x1, (list, np.ndarray)):
        # 如果是列表或数组，逐个元素应用 sigmoid 操作
        return np.array([1 / (1 + np.exp(-x)) for x in x1])
    else:
        # 如果是单个值，直接计算
        return 1 / (1 + np.exp(-x1))


add2 = _Function(function=np.add, name='add', arity=2)
sub2 = _Function(function=np.subtract, name='sub', arity=2)
mul2 = _Function(function=np.multiply, name='mul', arity=2)
div2 = _Function(function=_protected_division, name='div', arity=2)
sqrt1 = _Function(function=_protected_sqrt, name='sqrt', arity=1)
log1 = _Function(function=_protected_log, name='log', arity=1)
neg1 = _Function(function=np.negative, name='neg', arity=1)
inv1 = _Function(function=_protected_inverse, name='inv', arity=1)
abs1 = _Function(function=np.abs, name='abs', arity=1)
max2 = _Function(function=np.maximum, name='max', arity=2)
min2 = _Function(function=np.minimum, name='min', arity=2)
sin1 = _Function(function=np.sin, name='sin', arity=1)
cos1 = _Function(function=np.cos, name='cos', arity=1)
tan1 = _Function(function=np.tan, name='tan', arity=1)
sig1 = _Function(function=_sigmoid, name='sig', arity=1)


#########################################################################################################
#########################################################################################################
#########################################################################################################

# 参数范围：本项目示例用的是BTCUSDT_PERP的1小时k线，为了确保参数取值具有经济学含义，把参数范围设置成一个list包含1天、3天、7天、14天、21天和30天
para_list = [24, 24 * 3, 24 * 7, 24 * 14, 24 * 21, 24 * 30]


# #### ALL FUNCTION #####

# #### TIME SERIES FUNCTION #####

def _ts_shift(X, d):
    d = d[0] if isinstance(d, (np.ndarray, list)) else d
    d = int(d)
    res = np.empty_like(X, dtype=np.float64)
    res.fill(np.nan)
    res[d:] = res[:-d]
    return res


ts_shift = functions.make_function(function=_ts_shift, name='ts_shift', arity=2, function_type='time_series',
                                   param_type=[{'vector': {'number': (None, None)}},
                                               {'scalar': {'int': para_list}}])


def _ts_delta(X, d):
    d = d[0] if isinstance(d, (np.ndarray, list)) else d
    d = int(d)
    res = np.empty_like(X, dtype=np.float64)
    res.fill(np.nan)
    res[d:] = X[d:] - X[:-d]
    return res


ts_delta = functions.make_function(function=_ts_delta, name='ts_delta', arity=2, function_type='time_series',
                                   param_type=[{'vector': {'number': (None, None)}},
                                               {'scalar': {'int': para_list}}])


def _ts_mom(X, d):
    d = d[0] if isinstance(d, (np.ndarray, list)) else d
    d = int(d)
    res = np.empty_like(X, dtype=np.float64)
    res.fill(np.nan)
    res[d:] = X[d:] / X[:-d] - 1
    return res


ts_mom = functions.make_function(function=_ts_mom, name='ts_mom', arity=2, function_type='time_series',
                                 param_type=[{'vector': {'number': (None, None)}},
                                             {'scalar': {'int': para_list}}])


def _ts_min(X, d):
    # 确保 d 是整数
    d = d[0] if isinstance(d, (np.ndarray, list)) else d
    d = int(d)
    d = len(X) - 1 if d >= len(X) else d
    res = np.empty_like(X, dtype=np.float64)
    res.fill(np.nan)
    res = pd.Series(X).rolling(d).min().values
    return res


ts_min = functions.make_function(function=_ts_min, name='ts_min', arity=2, function_type='time_series',
                                 param_type=[{'vector': {'number': (None, None)}},
                                             {'scalar': {'int': para_list}}])


def _ts_max(X, d):
    # 确保 d 是整数
    d = d[0] if isinstance(d, (np.ndarray, list)) else d
    d = int(d)
    d = len(X) - 1 if d >= len(X) else d

    # 为结果数组分配空间并初始化为 NaN
    res = np.empty_like(X, dtype=np.float64)
    res.fill(np.nan)

    res = pd.Series(X).rolling(d).max().values
    return res


ts_max = functions.make_function(function=_ts_max, name='ts_max', arity=2, function_type='time_series',
                                 param_type=[{'vector': {'number': (None, None)}},
                                             {'scalar': {'int': para_list}}])


@nb.njit(nb.int32[:](nb.float64[:], nb.int32), cache=True)
def rolling_argmax(arr, n):
    results = np.empty(len(arr), dtype=np.int32)
    for i, x in enumerate(arr):
        if i < n - 1:
            results[i] = np.nan
        else:
            results[i] = np.argmax(arr[i - n + 1: i + 1])
    return results


@nb.njit(nb.int32[:](nb.float64[:], nb.int32), cache=True)
def rolling_argmin(arr, n):
    results = np.empty(len(arr), dtype=np.int32)
    for i, x in enumerate(arr):
        if i < n - 1:
            results[i] = np.nan
        else:
            results[i] = np.argmin(arr[i - n + 1: i + 1])
    return results


@nb.njit(nb.int32[:](nb.float64[:], nb.int32), cache=True)
def rolling_argsort(arr, n):
    results = np.empty(len(arr), dtype=np.int32)
    for i, x in enumerate(arr):
        if i < n - 1:
            results[i] = np.nan
        else:
            results[i] = np.argsort(arr[i - n + 1: i + 1])[-1]
    return results


def _ts_argmax(X, d):
    # 检查 d 是否为数组，如果是，则取第一个元素
    if isinstance(d, np.ndarray):
        d = d[0]
    d = int(d)  # 转换为整数
    d = len(X) - 1 if d >= len(X) else d  # 处理边界情况

    res = pd.Series(rolling_argmax(X.astype(float), d)).values.astype(int)
    return res


ts_argmax = functions.make_function(function=_ts_argmax, name='ts_argmax', arity=2, function_type='time_series',
                                    param_type=[{'vector': {'number': (None, None)}},
                                                {'scalar': {'int': para_list}}])


def _ts_argmin(X, d):
    # 检查 d 是否为数组，如果是，则取第一个元素
    if isinstance(d, np.ndarray):
        d = d[0]
    d = int(d)  # 转换为整数
    n = len(X)
    d = n - 1 if d >= n else d  # 处理边界情况

    res = pd.Series(rolling_argmin(X.astype(float), d)).values.astype(int)
    return res


ts_argmin = functions.make_function(function=_ts_argmin, name='ts_argmax', arity=2, function_type='time_series',
                                    param_type=[{'vector': {'number': (None, None)}},
                                                {'scalar': {'int': para_list}}])


def _ts_rank(X, d):
    # 检查 d 是否为数组，如果是，则取第一个元素
    if isinstance(d, np.ndarray):
        d = d[0]
    d = int(d)  # 转换为整数
    n = len(X)
    d = n - 1 if d >= n else d
    res = pd.Series(rolling_argsort(X.astype(float), d)).values.astype(int) / d
    return res


ts_rank = functions.make_function(function=_ts_rank, name='ts_rank', arity=2, function_type='time_series',
                                  param_type=[{'vector': {'number': (None, None)}},
                                              {'scalar': {'int': para_list}}])


# ts_sum: 计算局部和
def _ts_sum(X, d):
    # 检查 d 是否为数组，如果是，则取第一个元素
    if isinstance(d, np.ndarray):
        d = d[0]
    d = int(d)  # 转换为整数
    n = len(X)
    d = n - 1 if d >= n else d  # 处理边界情况
    res = np.full(n, np.nan)  # 初始化结果数组
    res = pd.Series(X).rolling(d).sum().values
    return res


ts_sum = functions.make_function(function=_ts_sum, name='ts_sum', arity=2, function_type='time_series',
                                 param_type=[{'vector': {'number': (None, None)}},
                                             {'scalar': {'int': para_list}}])


# ts_stddev: 计算局部标准差
def _ts_std(X, d):
    # 检查 d 是否为数组，如果是，则取第一个元素
    if isinstance(d, np.ndarray):
        d = d[0]
    d = int(d)  # 转换为整数
    d = len(X) - 1 if d >= len(X) else d  # 处理边界情况
    res = np.full(len(X), np.nan)  # 初始化结果数组

    res = pd.Series(X).rolling(d).std(ddof=1).values
    return res


ts_std = functions.make_function(function=_ts_std, name='ts_std', arity=2, function_type='time_series',
                                 param_type=[{'vector': {'number': (None, None)}},
                                             {'scalar': {'int': para_list}}])


def _ts_corr(X, Y, d):
    # 检查 d 是否为数组，如果是，则取第一个元素
    if isinstance(d, np.ndarray):
        d = d[0]
    d = int(d)  # 转换为整数
    d = len(X) - 1 if d >= len(X) else d  # 处理边界情况
    res = np.full(len(X), np.nan)  # 初始化结果数组
    res = pd.Series(X).rolling(d).corr(pd.Series(Y)).values
    return res


ts_corr = functions.make_function(function=_ts_corr, name='ts_corr', arity=3, function_type='time_series',
                                  param_type=[{'vector': {'number': (None, None)}},
                                              {'vector': {'number': (None, None)}},
                                              {'scalar': {'int': para_list}}])


def _ts_cdlbodym(X, Y, d):
    # 检查 d 是否为数组，如果是，则取第一个元素
    if isinstance(d, np.ndarray):
        d = d[0]
    d = int(d)  # 转换为整数
    d = len(X) - 1 if d >= len(X) else d  # 处理边界情况
    S_body = pd.Series(X) - pd.Series(Y)
    S_body_up = pd.Series(np.where(S_body > 0, 1, 0), index=S_body.index)
    S_body_down = pd.Series(np.where(S_body < 0, -1, 0), index=S_body.index)
    S_body_up_sum = S_body_up.rolling(d).sum()
    S_body_down_sum = S_body_down.rolling(d).sum()
    res = (S_body_up_sum / (S_body_up_sum + S_body_down_sum)).values
    return res


ts_cdlbodym = functions.make_function(function=_ts_cdlbodym, name='ts_cdlbodym', arity=3, function_type='time_series',
                                      param_type=[{'vector': {'number': (None, None)}},
                                                  {'vector': {'number': (None, None)}},
                                                  {'scalar': {'int': para_list}}])


def _ts_bar_bs(X, Y, d):
    # 检查 d 是否为数组，如果是，则取第一个元素
    if isinstance(d, np.ndarray):
        d = d[0]
    d = int(d)  # 转换为整数
    d = len(X) - 1 if d >= len(X) else d  # 处理边界情况
    S_high = pd.Series(X)
    S_low = pd.Series(Y)
    S_bar_high = S_high - S_high.shift(1)
    S_bar_low = S_low - S_low.shift(1)
    bar_big_arr = np.where((S_bar_high > 0) & (S_bar_low < 0), 1, 0)
    bar_big = pd.Series(np.where(bar_big_arr, 1, 0), index=S_bar_high.index)
    bar_small_arr = np.where((S_bar_high < 0) & (S_bar_low > 0), 1, 0)
    bar_small = pd.Series(np.where(bar_small_arr, 1, 0), index=S_bar_low.index)
    bar_big_sum = bar_big.rolling(d).sum()
    bar_small_sum = bar_small.rolling(d).sum()
    res = (bar_big_sum / (bar_big_sum + bar_small_sum)).values
    return res


ts_bar_bs = functions.make_function(function=_ts_bar_bs, name='ts_bar_bs', arity=3, function_type='time_series',
                                    param_type=[{'vector': {'number': (None, None)}},
                                                {'vector': {'number': (None, None)}},
                                                {'scalar': {'int': para_list}}])


def _ts_aroon(X, Y, d):
    # 检查 d 是否为数组，如果是，则取第一个元素
    if isinstance(d, np.ndarray):
        d = d[0]
    d = int(d)  # 转换为整数
    d = len(X) - 1 if d >= len(X) else d  # 处理边界情况
    high_arr = X
    low_arr = Y
    factor_arr = (pd.Series(rolling_argmax(high_arr.astype(float), d)).values.astype(int) - pd.Series(
        rolling_argmin(low_arr.astype(float), d)).values.astype(int)) / d
    return factor_arr


ts_aroon = functions.make_function(function=_ts_aroon, name='ts_aroon', arity=3, function_type='time_series',
                                   param_type=[{'vector': {'number': (None, None)}},
                                               {'vector': {'number': (None, None)}},
                                               {'scalar': {'int': para_list}}])


def _ts_adx(X, Y, Z, d):
    # 检查 d 是否为数组，如果是，则取第一个元素
    if isinstance(d, np.ndarray):
        d = d[0]
    d = int(d)  # 转换为整数
    d = len(X) - 1 if d >= len(X) else d  # 处理边界情况
    S_high = pd.Series(X)
    S_low = pd.Series(Y)
    S_close = pd.Series(Z)
    tr_a = (S_high - S_close.shift(1)).abs()
    tr_b = (S_low - S_close.shift(1)).abs()
    tr_c = (S_high - S_low).abs()
    tr = pd.Series(np.maximum(np.maximum(tr_a, tr_b), tr_c), index=S_high.index)
    atr = tr.rolling(d).mean()
    up = S_high - S_high.shift(1)
    down = S_low.shift(1) - S_low
    up_move = pd.Series(np.where(up > down, np.maximum(up, 0), 0), index=S_low.index)
    down_move = pd.Series(np.where(down > up, np.maximum(down, 0), 0), index=S_low.index)
    plus_di = up_move.rolling(d).mean() / atr
    minus_di = down_move.rolling(d).mean() / atr
    res = (((plus_di - minus_di) / (plus_di + minus_di)).abs()).values
    return res


ts_adx = functions.make_function(function=_ts_adx, name='ts_adx', arity=4, function_type='time_series',
                                 param_type=[{'vector': {'number': (None, None)}},
                                             {'vector': {'number': (None, None)}},
                                             {'vector': {'number': (None, None)}},
                                             {'scalar': {'int': para_list}}])


def _ts_bopr(X, Y, Z, A, d):
    # 检查 d 是否为数组，如果是，则取第一个元素
    if isinstance(d, np.ndarray):
        d = d[0]
    d = int(d)  # 转换为整数
    d = len(X) - 1 if d >= len(X) else d  # 处理边界情况
    S_open = pd.Series(X)
    S_high = pd.Series(Y)
    S_low = pd.Series(Z)
    S_close = pd.Series(A)
    bop = (S_close - S_open) / (S_high - S_low)
    res = bop.rolling(d).mean().values
    return res


ts_bopr = functions.make_function(function=_ts_bopr, name='ts_bopr', arity=5, function_type='time_series',
                                  param_type=[{'vector': {'number': (None, None)}},
                                              {'vector': {'number': (None, None)}},
                                              {'vector': {'number': (None, None)}},
                                              {'vector': {'number': (None, None)}},
                                              {'scalar': {'int': para_list}}])


def _ts_one_ols_k(X, Y, d):
    # 检查 d 是否为数组，如果是，则取第一个元素
    if isinstance(d, np.ndarray):
        d = d[0]
    d = int(d)  # 转换为整数
    d = len(X) - 1 if d >= len(X) else d  # 处理边界情况
    S_x = pd.Series(X)
    S_y = pd.Series(Y)
    df = pd.concat([S_x, S_y], axis=1)
    df.columns = ["x", "y"]
    df["xy"] = df["x"] * df["y"]
    df["x2"] = df["x"] * df["x"]
    R = df.rolling(d).sum()
    beta = ((d * R["xy"] - R["x"] * R["y"]) / (d * R["x2"] - R["x"] * R["x"])).values
    return beta


ts_one_ols_k = functions.make_function(function=_ts_one_ols_k, name='ts_one_ols_k', arity=3,
                                       function_type='time_series',
                                       param_type=[{'vector': {'number': (None, None)}},
                                                   {'vector': {'number': (None, None)}},
                                                   {'scalar': {'int': para_list}}])


def _ts_one_ols_resid(X, Y, d):
    # 检查 d 是否为数组，如果是，则取第一个元素
    if isinstance(d, np.ndarray):
        d = d[0]
    d = int(d)  # 转换为整数
    d = len(X) - 1 if d >= len(X) else d  # 处理边界情况
    S_x = pd.Series(X)
    S_y = pd.Series(Y)
    df = pd.concat([S_x, S_y], axis=1)
    df.columns = ["x", "y"]
    df["xy"] = df["x"] * df["y"]
    df["x2"] = df["x"] * df["x"]
    R = df.rolling(d).sum()
    beta = ((d * R["xy"] - R["x"] * R["y"]) / (d * R["x2"] - R["x"] * R["x"])).values
    df['beta'] = beta
    resid = (R["y"] - beta * R["x"]).values
    return resid


ts_one_ols_resid = functions.make_function(function=_ts_one_ols_resid, name='ts_one_ols_resid', arity=3,
                                           function_type='time_series',
                                           param_type=[{'vector': {'number': (None, None)}},
                                                       {'vector': {'number': (None, None)}},
                                                       {'scalar': {'int': para_list}}])


def _ts_stochf(X, Y, Z, d):
    # 检查 d 是否为数组，如果是，则取第一个元素
    if isinstance(d, np.ndarray):
        d = d[0]
    d = int(d)  # 转换为整数
    d = len(X) - 1 if d >= len(X) else d  # 处理边界情况
    S_high = pd.Series(X)
    S_low = pd.Series(Y)
    S_close = pd.Series(Z)
    Lowest_Low = S_low.rolling(window=d).min()
    Highest_high = S_high.rolling(window=d).max()
    res = ((S_close - Lowest_Low) / (Highest_high - Lowest_Low)).values
    return res


ts_stochf = functions.make_function(function=_ts_stochf, name='ts_stochf', arity=4, function_type='time_series',
                                    param_type=[{'vector': {'number': (None, None)}},
                                                {'vector': {'number': (None, None)}},
                                                {'vector': {'number': (None, None)}},
                                                {'scalar': {'int': para_list}}])


def _ts_cmo(X, d):
    # 检查 d 是否为数组，如果是，则取第一个元素
    if isinstance(d, np.ndarray):
        d = d[0]
    d = int(d)  # 转换为整数
    d = len(X) - 1 if d >= len(X) else d  # 处理边界情况
    S_price = pd.Series(X)
    S_price_diff = S_price.diff()
    Up_Move = pd.Series(np.where(S_price_diff > 0, S_price_diff, 0), index=S_price.index)
    Down_Move = pd.Series(np.where(S_price_diff < 0, abs(S_price_diff), 0), index=S_price.index)
    sum_up = Up_Move.rolling(d).mean()
    sum_down = Down_Move.rolling(d).mean()
    res = ((sum_up - sum_down) / (sum_up + sum_down)).values
    return res


ts_cmo = functions.make_function(function=_ts_cmo, name='ts_cmo', arity=2, function_type='time_series',
                                 param_type=[{'vector': {'number': (None, None)}},
                                             {'scalar': {'int': para_list}}])


def _ts_ema(X, d):
    # 检查 d 是否为数组，如果是，则取第一个元素
    if isinstance(d, np.ndarray):
        d = d[0]
    d = int(d)  # 转换为整数
    d = len(X) - 1 if d >= len(X) else d  # 处理边界情况
    sv = X
    fv = np.full(sv.shape, np.nan)
    alpha = 2 / (d + 1)
    for i in (range(d, len(sv))):
        s_c = sv[i - d + 1: i + 1]
        ev = 0
        for j in range(len(s_c)):
            if j > 0:
                ev = alpha * s_c[j] + (1 - alpha) * ev
            else:
                ev = s_c[j]
        fv[i] = ev
    return fv


ts_ema = functions.make_function(function=_ts_ema, name='ts_ema', arity=2, function_type='time_series',
                                 param_type=[{'vector': {'number': (None, None)}},
                                             {'scalar': {'int': para_list}}])


def _ts_rsi(X, d):
    # 检查 d 是否为数组，如果是，则取第一个元素
    if isinstance(d, np.ndarray):
        d = d[0]
    d = int(d)  # 转换为整数
    d = len(X) - 1 if d >= len(X) else d  # 处理边界情况
    S_price = pd.Series(X)
    S_price_diff = S_price / S_price.shift(1) - 1
    rsi_pos = pd.Series(np.where(S_price_diff > 0, S_price_diff, 0), index=S_price.index)
    rsi_neg = pd.Series(np.where(S_price_diff < 0, S_price_diff, 0), index=S_price.index)
    rsi_pos_sum = rsi_pos.rolling(d).sum()
    rsi_neg_sum = rsi_neg.rolling(d).sum()
    fv = (rsi_pos_sum / rsi_neg_sum).values
    return fv


ts_rsi = functions.make_function(function=_ts_rsi, name='ts_rsi', arity=2, function_type='time_series',
                                 param_type=[{'vector': {'number': (None, None)}},
                                             {'scalar': {'int': para_list}}])


def _ts_xs_ratio(X, d):
    # 检查 d 是否为数组，如果是，则取第一个元素
    if isinstance(d, np.ndarray):
        d = d[0]
    d = int(d)  # 转换为整数
    d = len(X) - 1 if d >= len(X) else d  # 处理边界情况
    S_price = pd.Series(X)
    x_series = (S_price - S_price.shift(d)).abs()
    s_series = (S_price - S_price.shift(1)).abs().rolling(d).sum()
    fv = (x_series / s_series).values
    return fv


ts_xs_ratio = functions.make_function(function=_ts_xs_ratio, name='ts_xs_ratio', arity=2, function_type='time_series',
                                      param_type=[{'vector': {'number': (None, None)}},
                                                  {'scalar': {'int': para_list}}])


def _ts_macd(X, d1, d2, d3):
    # 检查 d 是否为数组，如果是，则取第一个元素
    if isinstance(d1, np.ndarray):
        d1 = d1[0]
    d1 = int(d1)  # 转换为整数
    d1 = len(X) - 1 if d1 >= len(X) else d1  # 处理边界情况

    if isinstance(d2, np.ndarray):
        d2 = d2[0]
    d2 = int(d2)  # 转换为整数
    d2 = len(X) - 1 if d2 >= len(X) else d2  # 处理边界情况

    if isinstance(d3, np.ndarray):
        d3 = d3[0]
    d3 = int(d3)  # 转换为整数
    d3 = len(X) - 1 if d3 >= len(X) else d3  # 处理边界情况

    S_price = pd.Series(X)
    S_short_ma = S_price.rolling(d1).mean()
    S_long_ma = S_price.rolling(d2).mean()
    S_cd = S_short_ma - S_long_ma
    fv = S_cd.rolling(d3).mean().values
    return fv


ts_macd = functions.make_function(function=_ts_macd, name='ts_macd', arity=4, function_type='time_series',
                                  param_type=[{'vector': {'number': (None, None)}},
                                              {'scalar': {'int': para_list}},
                                              {'scalar': {'int': para_list}},
                                              {'scalar': {'int': para_list}}])


def _ts_atr(X, Y, Z, d1, d2):
    # 检查 d 是否为数组，如果是，则取第一个元素
    if isinstance(d1, np.ndarray):
        d1 = d1[0]
    d1 = int(d1)  # 转换为整数
    d1 = len(X) - 1 if d1 >= len(X) else d1  # 处理边界情况

    if isinstance(d2, np.ndarray):
        d2 = d2[0]
    d2 = int(d2)  # 转换为整数
    d2 = len(X) - 1 if d2 >= len(X) else d2  # 处理边界情况

    S_high = pd.Series(X)
    S_low = pd.Series(Y)
    S_close = pd.Series(Z)
    tr_windows = d1
    mean_windows = d2
    S_high_max = S_high.rolling(tr_windows).max()
    S_low_min = S_low.rolling(tr_windows).min()
    tr_a = ((S_high_max - S_close.shift(tr_windows)).abs()).values
    tr_b = ((S_low_min - S_close.shift(tr_windows)).abs()).values
    tr_c = (S_high_max - S_low_min).values
    tr = pd.Series(np.maximum(np.maximum(tr_a, tr_b), tr_c))
    fv = (tr.rolling(mean_windows).mean() / S_close).values
    return fv


ts_atr = functions.make_function(function=_ts_atr, name='ts_atr', arity=5, function_type='time_series',
                                 param_type=[{'vector': {'number': (None, None)}},
                                             {'vector': {'number': (None, None)}},
                                             {'vector': {'number': (None, None)}},
                                             {'scalar': {'int': para_list}},
                                             {'scalar': {'int': para_list}}])


def _ts_hedge(X, Y, d1, d2):
    # 检查 d 是否为数组，如果是，则取第一个元素
    if isinstance(d1, np.ndarray):
        d1 = d1[0]
    d1 = int(d1)  # 转换为整数
    d1 = len(X) - 1 if d1 >= len(X) else d1  # 处理边界情况

    if isinstance(d2, np.ndarray):
        d2 = d2[0]
    d2 = int(d2)  # 转换为整数
    d2 = len(X) - 1 if d2 >= len(X) else d2  # 处理边界情况

    mean_windows = d1
    hedge_ratio = d2
    hedge_len = int(mean_windows * hedge_ratio)
    res = np.empty_like(X, dtype=np.float64)
    res.fill(np.nan)
    for i in range(mean_windows - 1, len(X)):
        x = X[i - mean_windows + 1: i + 1]
        y = Y[i - mean_windows + 1: i + 1]
        two_arr = np.column_stack([x, y])
        two_desc_arr = two_arr[two_arr[:, 0].argsort(kind="stable")[::-1]]
        res[i] = (two_desc_arr[:hedge_len, 1].mean() - two_desc_arr[- hedge_len:, 1].mean()) / X[i]
    return res


ts_hedge = functions.make_function(function=_ts_hedge, name='ts_hedge', arity=4, function_type='time_series',
                                   param_type=[{'vector': {'number': (None, None)}},
                                               {'vector': {'number': (None, None)}},
                                               {'scalar': {'int': para_list}},
                                               {'scalar': {'int': para_list}}])


# ts_mean: 计算局部均值
def _ts_mean(X, d):
    # 检查 d 是否为数组，如果是，则取第一个元素
    if isinstance(d, np.ndarray):
        d = d[0]
    d = int(d)  # 转换为整数
    d = len(X) - 1 if d >= len(X) else d  # 处理边界情况
    res = np.empty_like(X, dtype=np.float64)
    res.fill(np.nan)
    res = pd.Series(X).rolling(d).mean().values
    return res


ts_mean = functions.make_function(function=_ts_mean, name='ts_mean', arity=2,
                                  function_type='time_series',
                                  param_type=[{'vector': {'number': (None, None)}},
                                              {'scalar': {'int': para_list}}])


def _ts_skew(X, d):
    # 检查 d 是否为数组，如果是，则取第一个元素
    if isinstance(d, np.ndarray):
        d = d[0]
    d = int(d)  # 转换为整数
    d = len(X) - 1 if d >= len(X) else d  # 处理边界情况
    res = np.empty_like(X, dtype=np.float64)
    res.fill(np.nan)
    res = pd.Series(X).rolling(d).skew().values
    return res


ts_skew = functions.make_function(function=_ts_skew, name='ts_skew', arity=2,
                                  function_type='time_series',
                                  param_type=[{'vector': {'number': (None, None)}},
                                              {'scalar': {'int': para_list}}])


def _ts_kurt(X, d):
    # 检查 d 是否为数组，如果是，则取第一个元素
    if isinstance(d, np.ndarray):
        d = d[0]
    d = int(d)  # 转换为整数
    d = len(X) - 1 if d >= len(X) else d  # 处理边界情况
    res = np.empty_like(X, dtype=np.float64)
    res.fill(np.nan)
    res = pd.Series(X).rolling(d).skew().values
    return res


ts_kurt = functions.make_function(function=_ts_kurt, name='ts_kurt', arity=2,
                                  function_type='time_series',
                                  param_type=[{'vector': {'number': (None, None)}},
                                              {'scalar': {'int': para_list}}])


def _ts_zscore(X, d):
    # 检查 d 是否为数组，如果是，则取第一个元素
    if isinstance(d, np.ndarray):
        d = d[0]
    d = int(d)  # 转换为整数
    N = len(X)
    d = N - 1 if d >= N else d  # 处理边界情况
    res = ((pd.Series(X) - pd.Series(X).rolling(d).mean()) / pd.Series(X).rolling(d).std(ddof=1)).values
    return res


ts_zscore = functions.make_function(function=_ts_zscore, name='ts_zscore', arity=2,
                                    function_type='time_series',
                                    param_type=[{'vector': {'number': (None, None)}},
                                                {'scalar': {'int': para_list}}])


def _ts_bband(X, d1, d2):
    # 检查 d 是否为数组，如果是，则取第一个元素
    if isinstance(d1, np.ndarray):
        d1 = d1[0]
    d1 = int(d1)  # 转换为整数
    d1 = len(X) - 1 if d1 >= len(X) else d1  # 处理边界情况

    if isinstance(d2, np.ndarray):
        d2 = d2[0]
    d2 = int(d2)  # 转换为整数
    d2 = len(X) - 1 if d2 >= len(X) else d2  # 处理边界情况
    S = pd.Series(X)
    ma_series = S.rolling(d1).mean()
    std_series = S.rolling(d1).std(ddof=1)
    res = (ma_series + d2 * std_series).values
    return res


ts_bband = functions.make_function(function=_ts_bband, name='ts_bband', arity=3,
                                   function_type='time_series',
                                   param_type=[{'vector': {'number': (None, None)}},
                                               {'scalar': {'int': para_list}},
                                               {'scalar': {'int': para_list}}])


# ts_freq: 计算局部元素频率
@nb.njit(nb.int32[:](nb.float64[:], nb.int32), cache=True)
def rolling_freq(arr, n):
    results = np.empty(len(arr), dtype=np.int32)
    for i, x in enumerate(arr):
        if i < n - 1:
            results[i] = np.nan
        else:
            results[i] = np.sum(arr[i - n + 1: i + 1] == arr[i])
    return results


def _ts_freq(X, d):
    # 检查 d 是否为数组，如果是，则取第一个元素
    if isinstance(d, np.ndarray):
        d = d[0]
    d = int(d)  # 转换为整数
    d = len(X) - 1 if d >= len(X) else d  # 处理边界情况
    res = pd.Series(rolling_freq(X.astype(float), d)).values.astype(int)
    return res


ts_freq = functions.make_function(function=_ts_freq, name='ts_freq', arity=2,
                                  function_type='time_series',
                                  param_type=[{'vector': {'category': (None, None)}},
                                              {'scalar': {'int': para_list}}])

_function_map = {'add': add2,
                 'sub': sub2,
                 'mul': mul2,
                 'div': div2,
                 'sqrt': sqrt1,
                 'log': log1,
                 'abs': abs1,
                 'neg': neg1,
                 'inv': inv1,
                 'max': max2,
                 'min': min2,
                 'sin': sin1,
                 'cos': cos1,
                 'tan': tan1,
                 'sig': sig1,

                 # ts
                 'ts_shift': ts_shift,
                 'ts_delta': ts_delta,
                 'ts_mom': ts_mom,
                 'ts_min': ts_min,
                 'ts_max': ts_max,
                 'ts_argmax': ts_argmax,
                 'ts_argmin': ts_argmin,
                 'ts_rank': ts_rank,
                 'ts_sum': ts_sum,
                 'ts_std': ts_std,
                 'ts_corr': ts_corr,
                 'ts_mean': ts_mean,
                 'ts_zscore': ts_zscore,
                 'ts_freq': ts_freq,
                 'ts_cdlbodym': ts_cdlbodym,
                 'ts_bar_bs': ts_bar_bs,
                 'ts_adx': ts_adx,
                 'ts_aroon': ts_aroon,
                 'ts_bopr': ts_bopr,
                 'ts_cmo': ts_cmo,
                 'ts_ema': ts_ema,
                 'ts_macd': ts_macd,
                 'ts_rsi': ts_rsi,
                 'ts_stochf': ts_stochf,
                 'ts_xs_ratio': ts_xs_ratio,
                 'ts_one_ols_k': ts_one_ols_k,
                 'ts_one_ols_resid': ts_one_ols_resid,
                 'ts_skew': ts_skew,
                 'ts_kurt': ts_kurt,
                 'ts_atr': ts_atr,
                 'ts_hedge': ts_hedge
                 }

raw_function_list = ['add', 'sub', 'mul', 'div', 'sqrt',
                     'log', 'abs', 'neg', 'inv',
                     'max', 'min',
                     'sig'
                     ]

all_function = raw_function_list.copy()

section_function = []

time_series_function = [
    'ts_shift',
    'ts_delta',
    'ts_mom',
    'ts_min',
    'ts_max',
    'ts_argmax',
    'ts_argmin',
    'ts_rank',
    'ts_sum',
    'ts_std',
    'ts_corr',
    'ts_mean',
    'ts_zscore',
    # 'ts_freq',
    'ts_cdlbodym',
    'ts_bar_bs',
    'ts_adx',
    'ts_aroon',
    'ts_bopr',
    'ts_cmo',
    'ts_macd',
    'ts_rsi',
    'ts_stochf',
    'ts_xs_ratio',
    'ts_one_ols_k',
    'ts_one_ols_resid',
    'ts_skew',
    'ts_kurt',
    'ts_atr',
    'ts_hedge'
]

all_function.extend(time_series_function)
