from copy import copy, deepcopy
import numpy as np
from sklearn.utils.random import sample_without_replacement

from .functions import _Function, _groupby
from .utils import check_random_state
import logging


class _Program(object):
    '''

    '''

    def __init__(self,
                 function_dict,
                 arities,
                 init_depth,
                 init_method,
                 n_features,
                 const_range,
                 metric,
                 p_point_replace,
                 parsimony_coefficient,
                 random_state,
                 data_type,
                 n_cat_features,
                 transformer=None,
                 feature_names=None,
                 program=None):
        '''

        Parameters
        ----------
        function_dict: 储存基础函数，原为function_set {'number': [], 'category': []}
        arities: 函数参数个数
        init_depth：初始深度, 接受元组（min_depth, max_depth）
        init_method：生成方式，
        n_features：特征个数
        const_range：常数范围, (-1, 1)
        metric：目标函数，’MAE‘,'MSE'
        p_point_replace：点变异概率
        parsimony_coefficient:惩罚系数，'auto'护着浮点数，默认0.01
        random_state：随机对象
        data_type：新增参数 截面，时序or面板， ’section‘， ’time_series', 'panel'
        n_cat_features：新增参数 分类特征个数
        transformer
        feature_names
        program
        '''
        self.function_dict = function_dict
        self.arities = arities
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.init_method = init_method
        self.n_features = n_features
        self.const_range = const_range
        self.metric = metric
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.data_type = data_type
        self.transformer = transformer
        self.feature_names = feature_names
        self.program = program
        self.n_cat_features = n_cat_features

        self.num_func_number = len(self.function_dict['number'])
        self.cat_func_number = len(self.function_dict['category'])

        if self.program is not None:
            # 验证当下树是否完整
            if not self.validate_program():
                raise ValueError('The supplied program is incomplete.')
        else:
            # Create a naive random program
            self.program = self.build_program(random_state)

        self.raw_fitness_ = None
        self.fitness_ = None
        self.parents = None
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None

    def generate_my_output(self):
        """Generates the LISP tree output and stores it in self.output."""
        terminals = [0]
        my_output = ''
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                terminals.append(node.arity)
                my_output += node.name + '('
            else:
                if isinstance(node, str):
                    if self.feature_names is None:
                        my_output += 'X%s' % node
                    else:
                        my_output += self.feature_names[int(node)]
                elif isinstance(node, int):
                    my_output += '%d' % node
                elif isinstance(node, float):
                    my_output += '%.3f' % node
                else:
                    raise ValueError('Error param type {}'.format(node))
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    my_output += ')'
                if i != len(self.program) - 1:
                    my_output += ', '
        return my_output
        # self.my_output = my_output  # 存储到 self.output

    def build_program(self, random_state, type='number', first_build=True):
        """
        参数中无program 初始化方法
        # v1.55 修改数的生成逻辑
        :param random_state: RandomState 对象， 随机数生成器
        :param type: 生成树返回数值还是分类
        :return: list,
        """
        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method
        max_depth = random_state.randint(*self.init_depth)

        # Start a program with a function to avoid degenerative programs
        # 公式树返回类型必须为数值类型，随机挑选一个返回数值向量的函数作为公式树的根节点
        if first_build == True:
            valid_functions = [func for func in self.function_dict['number'] if
                                func.name not in ['ts_std', 'ts_kurt', 'ts_atr']]
            _root_function = random_state.choice(valid_functions)
        else:
            _root_function_num = random_state.randint(len(self.function_dict['number']))
            _root_function = self.function_dict['number'][_root_function_num]

        # 初始化公式树和工作栈，当前工作栈中仅有根节点,工作栈中存储参数类型列表，用于树的生成
        program = [_root_function]
        terminal_stack = [deepcopy(_root_function.param_type)]

        while terminal_stack:
            depth = len(terminal_stack)
            candidate_num = self.n_features + self.num_func_number + self.cat_func_number
            candidate_choice = random_state.randint(candidate_num)
            # Determine if we are adding a function or terminal
            # terminal_stack的元素必须是list
            if not isinstance(terminal_stack[-1], list):
                raise ValueError("element in terminal_stack should be list")
            # terminal_stack的元素的list内，元素须为dict
            if not isinstance(terminal_stack[-1][0], dict):
                raise ValueError("element in terminal_stack'element should be dict")

            # 深度优先的方式构建公式树，迭代处理工作栈中最后一个子树第一个子节点
            # 与gplearn主要不同点
            if ('vector' in terminal_stack[-1][0]) and (depth < max_depth) \
                    and (method == 'full' or candidate_choice < (self.num_func_number + self.cat_func_number)):
                # 插入函数的要求，1 该节点必须接受向量，2.当前深度比最大深度低， 3.随机种子选中了函数或者模式为‘full’
                
                # 决定选择数值型函数 还是 分类型函数
                # 若该节点都可以接受，则随机决定插入的函数类型
                # 否则根据可接受类型插入相应函数
                _choice = random_state.randint(self.cat_func_number + self.num_func_number)
                # 下面四行临时注释掉
                # if 'number' in terminal_stack[-1][0]['vector'] and 'category' in terminal_stack[-1][0]['vector']:
                #     key = 'number' if _choice < self.num_func_number else 'category'
                # else:
                #     key = 'number' if 'number' in terminal_stack[-1][0]['vector'] else 'category'

                # print(self.num_func_number, self.cat_func_number)
                key='number'
                function_choice = self.function_dict[key][_choice %
                                                   (self.num_func_number if key == 'number' else self.cat_func_number)]
                program.append(function_choice)
                terminal_stack.append(deepcopy(function_choice.param_type))
            else:
                # 插入向量或者常量
                _choice = random_state.randint(self.n_features + 1)
                # 根据特殊情况调整_choice
                # 1.若const_range为None 或者 不接受标量类型，则默认插入向量
                # 2.若不接受向量类型，则默认插入标量
                # 3.其他情况按照随机数决定
                if _choice == self.n_features and \
                        ((self.const_range is None) or \
                        (('scalar') not in terminal_stack[-1][0])):
                    # 只能插入向量的情况
                    if 'vector' not in terminal_stack[-1][0]:
                        raise ValueError('Error param type {}'.format(terminal_stack[-1][0]))

                    _choice = random_state.randint(self.n_features)
                elif ('vector' not in terminal_stack[-1][0]):
                    # 只能插入常量的情况
                    _choice = self.n_features

                if _choice < self.n_features:
                    # 插入向量
                    if 'number' in terminal_stack[-1][0]['vector'] and 'category' in terminal_stack[-1][0][
                        'vector']:
                        # 可插入数值向量也可插入分类向量
                        key = 'category' if _choice < self.n_cat_features else 'number'
                    else:
                        key = 'number' if 'number' in terminal_stack[-1][0]['vector'] else 'category'
                    if self.n_cat_features == 0 and key == 'category':
                        # 需要插入分类向量，特征中却没有分类向量的情况，插入常数分类向量1, 默认0
                        candicate_var = 0
                    else:
                        candicate_var = (_choice % self.n_cat_features) + 1 if key == 'category' else \
                                ((_choice % (self.n_features - self.n_cat_features) + self.n_cat_features) + 1)
                    program.append(str(candicate_var))
                else:
                    # 插入常量
                    if 'float' in terminal_stack[-1][0]['scalar']:
                        _choice = random_state.uniform(*terminal_stack[-1][0]['scalar']['float'])
                    # elif 'int' in terminal_stack[-1][0]['scalar']:
                    #     _choice = random_state.randint(*terminal_stack[-1][0]['scalar']['int'])
                    #     _choice = random_state.randint(*terminal_stack[-1][0]['scalar']['int'])
                    elif 'int' in terminal_stack[-1][0]['scalar']:
                        # if isinstance(terminal_stack[-1][0]['scalar']['int'], tuple):
                        #     _choice = random_state.randint(*terminal_stack[-1][0]['scalar']['int'])
                        # if isinstance(terminal_stack[-1][0]['scalar']['int'], list):
                        # print(random_state)
                        _choice = random_state.choice(terminal_stack[-1][0]['scalar']['int'])
                    else:
                        raise ValueError('Error param type {}'.format(terminal_stack[-1][0]))
                    program.append(_choice)

                terminal_stack[-1].pop(0)
                while len(terminal_stack[-1]) == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1].pop(0)
        # We should never get here
        return None

    # 检查函数是否可用，不包括类型检查
    def validate_program(self):
        """Rough check that the embedded program in the object is valid."""
        terminals = [0]
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]

    # 打印树
    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        terminals = [0]
        output = ''
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                terminals.append(node.arity)
                output += node.name + '('
            else:
                if isinstance(node, str):
                    if self.feature_names is None:
                        output += 'X%s' % node
                    else:
                        output += self.feature_names[int(node)]
                # elif isinstance(node, int):
                elif isinstance(node, (int, np.integer)):
                    output += '%d' % node
                elif isinstance(node, float):
                    output += '%.3f' % node
                else:
                    print(type(node))
                    raise ValueError('Error param type {}'.format(node))
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    output += ')'
                if i != len(self.program) - 1:
                    output += ', '
        return output

    # 可视化整个树
    def export_graphviz(self, fade_nodes=None):
        """Returns a string, Graphviz script for visualizing the program.

        Parameters
        ----------
        fade_nodes : list, optional
            A list of node indices to fade out for showing which were removed
            during evolution.

        Returns
        -------
        output : string
            The Graphviz script to plot the tree representation of the program.

        """
        terminals = []
        if fade_nodes is None:
            fade_nodes = []
        output = 'digraph program {\nnode [style=filled]\n'
        for i, node in enumerate(self.program):
            fill = '#cecece'
            if isinstance(node, _Function):
                if i not in fade_nodes:
                    fill = '#136ed4'
                terminals.append([node.arity, i])
                output += ('%d [label="%s", fillcolor="%s"] ;\n'
                           % (i, node.name, fill))
            else:
                if i not in fade_nodes:
                    fill = '#60a6f6'

                if isinstance(node, str):
                    if self.feature_names is None:
                        feature_name = 'X%s' % node
                    else:
                        feature_name = self.feature_names[int(node)]
                    output += ('%d [label="%s", fillcolor="%s"] ;\n'
                               % (i, feature_name, fill))
                elif isinstance(node, int):
                    output += ('%d [label="%d", fillcolor="%s"] ;\n'
                               % (i, node, fill))
                elif isinstance(node, int):
                    output += ('%d [label="%.3f", fillcolor="%s"] ;\n'
                               % (i, node, fill))
                else:
                    raise ValueError('Error param type {}'.format(node))

                if i == 0:
                    # A degenerative program of only one node
                    return output + '}'
                terminals[-1][0] -= 1
                terminals[-1].append(i)
                while terminals[-1][0] == 0:
                    output += '%d -> %d ;\n' % (terminals[-1][1],
                                                terminals[-1][-1])
                    terminals[-1].pop()
                    if len(terminals[-1]) == 2:
                        parent = terminals[-1][-1]
                        terminals.pop()
                        if not terminals:
                            return output + '}'
                        terminals[-1].append(parent)
                        terminals[-1][0] -= 1

        # We should never get here
        return None

    # 计算树的深度
    def _depth(self):
        """Calculates the maximum depth of the program tree."""
        terminals = [0]
        depth = 1
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    # 计算公式中函数和变量的数量
    def _length(self):
        """Calculates the number of functions and terminals in the program."""
        return len(self.program)

    # 计算参数X的函数结果
    def execute(self, X):
        """Execute the program according to X.

        Parameters
        ----------
        X : {array-like}
            若数据类型为'section'，'time_series'则为[n_samples, n_features + 1]
            若数据类型为'panel', 则为[n_samples, n_features + 3]

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.

        """
        # 检验X列数是否正确
        if self.data_type == 'panel' and X.shape[1] != self.n_features + 3:
            raise ValueError("For panel Data, the col number of X should be n_features + 3")
        elif self.data_type in ['section', 'time_series'] and X.shape[1] != self.n_features + 1:
            raise ValueError("For section or time_series Data, the col number of X should be n_features + 1")

        # Check for single-node programs
        node = self.program[0]
        # 常数
        if isinstance(node, (float, int)):
            return np.repeat(node, X.shape[0])
        # 变量
        if isinstance(node, str):
            return X[:, int(node)]

        apply_stack = []
        for node in self.program:

            if isinstance(node, _Function):
                apply_stack.append([node])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = [np.repeat(t, X.shape[0]) if isinstance(t, (float, int))
                             else (X[:, int(t)] if isinstance(t, str)
                                   else t) for t in apply_stack[-1][1:]]
                # 对于时序和截面函数加入管道
                if self.data_type == 'panel' and function.function_type == 'section':
                    time_series_data = X[:, -1]
                    intermediate_result = _groupby(time_series_data, function, *terminals)
                elif self.data_type == 'panel' and function.function_type == 'time_series':
                    security_data = X[:, -2]
                    intermediate_result = _groupby(security_data, function, *terminals)
                else:
                    intermediate_result = function(*terminals)
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result

        # We should never get here
        return None

    # 选择部分样本
    def get_all_indices(self, n_samples=None, max_samples=None,
                        random_state=None):
        """Get the indices on which to evaluate the fitness of a program.

        Parameters
        ----------
        n_samples : int
            The number of samples.

        max_samples : int
            The maximum number of samples to use.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        indices : array-like, shape = [n_samples]
            The in-sample indices.
            抽样内index

        not_indices : array-like, shape = [n_samples]
            The out-of-sample indices.
            抽样外index

        """
        if self._indices_state is None and random_state is None:
            raise ValueError('The program has not been evaluated for fitness '
                             'yet, indices not available.')

        if n_samples is not None and self._n_samples is None:
            self._n_samples = n_samples
        if max_samples is not None and self._max_samples is None:
            self._max_samples = max_samples
        if random_state is not None and self._indices_state is None:
            self._indices_state = random_state.get_state()

        indices_state = check_random_state(None)
        indices_state.set_state(self._indices_state)

        not_indices = sample_without_replacement(
            self._n_samples,
            self._n_samples - self._max_samples,
            random_state=indices_state)
        sample_counts = np.bincount(not_indices, minlength=self._n_samples)
        indices = np.where(sample_counts == 0)[0]

        return indices, not_indices

    # 获取衡量模型适应度的指标
    def _indices(self):
        """Get the indices used to measure the program's fitness."""
        return self.get_all_indices()[0]

    # 原始适应度
    def raw_fitness(self, X, y, sample_weight):
        """Evaluate the raw fitness of the program according to X, y.

        Parameters
        ----------
        X : {array-like}
            若数据类型为'section'，'time_series'则为[n_samples, n_features + 1]
            若数据类型为'panel', 则为[n_samples, n_features + 3]

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples]
            Weights applied to individual samples.

        Returns
        -------
        raw_fitness : float
            The raw fitness of the program.

        """
        if X.shape[0] != len(y):
            raise ValueError("The length of y should be equal to X")
        y_pred = self.execute(X)
        if self.transformer:
            y_pred = self.transformer(y_pred)
        raw_fitness = self.metric(y, y_pred, sample_weight)

        return raw_fitness

    # todo 引入非线性适应度
    # 惩罚后适应度 对函数长度进行惩罚
    def fitness(self, parsimony_coefficient=None):
        """Evaluate the penalized fitness of the program according to X, y.

        Parameters
        ----------
        parsimony_coefficient : float, optional
            If automatic parsimony is being used, the computed value according
            to the population. Otherwise the initialized value is used.

        Returns
        -------
        fitness : float
            The penalized fitness of the program.

        """
        if parsimony_coefficient is None:
            parsimony_coefficient = self.parsimony_coefficient
        penalty = parsimony_coefficient * len(self.program) * self.metric.sign
        return self.raw_fitness_ - penalty

    # 此函数为获得指定子树
    def get_subtree(self, start, program=None):
        """

        Parameters
        ----------
        start: 子树的根节点位置
        program
        Returns
        -------
        start
        end 子树截止位置 + 1 便于索引
        """
        if program is None:
            program = self.program
        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if isinstance(node, _Function):
                stack += node.arity
            end += 1

        # if isinstance(program[start], _Function):
        #     return_type = _Function.return_type
        if isinstance(program[start], _Function):
            return_type = program[start].return_type

        elif isinstance(program[start], str):
            if int(program[start]) == 0:
                raise ValueError("The return of sub_tree's root should not be const_1")
            return_type = 'category' if int(program[start]) <= self.n_cat_features else 'number'
        else:
            raise ValueError("The return type of sub_tree's root should be number or category")
        return start, end, return_type

    # 此函数为获得随机子树
    # 此处做了修改，不会选到标量
    # 需要考虑返回类型
    def get_random_subtree(self, random_state, program=None, return_type=None):
        """Get a random subtree from the program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        program : list, optional (default=None)
            The flattened tree representation of the program. If None, the
            embedded tree in the object will be used.

        return_type: 子数的返回类型限定 默认 None, number 和 category都可以选择

        Returns
        -------
        start, end : tuple of two ints
            The indices of the start and end of the random subtree.
        return_type: 子数返回类型，数值向量 还是 分类向量， 防止交叉时出现错误
        """
        if program is None:
            program = self.program
        # Choice of crossover points follows Koza's (1992) widely used approach
        # 子数节点概率权重90%，向量叶子节点概率权重10%，标量叶包括常分类向量子节点概率权重0
        # 若type为number， 所有返回category的节点概率权重为0
        # 若type为category， 所有返回number的节点概率权重为0
        if return_type not in ['number', 'category', None]:
            raise ValueError("Type of sub_tree should be number, category or None")
        if return_type == 'number':
            probs = np.array([0.9 if isinstance(node, _Function) and node.return_type == 'number'
                              else (0.1 if isinstance(node, str) and int(node) > self.n_cat_features else 0.0)
                              for node in program])
        elif return_type == 'category':
            probs = np.array([0.9 if isinstance(node, _Function) and node.return_type == 'category'
                              else (0.1 if isinstance(node, str) and int(node) <= self.n_cat_features
                                           and int(node) != 0 else 0.0)
                              for node in program])
        else:
            probs = np.array([0.9 if isinstance(node, _Function)
                              else (0.1 if isinstance(node, str)
                                           and int(node) != 0 else 0.0)
                              for node in program])
        probs = np.cumsum(probs / probs.sum())
        start = np.searchsorted(probs, random_state.uniform())
        return self.get_subtree(start, program)

    def reproduce(self):
        """Return a copy of the embedded program."""
        return copy(self.program)

    def vaild_category(self, program=None):
        """验证公式树中是否包含分类向量或子树， 不包括常数分类向量"""
        if program is None:
            program = self.program
        for node in program:
            if isinstance(node, _Function) and node.return_type == 'category':
                return True
            elif isinstance(node, str) and int(node) != 0 and int(node) <= self.n_cat_features:
                return True
        return False

    # 交换self 和 donor 的子树
    # 此处不会交换常数
    def crossover(self, donor, random_state):
        """Perform the crossover genetic operation on the program.

        Crossover selects a random subtree from the embedded program to be
        replaced. A donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring.

        Parameters
        ----------
        donor : list
            The flattened tree representation of the donor program.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        # 若都包含
        if self.vaild_category() and self.vaild_category(donor):
            start, end, self_return_type = self.get_random_subtree(random_state)
        else:
            start, end, self_return_type = self.get_random_subtree(random_state, return_type='number')
        removed = range(start, end)
        # Get a subtree to donate
        donor_start, donor_end, donor_return_type = self.get_random_subtree(random_state, donor, self_return_type)
        donor_removed = list(set(range(len(donor))) -
                             set(range(donor_start, donor_end)))
        # Insert genetic material from donor
        return (self.program[:start] +
                donor[donor_start:donor_end] +
                self.program[end:]), removed, donor_removed

    # 此处不会选择常数
    # 子数变异
    def subtree_mutation(self, random_state):
        """Perform the subtree mutation operation on the program.

        Subtree mutation selects a random subtree from the embedded program to
        be replaced. A donor subtree is generated at random and this is
        inserted into the original parent to form an offspring. This
        implementation uses the "headless chicken" method where the donor
        subtree is grown using the initialization methods and a subtree of it
        is selected to be donated to the parent.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Build a new naive program
        chicken = self.build_program(random_state, first_build=False)
        # Do subtree mutation via the headless chicken method!
        return self.crossover(chicken, random_state)

    def get_hoist_list(self, program=None):
        """
        判断公式树哪些节点可以做hoist变异, 该节点非叶子节点 且 存在与自身同类型的子树， 常分类向量不算分类向量的同类型
        Parameters
        ----------
        program

        Returns
        -------
        hoist_list
        """
        if program is None:
            program = self.program

        apply_stack = []
        hoist_list = [False] * len(program)

        for i, node in enumerate(program):
            logging.debug(f"Processing node {i}: {node}")
            if isinstance(node, _Function):
                apply_stack.append([i, node])
                logging.debug(f"Pushed to apply_stack: {apply_stack[-1]}")
            else:
                if not apply_stack:
                    # logging.warning(f"Skipping terminal node at index {i}: {node} (No parent function)")
                    continue  # 跳过没有父函数的终端节点
                apply_stack[-1].append(node)
                logging.debug(f"Appended node to apply_stack[-1]: {apply_stack[-1]}")

            # 处理堆栈中的函数节点
            while len(apply_stack[-1]) == apply_stack[-1][1].arity + 2:
                father_type = apply_stack[-1][1].return_type
                logging.debug(f"Processing function at index {apply_stack[-1][0]} with return type {father_type}")
                type_list = [
                    t if isinstance(t, list) else
                    (['number'] if isinstance(t, str) and int(t) > self.n_cat_features else
                     (['category'] if isinstance(t, str) and int(t) <= self.n_cat_features and int(t) != 0
                      else []))
                    for t in apply_stack[-1][2:]
                ]
                if father_type in list(set().union(*type_list)):
                    hoist_list[apply_stack[-1][0]] = True
                    logging.debug(f"Marked hoist_list[{apply_stack[-1][0]}] as True")
                type_list.append([father_type])

                intermediate_result = list(set().union(*type_list))
                if len(apply_stack) > 1:
                    popped = apply_stack.pop()
                    logging.debug(f"Popped from apply_stack: {popped}")
                    apply_stack[-1].append(intermediate_result)
                    logging.debug(f"Appended intermediate_result to apply_stack[-1]: {apply_stack[-1]}")
                else:
                    logging.debug("Reached root node, finishing traversal")
                    break  # 继续遍历剩余节点

        return hoist_list

    def hoist_mutation(self, random_state):
        """Perform the hoist mutation operation on the program.

        Hoist mutation selects a random subtree from the embedded program to
        be replaced. A random subtree of that subtree is then selected and this
        is 'hoisted' into the original subtrees location to form an offspring.
        This method helps to control bloat.

        gplearnplus修改，由于引入了变量类型，需要先考哪些节点可以hosit变异的节点
        要求
        1. 该节点下存在于节点同类型的子树

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.
        removed : list
            List of indices or nodes that were removed during mutation.

        """
        # 获取可以进行 hoist 变异的节点列表
        hoist_list = self.get_hoist_list()

        if sum(hoist_list) == 0:
            # 如果没有可进行 hoist 变异的节点，返回原始 program 和空的 removed 列表
            return self.program, []

        # 随机选取一个可以进行 hoist 变异的节点
        hoist_root = random_state.choice(np.where(hoist_list)[0])

        # 获取选定节点对应的子树
        start, end, return_type = self.get_subtree(hoist_root)
        subtree = self.program[start:end]

        # 从选定的子树中随机选取一个子子树进行 hoist
        sub_start, sub_end, _ = self.get_random_subtree(random_state, subtree, return_type=return_type)
        hoist = subtree[sub_start:sub_end]

        # 确定被移除的节点（用于绘图或跟踪）
        removed = list(set(range(start, end)) - set(range(start + sub_start, start + sub_end)))

        # 执行 hoist 变异，将选定子树替换为 hoisted 子子树
        mutated_program = self.program[:start] + hoist + self.program[end:]

        return mutated_program, removed

    # 点变异完全修改
    # 要求函数满足is_point_mutation条件
    # 由于无法得知范围，常数不变异
    def point_mutation(self, random_state):
        """Perform the point mutation operation on the program.

        Point mutation selects random nodes from the embedded program to be
        replaced. Terminals are replaced by other terminals and functions are
        replaced by other functions that require the same number of arguments
        as the original node. The resulting tree forms an offspring.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        program = copy(self.program)

        # Get the nodes to modify
        mutate = np.where(random_state.uniform(size=len(program)) <
                          self.p_point_replace)[0]
        tag = np.array([True] * len(mutate))
        for i, node in enumerate(mutate):
            if isinstance(program[node], _Function):
                arity = program[node].arity
                # Find a valid replacement with same arity
                replacement_list = [func_ for func_ in self.arities[arity] if program[node].is_point_mutation(func_)]
                if len(replacement_list) == 0:
                    # 没有满足条件的变异
                    tag[i] = False
                    continue
                replacement = random_state.randint(len(replacement_list))
                replacement = replacement_list[replacement]
                program[node] = replacement
            elif isinstance(program[node], str):
                # We've got a terminal, add a const or variable
                terminal = random_state.randint(1, self.n_features + 1)
                program[node] = str(terminal)
            else:
                # 常数不发生变异
                tag[i] = False
        if len(mutate):
            mutate = mutate[tag]
        return program, list(mutate)

    depth_ = property(_depth)
    length_ = property(_length)
    indices_ = property(_indices)
