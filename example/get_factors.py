import datetime
import time
import json
import pandas as pd
import numpy as np
import requests
import sys
import os
sys.path.append(os.getcwd())
from QuantGplearn.genetic import SymbolicTransformer
from QuantGplearn import fitness
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from QuantGplearn import functions
import dill
import matplotlib.pyplot as plt
import seaborn as sns
from utils.backtest_tool.generate_performance import get_performance, get_pnl

function_set = functions.all_function

windows_1d = 24
windows_3d = 24 * 3
windows_7d = 24 * 7
windows_14d = 24 * 14
windows_21d = 24 * 21
windows_30d = 24 * 30


def get_kline(symbol, end_datetime):
    try:
        df_symbol_all = pd.read_hdf(f"data/{symbol}_1h_{end_datetime}.h5", key="data")
    except FileNotFoundError as e:
        base_url = "https://fapi.binance.com"
        end_ts = int(end_datetime.timestamp() * 1000)
        data_list = []
        t0 = datetime.datetime.now()
        print(f"{t0} start download data...")
        while True:
            res = requests.get(f"{base_url}/fapi/v1/continuousKlines",
                               params={"pair": symbol,
                                       "contractType": "PERPETUAL",
                                       "interval": "1h",
                                       "endTime": end_ts,
                                       "limit": 1500
                                       },
                               timeout=10)
            time.sleep(0.1)
            data = pd.DataFrame(json.loads(res.text))
            if len(data) <= 0:
                break
            else:
                data = data.sort_values(by=0, ascending=True)
                data_list.append(data)
                end_ts = data[0][0] - 60 * 60 * 1000
        df_symbol_all = pd.concat(data_list, axis=0)
        df_symbol_all = df_symbol_all.sort_values(by=0, ascending=True)
        df_symbol_all = df_symbol_all.reset_index(drop=True)
        df_symbol_all.columns = ['open_time', 'open_price', 'high_price', 'low_price', 'close_price', 'volume', 'close_time',
                      'amount', 'count', 'buy_volume', 'buy_amount', 'ignore']
        df_symbol_all['date_time'] = pd.to_datetime(df_symbol_all['open_time'], unit="ms")
        df_symbol_all = df_symbol_all.set_index('date_time')
        df_symbol_all = df_symbol_all[['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'amount']]
        df_symbol_all = df_symbol_all.astype(float)
        df_symbol_all.to_hdf(f"data/{symbol}_1h_{end_datetime}.h5", key="data")
        t1 = datetime.datetime.now()
        print(f"{t1} finish download data, time escaped {t1 - t0}")
    return df_symbol_all

def my_gplearn(function_set, score_func_basic,
               population_size, hall_of_fame, n_components, generations, tournament_size, stopping_criteria, init_depth,
               p_crossover, p_subtree_mutation, p_hoist_mutation, p_point_mutation, p_point_replace,
               random_state=2025, feature_names=None, first_train=True, last_train_elites=None):

    metric = fitness.make_fitness(function=score_func_basic, # function(y, y_pred, sample_weight) that returns a floating point number.
                        greater_is_better=True,  # 上述y是输入的目标y向量，y_pred是genetic program中的预测值，sample_weight是样本权重向量
                        wrap=False)  # 不保存，运行的更快 # gplearn.fitness.make_fitness(function, greater_is_better, wrap=True)
    return SymbolicTransformer(population_size=population_size,  # 每一代公式群体中的公式数量 500，100
                              hall_of_fame=hall_of_fame,
                              n_components=n_components,
                              generations=generations,  # 公式进化的世代数量 10，3
                              metric=metric,  # 适应度指标，这里是前述定义的通过 大于0做多，小于0做空的 累积净值/最大回撤 的评判函数
                              tournament_size=tournament_size,  # 在每一代公式中选中tournament的规模，对适应度最高的公式进行变异或繁殖 50
                              function_set=function_set,
                              const_range=(-1, 1.0),  # 公式中包含的常数范围
                              parsimony_coefficient='unauto',
                              stopping_criteria=stopping_criteria,  # 是对metric的限制（此处为收益/回撤）
                              init_depth=init_depth,  # 公式树的初始化深度，树深度最小2层，最大6层
                              init_method='half and half',  # 树的形状，grow生分枝整的不对称，full长出浓密
                              p_crossover=p_crossover,  # 交叉变异概率 0.8
                              p_subtree_mutation=p_subtree_mutation,  # 子树变异概率
                              p_hoist_mutation=p_hoist_mutation,  # hoist变异概率 0.15
                              p_point_mutation=p_point_mutation,  # 点变异概率
                              p_point_replace=p_point_replace,  # 点变异中每个节点进行变异进化的概率
                              max_samples=1.0,  # The fraction of samples to draw from X to evaluate each program on.
                              feature_names=feature_names, warm_start=False, low_memory=False,
                              n_jobs=mp.cpu_count() - 2,
                              verbose=2,
                              tolerable_corr = 0.7,
                              first_train=first_train,
                              last_train_elites=last_train_elites,
                              random_state=random_state)

def gp_save_factor(my_cmodel_gp, factor_num=''):
    with open(f'example/details/factors/factor_{factor_num}.pickle', 'wb') as f:
        dill.dump(my_cmodel_gp, f)

def score_func_basic(y, y_pred, sample_weight, time_series, close_series, vwap_series, train=False):  #
    try:
        holding_arr, tradetimes = bt.get_holding_for_gp(time_series, close_series, y_pred)
        df_target_pos = pd.DataFrame(index=time_series)
        df_target_pos[bt.symbol] = holding_arr
        df_match_price = pd.DataFrame(index=time_series)
        df_match_price[bt.symbol] = vwap_series.shift(-1).values
        time_arr, nv_arr, nv_without_fee_arr, AnnualRateOfRtn, max_drawdown, calmar_ratio, sharpe_ratio = bt.get_performance_without_pic(
            df_target_pos.iloc[:-1, :], df_match_price.iloc[:-1, :])
        factor_ret = ((calmar_ratio + sharpe_ratio) / 2) if max_drawdown != 0 else 0
        if tradetimes <= 5:
            factor_ret = 0
    except Exception as e:
        factor_ret = 0
        print(e)
    return factor_ret

class backtest:
    def __init__(self, symbol, strategy_name, data_end_datetime, start_backtest_datetime, fee_ratio):
        self.symbol = symbol
        self.strategy_name = strategy_name
        self.data_end_datetime = data_end_datetime
        self.start_backtest_datetime = start_backtest_datetime
        self.fee_ratio = fee_ratio
        self.cash_init = 1000000.
        self.strategy_num = 2
        self.population_size = 10
        self.hall_of_fame = 10
        self.n_components = self.strategy_num
        self.generations = 2
        self.tournament_size = 10
        self.stopping_criteria = 5
        self.init_depth = (2, 5)
        self.p_crossover = 0.2
        self.p_subtree_mutation = 0.2
        self.p_hoist_mutation = 0.2
        self.p_point_mutation = 0.2
        self.p_point_replace = 0.2

    def get_performance_without_pic(self, df_target_pos, df_match_price):
        df_profit_without_fee_arr, df_profit_with_fee_arr = get_pnl(df_target_pos, df_match_price, self.fee_ratio)
        _ = get_performance(self.strategy_name, self.cash_init, df_profit_with_fee_arr,
                                        df_profit_without_fee_arr, save_df=False, save_pic=False)
        return _

    def get_performance(self, df_target_pos, df_match_price):
        df_profit_without_fee_arr, df_profit_with_fee_arr = get_pnl(df_target_pos, df_match_price, self.fee_ratio)
        _ = get_performance(self.strategy_name, self.cash_init, df_profit_with_fee_arr,
                                        df_profit_without_fee_arr, save_df=True, save_pic=True)
        return _

    def get_holding_for_gp(self, time_series, close_series, gp_factor_series):
        gp_factor_series_r = pd.Series(gp_factor_series).rolling(windows_7d)
        gp_factor_series_min = gp_factor_series_r.min()
        gp_factor_series_max = gp_factor_series_r.max()
        quantile_series = (gp_factor_series - gp_factor_series_min) / (gp_factor_series_max - gp_factor_series_min)
        quantile_arr = quantile_series.values
        quantile_trend_para = 0.1
        tradetimes = 0
        holding_arr = np.full((len(time_series),), 0.)
        close_arr = close_series

        time_series = pd.to_datetime(time_series)

        cash_init = self.cash_init

        for i in range(0, len(time_series)):
            holding_arr[i] = holding_arr[i - 1]
            if holding_arr[i] == 0:
                if quantile_arr[i] >= 1 - quantile_trend_para:
                    holding_arr[i] = cash_init / close_arr[i]
                elif quantile_arr[i] <= quantile_trend_para:
                    holding_arr[i] = - cash_init / close_arr[i]
            elif holding_arr[i] > 0:
                if quantile_arr[i] < 0.5:
                    if quantile_arr[i] <= quantile_trend_para:
                        holding_arr[i] = - cash_init / close_arr[i]
                        tradetimes += 1
                    else:
                        holding_arr[i] = 0
                        tradetimes += 1
            elif holding_arr[i] < 0:
                if quantile_arr[i] > 0.5:
                    if quantile_arr[i] >= 1 - quantile_trend_para:
                        holding_arr[i] = cash_init / close_arr[i]
                        tradetimes += 1
                    else:
                        holding_arr[i] = 0
                        tradetimes += 1
        return holding_arr, tradetimes

    def get_holding(self, all_data):
        # 获取因子
        time_series = pd.Series(all_data.index)
        feature_names = list(all_data.columns)
        self.start_index = np.where(time_series >= self.start_backtest_datetime)[0][0]

        train_windows = 24 * (365 * 3 + 300)
        train_data = all_data.iloc[self.start_index - train_windows + 1: self.start_index + 1]
        train_time_series = pd.Series(train_data.index)
        train_close_price = train_data['close']
        train_vwap_series = train_data['vwap']
        score_func_with_fixed_params = partial(
            score_func_basic,
            time_series=train_time_series,
            close_series=train_close_price,
            vwap_series=train_vwap_series,
            train=True
        )
        t0 = datetime.datetime.now()
        my_gp_model = my_gplearn(function_set,
                                 score_func_with_fixed_params,
                                 population_size=self.population_size,
                                 hall_of_fame=self.hall_of_fame,
                                 n_components=self.n_components,
                                 generations=self.generations,
                                 tournament_size=self.tournament_size,
                                 stopping_criteria=self.stopping_criteria,
                                 init_depth=self.init_depth,
                                 p_crossover=self.p_crossover,
                                 p_subtree_mutation=self.p_subtree_mutation,
                                 p_hoist_mutation=self.p_hoist_mutation,
                                 p_point_mutation=self.p_point_mutation,
                                 p_point_replace=self.p_point_replace,
                                 feature_names=feature_names,
                                 first_train=True)
        my_gp_model.fit(train_data.values, train_data.loc[:, 'close'].values, max_length=20)
        print(f'耗时:{datetime.datetime.now() - t0}')
        print(f"-----------------------{datetime.datetime.now()} 训练结束--------------------------------------")
        print(my_gp_model)
        t0 = datetime.datetime.now()
        gp_save_factor(my_gp_model, self.strategy_name)
        print(f"{datetime.datetime.now()} 保存模型耗时：{datetime.datetime.now() - t0}")
        train_factors = pd.DataFrame(my_gp_model.transform(train_data))
        correlation_matrix = train_factors.corr()
        plt.figure(figsize=(20, 15))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, square=True)
        plt.title('Correlation Matrix Heatmap')
        plt.savefig(f"example/details/factors/{self.strategy_name}.png")
        plt.close()

        # 计算目标仓位
        holding_arr = np.full((len(all_data), self.strategy_num), 0.)

        close_arr = all_data['close'].values

        quantile_trend_para = 0.1
        cash_init = self.cash_init
        tradetimes = 0

        factors = pd.DataFrame(my_gp_model.transform(all_data.iloc[windows_7d:, :]))
        factors_r = factors.rolling(windows_7d)
        factor_max = factors_r.max()
        factor_min = factors_r.min()
        factors_quantile = (factors - factor_min) / (factor_max - factor_min)
        factor_quantile = factors_quantile.values

        for i in range(self.start_index, len(time_series)):
            holding_arr[i, :] = holding_arr[i - 1, :]
            for j in range(self.strategy_num):
                if holding_arr[i, j] == 0:
                    if factor_quantile[i - windows_7d, j] >= 1 - quantile_trend_para:
                        holding_arr[i, j] = cash_init / close_arr[i]
                        # print(time_series[i], f"trader_{j}", "开多， 价格：%f" % close_arr[i])
                    elif factor_quantile[i - windows_7d, j] <= quantile_trend_para:
                        holding_arr[i, j] = -cash_init / close_arr[i]
                        # print(time_series[i], f"trader_{j}", "开空， 价格：%f" % close_arr[i])
                elif holding_arr[i, j] > 0:
                    if factor_quantile[i - windows_7d, j] < 0.5:
                        if factor_quantile[i - windows_7d, j] <= quantile_trend_para:
                            holding_arr[i, j] = - cash_init / close_arr[i]
                            # print(time_series[i], f"trader_{j}", "平多并开空， 价格：%f" % close_arr[i])
                            tradetimes += 1
                        else:
                            holding_arr[i, j] = 0
                            # print(time_series[i], f"trader_{j}", "平多， 价格：%f" % close_arr[i])
                            tradetimes += 1
                elif holding_arr[i, j] < 0:
                    if factor_quantile[i - windows_7d, j] > 0.5:
                        if factor_quantile[i - windows_7d, j] >= 1 - quantile_trend_para:
                            holding_arr[i, j] = cash_init / close_arr[i]
                            # print(time_series[i], f"trader_{j}", "平空并开多， 价格：%f" % close_arr[i])
                            tradetimes += 1
                        else:
                            holding_arr[i, j] = 0
                            # print(time_series[i], f"trader_{j}", "平空， 价格：%f" % close_arr[i])
                            tradetimes += 1
        try:
            print(f"日均交易次数:{tradetimes / self.strategy_num / (len(time_series[self.start_index:]) / (4 * 24))}")
        except Exception as e:
            print(e)
        return holding_arr

    def run(self):
        df_kline = get_kline(self.symbol, self.data_end_datetime)
        # print(df_kline)
        df_kline_1H = df_kline.resample("1H")
        amt_series = df_kline_1H['amount'].sum().fillna(0)
        qty_series = df_kline_1H['volume'].sum().fillna(0)
        time_series = pd.Series(amt_series.index)
        open_series = df_kline_1H['open_price'].last().ffill()
        close_series = df_kline_1H['close_price'].last().ffill()
        high_series = df_kline_1H['high_price'].last().ffill()
        low_series = df_kline_1H['low_price'].last().ffill()
        vwap_series = (amt_series / qty_series).replace([np.inf, -np.inf], np.nan).ffill()
        all_data = pd.concat([open_series, high_series, low_series, close_series, vwap_series], axis=1)
        all_data.columns = ["open", "high", "low", "close", "vwap"]
        all_data['mom_1d'] = all_data['vwap'] / all_data['vwap'].shift(windows_1d) - 1
        all_data['mom_3d'] = all_data['vwap'] / all_data['vwap'].shift(windows_3d) - 1
        all_data['mom_7d'] = all_data['vwap'] / all_data['vwap'].shift(windows_7d) - 1
        all_data['ma_1d'] = all_data['vwap'].rolling(windows_1d).mean()
        all_data['ma_3d'] = all_data['vwap'].rolling(windows_3d).mean()
        all_data['ma_7d'] = all_data['vwap'].rolling(windows_7d).mean()
        all_data['zscore_1d'] = (all_data['vwap'] - all_data['vwap'].rolling(windows_1d).mean()) / all_data[
            'vwap'].rolling(windows_1d).std(ddof=1)
        all_data['zscore_3d'] = (all_data['vwap'] - all_data['vwap'].rolling(windows_3d).mean()) / all_data[
            'vwap'].rolling(windows_3d).std(ddof=1)
        all_data['zscore_7d'] = (all_data['vwap'] - all_data['vwap'].rolling(windows_7d).mean()) / all_data[
            'vwap'].rolling(windows_7d).std(ddof=1)
        holding_arr = self.get_holding(all_data)
        df_target_pos = pd.DataFrame(holding_arr[self.start_index:], index=time_series[self.start_index:],
                                     columns=list(range(self.strategy_num)))
        df_match_price = pd.DataFrame(index=time_series[self.start_index:], columns=list(range(self.strategy_num)))
        match_price_arr = vwap_series.shift(-1).values[self.start_index:]
        for j in range(self.strategy_num):
            df_match_price[j] = match_price_arr
        self.get_performance(df_target_pos.iloc[:-1, :], df_match_price.iloc[:-1, :])


if __name__ == '__main__':
    symbol_list = ["BTCUSDT"]
    for symbol in symbol_list:
        bt = backtest(symbol = symbol,
                      strategy_name=f"gp_select_factors_{symbol.lower()}_quantile_7d",
                      data_end_datetime = datetime.datetime(2024, 12, 31),
                      start_backtest_datetime=datetime.datetime(2024, 1, 1, 1, 0, 0),
                      fee_ratio=3 / 10000)
        bt.run()


