import pandas as pd
import copy
import syntheticChrissAlmgren2 as sca
import utils
import numpy as np
import yfinance as yf
import time
import sys
import matplotlib.pyplot as plt
import torch
import math
from datetime import date, timedelta
from ddpg_agent import Agent
from collections import deque


class AC_RL_model():
    def __init__(self, ep, train_coverage, ticker, TOTAL_SHARES, NUM_TRADES, LIQUIDATION_TIME, start_date, end_date, ANNUAL_VOLAT,
                 BID_ASK_SP, llambda_list):

        self.TOTAL_SHARES = TOTAL_SHARES
        self.NUM_TRADES = NUM_TRADES
        self.LIQUIDATION_TIME = LIQUIDATION_TIME
        self.start_date = start_date
        self.end_date = end_date
        self.ticker = ticker
        self.num_ticker = len(ticker)
        self.data_origin = yf.download(ticker[0], start=start_date, end=end_date)
        self.TRAD_DAYS = len(self.data_origin['Open'])
        self.today = None
        self.date_list = []
        self.episode = 0

        self.subtract_days = timedelta(NUM_TRADES)
        self.slidingwindow_num = self.TRAD_DAYS - NUM_TRADES + 1
        self.randomseed = 0
        self.BID_ASK_SP = BID_ASK_SP
        self.tau = 1
        self.DAILY_VOLAT = ANNUAL_VOLAT / np.sqrt(self.TRAD_DAYS)
        self.llambda_list = llambda_list
        self.ep = ep
        self.num_train = train_coverage
        self.agent = None
        self.save_model_path = f'C:/Users/14ZD/Desktop/2022_Summer_LabIntern/Almgren and Chriss Model/models/{str(self.ep)+"epoches_"+str(self.num_train)+"daysTrain_"+str(self.TOTAL_SHARES)+"shares_"+str(self.LIQUIDATION_TIME)+"days"}.pt'


    def extract_dataFrame(self, stock):
        data_total = yf.download(stock, start=start_date - timedelta(self.num_train*2), end=end_date)
        data_train = data_total.tail(self.num_train+self.TRAD_DAYS).head(self.num_train)
        data_test = data_total.tail(self.num_train+self.TRAD_DAYS)
        data_trade = data_total.tail(self.TRAD_DAYS)
        
        return data_total, data_train, data_trade, data_test

    def run_RL(self, stock_num, data_total, data_train, iteration):

        end = -abs(self.num_train)
        self.agent = Agent(state_size=8,
                               action_size=1,
                               random_seed=0)
        for day in range(self.num_train):
            # 데이터 사용은 그 전날까지, 즉 logReturn 의 k번째도 전날이여야함, k-1 = 전전날
            # Parameters Calibration is determined by past data with length of train coverage
            data_calibration = data_total.iloc[day:self.num_train + day]

            # Financial Parameters Calibration Using Past Stock Data
            average_daily_volume = np.mean(data_calibration['Volume'])
            average_daily_spread = np.mean(data_calibration['High'] - data_calibration['Low'])
            epsilon = average_daily_spread / 2
            eta = average_daily_spread / (0.01 * average_daily_volume)
            gamma = average_daily_spread / (0.1 * average_daily_volume)
            llambda = self.llambda_list[0]

            startingPrice = data_train.head(1 + day)['Close'][0]
            self.today = data_train.index[day]
            singleStepVariance = (self.DAILY_VOLAT * startingPrice) ** 2


            # Create RL Environment
            rl_env = sca.MarketEnvironment()
            rl_env.__init__(0, self.LIQUIDATION_TIME, self.NUM_TRADES,
                            llambda, startingPrice, epsilon,
                            eta, gamma,
                            self.TOTAL_SHARES, singleStepVariance)

            # self.agent = Agent(state_size=rl_env.observation_space_dimension(),
            #                    action_size=rl_env.action_space_dimension(),
            #                    random_seed=0)

            rl_shortfall_hist = np.array([])
            rl_shortfall_deque = deque(maxlen=100)

            # Reset RL Environment in every episode
            assert self.LIQUIDATION_TIME >= 6

            for itr in range(int(iteration)):

                rl_env.__init__(0, self.LIQUIDATION_TIME, self.NUM_TRADES,
                                llambda, startingPrice, epsilon,
                                eta, gamma,
                                self.TOTAL_SHARES, singleStepVariance)
                cnt_log = 0
                data_log_return = data_total.iloc[end - self.TRAD_DAYS - 7 + cnt_log:end - self.TRAD_DAYS + cnt_log]
                logReturn_list = data_log_return['Close']
                logReturn = list(np.log((logReturn_list / logReturn_list.shift(1))).tail(6))

                rl_trl = rl_env.get_trade_list()
                rl_trade_list = utils.round_trade_list(rl_trl)
                cur_state = np.array(logReturn + [rl_env.timeHorizon / rl_env.num_n, rl_env.shares_remaining / rl_env.total_shares])

                rl_env.start_transactions()
                rl_action_list = []

                for trade in rl_trade_list:
                    # RL Total Capture Learning
                    rl_action = self.agent.act(cur_state, add_noise=True)
                    rl_action_list.append(rl_action[0])

                    new_state, rl_reward, done, rl_info = rl_env.step(rl_action)
                    # print("Reward:", rl_reward)

                    # env.step 할때마다 state가 바뀌므로 logReturn에 대한 업데이트 필요
                    cnt_log += 1
                    data_log_return = data_total.iloc[end - self.TRAD_DAYS - 7 + cnt_log:end - self.TRAD_DAYS + cnt_log]
                    logReturn_list = data_log_return['Close']
                    logReturn = list(np.log((logReturn_list / logReturn_list.shift(1))).tail(6))

                    new_state = np.array(logReturn + [rl_env.timeHorizon / rl_env.num_n,
                                                      rl_env.shares_remaining / rl_env.total_shares])

                    self.agent.step(cur_state, rl_action, rl_reward, new_state, done)

                    cur_state = new_state

                    if rl_info.done:
                        #print(rl_info.implementation_shortfall)
                        rl_shortfall_hist = np.append(rl_shortfall_hist, int(rl_info.implementation_shortfall))
                        rl_shortfall_deque.append(int(rl_info.implementation_shortfall))
                        break

                if (self.episode + 1) % 100 == 0:
                    print('\r{},{}, Episode [{}/{}]\Total Shortfall: ${:,.2f}'.format(self.ticker[stock_num], self.today, self.episode + 1,
                                                                                      self.ep,
                                                                                      np.mean(rl_shortfall_deque)))

                self.episode += 1
            
            
            #sys.exit()
            end += 1

                # Append Extracted Total Capture of a specified liquidation start date
                # rl_shortfall_list = np.append(rl_shortfall_list, np.mean(rl_shortfall_deque))

    def twap(self,data_trade, idx):
        starting_price = data_trade.iloc[idx]['Close']
        #print(data_trade.iloc[idx:self.LIQUIDATION_TIME+idx])
        high_list = data_trade.iloc[idx:self.LIQUIDATION_TIME+idx]['High']
        #print("high:", high_list)
        low_list = data_trade.iloc[idx:self.LIQUIDATION_TIME+idx]['Low']
        #print("low:", low_list)
        mid_price_list = [int((high+low)/2) for high,low in zip(high_list,low_list)]
        trade_shares = self.TOTAL_SHARES // self.LIQUIDATION_TIME
        remaining_shares = self.TOTAL_SHARES % self.LIQUIDATION_TIME
        trade_list = []
        for num in range(self.LIQUIDATION_TIME):
            if num ==0:
                trade_list.append(trade_shares+remaining_shares)
            else:
                trade_list.append(trade_shares)
        capture = [price*share for price, share in zip(mid_price_list, trade_list)]
        shortfall = self.TOTAL_SHARES*starting_price - sum(capture)
        #print("Starting Price from TWAP:",starting_price)
        #print("TWAP Trade List:", trade_list)
        #print("TWAP Price List:", mid_price_list)
        #print("TWAP Capture:", sum(capture))

        assert sum(trade_list) == self.TOTAL_SHARES

        return int(shortfall)



    def ac_optimal(self,data_calibration, data_trade, idx):

            # Financial Parameters Calibration Using Past Stock Data
            average_daily_volume = np.mean(data_calibration['Volume'])
            average_daily_spread = np.mean(data_calibration['High'] - data_calibration['Low'])
            epsilon = average_daily_spread / 2
            eta = average_daily_spread / (0.01 * average_daily_volume)
            gamma = average_daily_spread / (0.1 * average_daily_volume)
            llambda = self.llambda_list[0]

            startingPrice = data_trade.head(1 + idx)['Close'][-1]
            self.today = data_trade.index[idx]
            self.date_list.append(self.today)
            singleStepVariance = (self.DAILY_VOLAT * startingPrice) ** 2

            # AC Environment Initialized with new financial parameters
            env = sca.MarketEnvironment()
            env.__init__(0, self.LIQUIDATION_TIME, self.NUM_TRADES,
                            llambda, startingPrice, epsilon,
                            eta, gamma,
                            self.TOTAL_SHARES, singleStepVariance)

            trl = env.get_trade_list()
            trade_list = utils.round_trade_list(trl)
            #print("AC Trade List:", trade_list)
            env.start_transactions()
            price_hist = np.array([])
            action_list = []
            int_list = []
            for trade in trade_list:
                int_list.append(int(trade))

            #print(int_list)
            assert sum(int_list) == self.TOTAL_SHARES

            for trade in int_list:
                action = trade / env.shares_remaining
                #print(env.shares_remaining)
                _, _, _, info = env.step(action)
                action_list.append(action)
                price_hist = np.append(price_hist, info.exec_price)
                if info.done:
                    shortfall = info.implementation_shortfall
                    
                # If all shares have been sold, stop making transactions and get the implementation shortfall
                if info.done:
                    # print('Implementation Shortfall: ${:,.2f} \n'.format(info.implementation_shortfall))
                    break
            
            #print("AC Trade List:", action_list)
            return int(shortfall)      


    def rl_optimal(self, data_calibration, data_trade, data_test, idx):
        
        # Financial Parameters Calibration Using Past Stock Data
        average_daily_volume = np.mean(data_calibration['Volume'])
        average_daily_spread = np.mean(data_calibration['High'] - data_calibration['Low'])
        epsilon = average_daily_spread / 2
        eta = average_daily_spread / (0.01 * average_daily_volume)
        gamma = average_daily_spread / (0.1 * average_daily_volume)
        llambda = self.llambda_list[0]
        #print(data_trade.head(1+idx))
        startingPrice = data_trade.head(1 + idx)['Close'][-1]
        singleStepVariance = (self.DAILY_VOLAT * startingPrice) ** 2

        rl_env = sca.MarketEnvironment()
        rl_env.__init__(0, self.LIQUIDATION_TIME, self.NUM_TRADES,
                        llambda, startingPrice, epsilon, 
                        eta, gamma, 
                        self.TOTAL_SHARES, singleStepVariance)

        rl_trl = rl_env.get_trade_list()
        rl_trade_list = utils.round_trade_list(rl_trl)

        model = torch.load(self.save_model_path)

        agent = Agent(state_size=rl_env.observation_space_dimension(), action_size=rl_env.action_space_dimension(), 
                        random_seed=0)
        agent.actor_local.load_state_dict(model)

        cnt = 0 
        data_log_return = data_test.iloc[-self.TRAD_DAYS-7+idx+cnt:-self.TRAD_DAYS+idx+cnt]
        logReturn_list = data_log_return['Close']
        logReturn = list(np.log((logReturn_list / logReturn_list.shift(1))).tail(6))
        cur_state = np.array(logReturn + [rl_env.timeHorizon / rl_env.num_n, rl_env.shares_remaining / rl_env.total_shares])


        rl_env.start_transactions()
        rl_action_list = []

        for trade in rl_trade_list:

            rl_action = agent.act(cur_state, add_noise = True)
            #print(rl_env.shares_remaining)
            new_state, rl_reward, done, rl_info = rl_env.step(rl_action)
            rl_action_list.append(rl_action[0])

            # env.step 할때마다 state가 바뀌므로 logReturn에 대한 업데이트 필요
            cnt += 1
            data_log_return = data_test.iloc[-self.TRAD_DAYS-7+idx+cnt:-self.TRAD_DAYS+idx+cnt]
            logReturn_list = data_log_return['Close']
            logReturn = list(np.log((logReturn_list / logReturn_list.shift(1))).tail(6))
            new_state = np.array(logReturn + [rl_env.timeHorizon / rl_env.num_n, rl_env.shares_remaining / rl_env.total_shares])
            cur_state = new_state

            if rl_info.done:
                rl_shortfall = rl_info.implementation_shortfall
                break
        #print("RL Trade List:", rl_action_list)
        
        return int(rl_shortfall)
    
    def train(self):
            iteration = self.ep/self.num_ticker//self.num_train
            self.episode = 0

            for stock_num in range(self.num_ticker):
                data_total, data_train, data_trade, data_test = self.extract_dataFrame(self.ticker[stock_num])
                self.run_RL(stock_num, data_total, data_train, iteration)

            torch.save(self.agent.actor_local.state_dict(), self.save_model_path)
    
    def test(self):
        for code in self.ticker:
                
            data_total, data_train, data_trade, data_test = self.extract_dataFrame(code)
            #print(code, data_trade)
            ac_shortfall_list = []
            twap_shortfall_list = []
            rl_shortfall_list = []
            self.date_list = []
            for idx in range(self.slidingwindow_num):
                #AIS = averaged implementation shortfall
                data_calibration = data_test.iloc[idx:self.num_train + idx]
                ac_shortfall = self.ac_optimal(data_calibration, data_trade, idx)
                ac_shortfall_list = np.append(ac_shortfall_list, round(ac_shortfall/self.TOTAL_SHARES,4))
                twap_shortfall = self.twap(data_trade, idx)
                twap_shortfall_list = np.append(twap_shortfall_list, round(twap_shortfall/self.TOTAL_SHARES,4))
                rl_shortfall = self.rl_optimal(data_calibration, data_trade, data_test, idx)
                rl_shortfall_list = np.append(rl_shortfall_list, round(rl_shortfall/self.TOTAL_SHARES,4))

                # print()
                # print(twap_shortfall_list)
                # print(ac_shortfall_list)
                # print(rl_shortfall_list)
                # print()
                # print("ㅡ"*50)
                # if idx ==2:    
                #     sys.exit()

            assert len(ac_shortfall_list) == len(twap_shortfall_list) and len(twap_shortfall_list) == len(rl_shortfall_list) and len(rl_shortfall_list) == len(self.date_list)

            # print(len(ac_shortfall_list))
            # print(len(twap_shortfall_list))
            # print(len(rl_shortfall_list))
            # print(len(self.date_list))

            result = [ac_shortfall_list > rl_shortfall_list]
            df = pd.DataFrame({'TWAP': twap_shortfall_list, 'AC':ac_shortfall_list, 'RL':rl_shortfall_list}, index=self.date_list)
            
            if code == self.ticker[-1]:
                print()
                print('='*70)
                print(code)
                print(df)
                print('='*70)
                print()


            # print(result)
            # sys.exit()






TOTAL_SHARES = 1000000
NUM_TRADES = 6
LIQUIDATION_TIME = 6
start_date = date(2022, 1, 3)
end_date = '2022-02-01'
ticker = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

#Hyperparameter
ANNUAL_VOLAT = 0.12
BID_ASK_SP = 1 / 8
llambda_list = [1e-06]

train_coverage = 252
eps = [2000,4000,6000]

for ep in eps:
    simulation = AC_RL_model(ep, train_coverage, ticker, TOTAL_SHARES, NUM_TRADES, LIQUIDATION_TIME, start_date, end_date, ANNUAL_VOLAT,
                                BID_ASK_SP, llambda_list)
    plt.rcParams['figure.figsize'] = [16.0, 9.0]                            
    simulation.train()
    simulation.test()

# simulation = AC_RL_model(ep, train_coverage, ticker, 10000, 10, 10, start_date, end_date, ANNUAL_VOLAT,
#                             BID_ASK_SP, llambda_list)
# plt.rcParams['figure.figsize'] = [16.0, 9.0]                            
# #simulation.train()
# simulation.test()



