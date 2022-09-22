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
import os
import csv
from tqdm import tqdm
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
        self.cali_date = 0
        self.actual_train = 0
        self.convergence_list = []

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
        self.loss_list = []
        self.loss_df = None
        self.save_model_path = f'C:/Users/14ZD/Desktop/2022_Summer_LabIntern/Almgren and Chriss Model/models/{str(self.ep)+"epoches_"+str(self.num_train)+"daysTrain_"+str(self.TOTAL_SHARES)+"shares_"+str(self.LIQUIDATION_TIME)+"days"}.pt'


    def extract_dataFrame(self, stock):
        total_start_date = start_date - timedelta(self.num_train)
       
        data_total = yf.download(stock, start=total_start_date, end=end_date)
        calibration_days = len(data_total) //10
        self.cali_date = calibration_days
        data_train = data_total.tail(-self.cali_date).head(-self.TRAD_DAYS)
        self.actual_train = len(data_train)
        data_test = data_total.tail(-self.cali_date)
        data_trade = data_total.tail(self.TRAD_DAYS)

        # print(data_total)
        # print()
        # print(calibration_days)
        # print(data_train)
        # print(data_test, data_trade)
        
        return data_total, data_train, data_trade, data_test

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

            data_liquidation = data_trade.head(1 + idx + self.LIQUIDATION_TIME)
            startingPriceData = data_liquidation.head(self.LIQUIDATION_TIME)
            startingPrice = startingPriceData['Close'][0]
            self.today = data_trade.index[idx]
            self.date_list.append(self.today)
            singleStepVariance = (self.DAILY_VOLAT * startingPrice) ** 2

            # AC Environment Initialized with new financial parameters
            env = sca.MarketEnvironment()
            env.__init__(0, self.LIQUIDATION_TIME, self.NUM_TRADES,
                            llambda, startingPrice, epsilon,
                            eta, gamma,
                            self.TOTAL_SHARES, singleStepVariance, startingPriceData)

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

        data_liquidation = data_trade.head(1 + idx + self.LIQUIDATION_TIME)
        startingPriceData = data_liquidation.head(self.LIQUIDATION_TIME)
        startingPrice = startingPriceData['Close'][0]
        singleStepVariance = (self.DAILY_VOLAT * startingPrice) ** 2

        rl_env = sca.MarketEnvironment()
        rl_env.__init__(0, self.LIQUIDATION_TIME, self.NUM_TRADES,
                        llambda, startingPrice, epsilon, 
                        eta, gamma, 
                        self.TOTAL_SHARES, singleStepVariance, startingPriceData)

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
    
    def run_RL(self, data_total, data_train):

        end = -abs(len(data_train))
        mean_reward_per_day = []
        IS_list = []
        for day in range(self.actual_train):
            # 데이터 사용은 그 전날까지, 즉 logReturn 의 k번째도 전날이여야함, k-1 = 전전날
            # Parameters Calibration is determined by past data with length of train coverage
            data_calibration = data_total.iloc[day:self.cali_date + day]

            # Financial Parameters Calibration Using Past Stock Data
            average_daily_volume = np.mean(data_calibration['Volume'])
            average_daily_spread = np.mean(data_calibration['High'] - data_calibration['Low'])
            epsilon = average_daily_spread / 2
            eta = average_daily_spread / (0.01 * average_daily_volume)
            gamma = average_daily_spread / (0.1 * average_daily_volume)
            llambda = self.llambda_list[0]

            data_liquidation = data_train.head(1 + day + self.LIQUIDATION_TIME)
            startingPriceData = data_liquidation.head(self.LIQUIDATION_TIME)
            startingPrice = startingPriceData['Close'][0]


            self.today = data_train.index[day]
            assert data_total.iloc[day:self.cali_date + day+1].index[-1] == self.today
            singleStepVariance = (self.DAILY_VOLAT * startingPrice) ** 2


            # Create RL Environment
            rl_env = sca.MarketEnvironment()
            rl_env.__init__(0, self.LIQUIDATION_TIME, self.NUM_TRADES,
                            llambda, startingPrice, epsilon,
                            eta, gamma,
                            self.TOTAL_SHARES, singleStepVariance, startingPriceData)

            rl_shortfall_hist = np.array([])
            rl_shortfall_deque = deque(maxlen=100)

            # Reset RL Environment in every episode
            assert self.LIQUIDATION_TIME >= 6
            cnt_log = 0
            data_log_return = data_total.iloc[end - self.TRAD_DAYS - 7 + cnt_log:end - self.TRAD_DAYS + cnt_log]

            assert data_total.iloc[end - self.TRAD_DAYS - 7 + cnt_log:end - self.TRAD_DAYS + cnt_log+1].index[-1] == self.today

            logReturn_list = data_log_return['Close']
            logReturn = list(np.log((logReturn_list / logReturn_list.shift(1))).tail(6))
            # print()
            # print(data_total)
            # print(data_train.head(1))
            # print(end - self.TRAD_DAYS - 7 + cnt_log)
            # print(end - self.TRAD_DAYS + cnt_log)
            # print(data_log_return)

            rl_trl = rl_env.get_trade_list()
            rl_trade_list = utils.round_trade_list(rl_trl)
            cur_state = np.array(logReturn + [rl_env.timeHorizon / rl_env.num_n, rl_env.shares_remaining / rl_env.total_shares])

            rl_env.start_transactions()
            rl_action_list = []
            reward_list = []

            for trade in range(len(rl_trade_list)):
                # RL Total Capture Learning
                rl_action = self.agent.act(cur_state, add_noise=True)
                rl_action_list.append(rl_action[0])

                #negative reward can be loss in RL
                new_state, rl_reward, done, rl_info = rl_env.step(rl_action)
                rl_env.price_data = rl_env.price_data.iloc[1:]
                reward_list.append(round(rl_reward[0],8))
                
                # env.step 할때마다 state가 바뀌므로 logReturn에 대한 업데이트 필요
                cnt_log += 1
                data_log_return = data_total.iloc[end - self.TRAD_DAYS - 7 + cnt_log:end - self.TRAD_DAYS + cnt_log]
                logReturn_list = data_log_return['Close']
                logReturn = list(np.log((logReturn_list / logReturn_list.shift(1))).tail(6))

                new_state = np.array(logReturn + [rl_env.timeHorizon / rl_env.num_n,
                                                    rl_env.shares_remaining / rl_env.total_shares])
                


                self.agent.step(cur_state, rl_action, rl_reward, new_state, done)

                cur_state = new_state

                #LIQUIDATION TIME 내에 청산이 끝나도 청산기간동안 reward를 계속 받아야될까?
                if rl_info.done:
                    rl_shortfall_hist = np.append(rl_shortfall_hist, int(rl_info.implementation_shortfall))
                    rl_shortfall_deque.append(int(rl_info.implementation_shortfall))
                    mean_reward_per_day.append(np.mean(reward_list))
                    IS_list.append(int(rl_info.implementation_shortfall))
                    # print(mean_reward_per_day)
                    # if list(rl_env.utility_list.keys())[0] == self.TOTAL_SHARES:
                    # print(list(rl_env.utility_list.keys())[0])
                    # print(reward_list)
                    
                    break



            # if (self.episode + 1) % 100 == 0:
            #     print('\r{},{}, Episode [{}/{}]\Total Shortfall: ${:,.2f}'.format(self.ticker[stock_num], self.today, self.episode + 1,
            #                                                                         self.ep,
            #                                                                         np.mean(rl_shortfall_deque)))

            self.episode += 1
            end += 1

        return -abs(np.mean(mean_reward_per_day)), np.mean(IS_list)
    
    def train(self):
        self.episode = 0
        self.agent = Agent(state_size=8,action_size=1,random_seed=0)
        stock = {}
        
        for idx in range(len(self.ticker)):
            data_total, data_train, data_trade, data_test = self.extract_dataFrame(self.ticker[idx])
            stock[str(idx)+"_data_total"] = data_total
            stock[str(idx)+"_data_train"] = data_train
            stock[str(idx)+"_data_trade"] = data_trade
            stock[str(idx)+"_data_test"] = data_test
        
        # print(stock["1_data_total"])
        # print(stock["1_data_train"])

        for episode in tqdm(range(self.ep)):
            loss_train = []
            convergence_sub_list = []
            for stock_num in range(len(self.ticker)):
                stock_loss, mean_IS = self.run_RL(stock[str(stock_num)+"_data_total"], stock[str(stock_num)+"_data_train"])
                loss_train.append(stock_loss)
                convergence_sub_list = np.append(convergence_sub_list, int(mean_IS))
                #print(loss_train)
            self.loss_list.append(np.mean(loss_train))
            self.convergence_list.append(convergence_sub_list)
        
        self.convergence_list = np.array(self.convergence_list)
        
        df_convergence = pd.DataFrame(self.convergence_list)
        df_convergence.columns = ['AAPL','MSFT','GOOGL','AMZN','TSLA']

        plt.plot(df_convergence)
        plt.xlabel('Epoches')
        plt.ylabel('Mean Implementation Shortfall')
        plt.legend(['AAPL','MSFT','GOOGL','AMZN','TSLA'])
        plt.savefig('IS_Convergence_{}epoches_{}trainDays_{}shares_{}liquidation'.format(self.ep, self.num_train, self.TOTAL_SHARES, self.LIQUIDATION_TIME))

        self.loss_df = pd.DataFrame({"Train":self.loss_list})
        plt.plot(self.loss_df)
        # plt.legend()
        plt.savefig('{}epoches_{}trainDays_{}shares_{}liquidation'.format(self.ep, self.num_train, self.TOTAL_SHARES, self.LIQUIDATION_TIME))
        torch.save(self.agent.actor_local.state_dict(), self.save_model_path)
    
    def test(self):
        csv_list = []
        csv_list.extend([self.num_train, self.ep])
        for code in self.ticker:
                
            data_total, data_train, data_trade, data_test = self.extract_dataFrame(code)
            #print(code, data_trade)
            ac_shortfall_list = []
            twap_shortfall_list = []
            rl_shortfall_list = []
            self.date_list = []
            for idx in range(self.slidingwindow_num):
                #AIS = averaged implementation shortfall
                cali_data = data_test.tail(self.cali_date+self.TRAD_DAYS)
                data_calibration = cali_data.iloc[idx:self.cali_date + idx]
                ac_shortfall = self.ac_optimal(data_calibration, data_trade, idx)
                ac_shortfall_list = np.append(ac_shortfall_list, round(ac_shortfall/self.TOTAL_SHARES,4))
                twap_shortfall = self.twap(data_trade, idx)
                twap_shortfall_list = np.append(twap_shortfall_list, round(twap_shortfall/self.TOTAL_SHARES,4))
                rl_shortfall = self.rl_optimal(data_calibration, data_trade, data_test, idx)
                rl_shortfall_list = np.append(rl_shortfall_list, round(rl_shortfall/self.TOTAL_SHARES,4))

            rl_twap_list = [a - b for a, b in zip(rl_shortfall_list,twap_shortfall_list)]
            mean_rl_twap = round(np.mean(rl_twap_list),2)
            std_rl_twap = round(np.std(rl_twap_list),2)
            win_rl_twap = [twap_shortfall_list > rl_shortfall_list]
            winrate_rl_twap = str(int(np.sum(win_rl_twap[0])/len(win_rl_twap[0]) * 100)) + "%"

            rl_ac_list = [c - d for c, d in zip(rl_shortfall_list,ac_shortfall_list)]
            mean_rl_ac = round(np.mean(rl_ac_list),2)
            std_rl_ac = round(np.std(rl_ac_list),2)
            win_rl_ac = [ac_shortfall_list > rl_shortfall_list]
            winrate_rl_ac = str(int(np.sum(win_rl_ac[0])/len(win_rl_ac[0]) * 100)) + "%"

            csv_list.extend([mean_rl_twap, std_rl_twap, winrate_rl_twap, mean_rl_ac, std_rl_ac, winrate_rl_ac])

                # print()
                # print(twap_shortfall_list)
                # print(ac_shortfall_list)
                # print(rl_shortfall_list)
                # print()
                # print("ㅡ"*50)
                # if idx ==2:    
                #     sys.exit()

            assert len(ac_shortfall_list) == len(twap_shortfall_list) and len(twap_shortfall_list) == len(rl_shortfall_list) and len(rl_shortfall_list) == len(self.date_list)

            result = [ac_shortfall_list > rl_shortfall_list]
            df = pd.DataFrame({'TWAP': twap_shortfall_list, 'AC':ac_shortfall_list, 'RL':rl_shortfall_list}, index=self.date_list)
            

            # print()
            # print('='*70)
            # print(code)
            # print(df)
            # print('='*70)
            # print()
        # print(csv_list)

        return csv_list



if __name__ == "__main__":
    TOTAL_SHARES = 500000
    LIQUIDATION_TIME = 6
    start_date = date(2022, 1, 3)
    end_date = '2022-02-01'

    NUM_TRADES = LIQUIDATION_TIME
    ticker = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

    #Hyperparameter
    ANNUAL_VOLAT = 0.12
    BID_ASK_SP = 1 / 8
    llambda_list = [1e-06]

    # train_coverage_list = [25,50,75,100,125,150,175,200,225,250]
    train_coverage_list = [2500]
    eps = [10]

    file_name = f'{start_date}_{end_date}_{TOTAL_SHARES}_{LIQUIDATION_TIME}.csv'
    directory = f'{os.getcwd()}/{file_name}'
    current_dir = os.getcwd()


    label = []
    label.extend(['Train Coverage', 'Total Epoches'])
    for code in ticker:
        label.append(f'{code}_Mean_RL-TWAP')
        label.append(f'{code}_STDEV_RL-TWAP')
        label.append(f'{code}_WinRate_RL-TWAP')
        label.append(f'{code}_Mean_RL-AC')
        label.append(f'{code}_STDEV_RL-AC')
        label.append(f'{code}_WinRate_RL-AC')

    if not os.path.isfile(directory):
        with open(file_name, 'w', newline='') as new_csv_file:
            wr = csv.writer(new_csv_file)
            wr.writerow(label)
            pass


    for ep in eps:
        for train_coverage in train_coverage_list:
            
            simulation = AC_RL_model(ep, train_coverage, ticker, TOTAL_SHARES, NUM_TRADES, LIQUIDATION_TIME, start_date, end_date, ANNUAL_VOLAT,
                                        BID_ASK_SP, llambda_list)

            model_name = f'{os.getcwd()}/models/{str(ep)+"epoches_"+str(train_coverage)+"daysTrain_"+str(TOTAL_SHARES)+"shares_"+str(LIQUIDATION_TIME)+"days"}.pt'                            

            if os.path.isfile(model_name):
                pass
                # print("File Already Exists")
            else:
                simulation.train()

            csv_row = simulation.test()
            f = open(file_name,'a', newline="")
            wr = csv.writer(f)
            wr.writerow(csv_row)
            f.close()

            # if os.path.isfile(model_name):
            #     pass
            #     # print("File Already Exists")
            # else:
                
            #     f = open(file_name,'a', newline='')
            #     wr = csv.writer(f)
            #     wr.writerow(csv_row)
            #     f.close()




