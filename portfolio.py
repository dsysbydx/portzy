import copy
import numpy as np
import pandas as pd
import datetime as dt
from currency import USD_THB
import gsutils
from price import YahooPrice
import traceback
from error import Error

#TODO: Add return
#TODO: Adj close issue vs dividend
#TODO: More unittests
#TODO: Refactor code

YH_THAI_SUFFIX = ".BK"

def is_local_ticker(ticker):
    return ticker.endswith(YH_THAI_SUFFIX)

def get_action_sign(action_str):
    if action_str.lower().startswith('b'):
        return 1
    if action_str.lower().startswith('s'):
        return -1
    #TODO: Add deposit, withdraw cash
    print(f"Unknown action {action_str}: return zero")
    return 0

def contain_keyword(s, keywords):
    for kw in keywords:
        if kw in s:
            return True
    return False


class Portfolio():

    DATE_COL = 'date'
    TICKER_COL = 'ticker'
    CASH_COL = 'cash'
    NUM_COL = 'num'
    PRICE_COL = 'price'
    ACTION_COL = 'action'
    REQUIRED_COL = [DATE_COL, TICKER_COL, NUM_COL, PRICE_COL, ACTION_COL]

    PRICE_SUFFIX = '_Price'
    VALUE_SUFFIX = '_Value'
    BENCHMARK_SUFFIX = '_bm'
    CASH_BM_COL = 'cash_bm'
    CASH_DEP_COL = 'cash_dep'
    CASH_DEP_BM_COL = 'cash_dep_bm'

    PORTFOLIO_VAL_COL = 'portfolio_value'
    PORTFOLIO_VAL_BM_COL = 'portfolio_value_bm'

    BENCHMARK_TICKERS = ["%5EGSPC","%5EDJI","%5EIXIC"]
    BENCHMARK_TICKER = "%5EGSPC"

    def __init__(self, df_price=None):
        self.pos = None
        self.price = df_price
        self.pos_value = None
        self.tickers = None
        self.initial_cash = None

        self.begin_date = None
        self.end_date= None

        self.df = None
        self.success = None
        self.error = None
        self.error_log = None

    @classmethod
    def from_gs(cls, gs_sheet_url, initial_cash=0.0):
        df_log = gsutils.read_gs(gs_sheet_url, row_th=5)
        return cls().from_log(df_log, initial_cash=initial_cash)

    @classmethod
    def from_log(cls, df_log, initial_cash=0.0):
        self = cls()
        self.process_trade_log(df_log, initial_cash=initial_cash)
        return self

    def get_position(self):
        return self.pos.copy() if self.pos is not None else None

    def get_portfolio_value(self):
        cols = [self.PORTFOLIO_VAL_COL, self.CASH_COL, self.CASH_DEP_COL, \
                self.PORTFOLIO_VAL_BM_COL, self.CASH_BM_COL, self.CASH_DEP_BM_COL]
        return self.pos[cols].copy() if self.pos is not None else None

    def _validate_input_log(self, df_log):
        #TODO: Handling case of multiple columns get mapped to the same thing
        df = df_log.copy()
        df.columns = [str(x).strip().lower() for x in df.columns]
        col_map = dict()
        for col in df.columns:
            if col in ['date']:
                col_map[col] = self.DATE_COL
            elif col in ['ticker','stock']:
                col_map[col] = self.TICKER_COL
            elif col in ['no.', 'num', 'qty', 'quantity']:
                col_map[col] = self.NUM_COL
            elif col in ['price', 'cost']:
                col_map[col] = self.PRICE_COL
            elif col in ['action']:
                col_map[col] = self.ACTION_COL
        left_cols = set(self.REQUIRED_COL).difference(set(col_map.values()).difference())
        if len(left_cols) > 0:
            self.success = False
            self.error = Error.MISSING_COLUMN
            self.error_log = Error.MISSING_COLUMN+": "+", ".join(left_cols)
            return False

        ##format tradelog
        self.trade_log = df.rename(columns=col_map)
        self.trade_log[self.DATE_COL] = pd.to_datetime(self.trade_log[self.DATE_COL]).dt.date
        for col in [self.PRICE_COL, self.NUM_COL]:
            self.trade_log[col] = self.trade_log[col].astype(float)
        return True

    def process_trade_log(self, df_log, initial_cash=0.0, end_date=dt.date.today()):

        ## map columns and validate input
        if self._validate_input_log(df_log) is False:
            return None

        try:
            self.trade_log.sort_values(by=self.DATE_COL,inplace=True)

            ## get begin date
            self.begin_date, self.end_date = self.trade_log[self.DATE_COL].min(), end_date
            date_ls = [self.begin_date + dt.timedelta(days=k) for k in range((self.end_date - self.begin_date).days + 1)]
            #print(self.begin_date, self.end_date, date_ls[0], date_ls[-1])

            ## get ticker list excluding cash
            ticker_ls = sorted(list(self.trade_log[self.TICKER_COL].unique()))
            ticker_ls = [x for x in ticker_ls if x.lower()!='cash']
            self.tickers = ticker_ls

            ## set initial cash
            self.initial_cash = initial_cash

            ## set up price
            self.price = self._get_price_data()

            ## set up current position dict
            cur_pos = {k: 0 for k in ticker_ls}
            cur_pos[self.CASH_COL] = initial_cash
            cur_pos[self.CASH_DEP_COL] = initial_cash

            b_cur_pos = {self.BENCHMARK_TICKER:0}
            b_cur_pos[self.CASH_COL] = initial_cash
            b_cur_pos[self.CASH_DEP_COL] = initial_cash

            ## set up daily position dict
            pos = {d: copy.copy({k: np.nan for k in ticker_ls}) for d in date_ls}
            pos[date_ls[0]][self.CASH_COL] = initial_cash

            b_pos = {d: copy.copy({self.BENCHMARK_TICKER: np.nan}) for d in date_ls}
            b_pos[date_ls[0]][self.CASH_COL] = initial_cash

            ## process ach row in trade logs
            cur_d = date_ls[0]
            for r in zip(self.trade_log[self.DATE_COL], \
                         self.trade_log[self.TICKER_COL], \
                         self.trade_log[self.ACTION_COL], \
                         self.trade_log[self.NUM_COL],\
                         self.trade_log[self.PRICE_COL]):
                d, ticker, a_str, num, p = r
                ## if new date, log eod data
                if d > cur_d:
                    pos[cur_d] = copy.copy(cur_pos)
                    b_pos[cur_d] = copy.copy(b_cur_pos)
                    cur_d = d
                # update position of ticker and date
                self._update_position(cur_pos, ticker, a_str, num, p)
                p_ben = self.price.loc[d, self.BENCHMARK_TICKER]
                self._update_benchmark_position(b_cur_pos, ticker, self.BENCHMARK_TICKER, a_str, num, p, p_ben)
            pos[cur_d] = copy.copy(cur_pos)
            b_pos[cur_d] = copy.copy(b_cur_pos)

            ## combine position and benchmark position
            self.pos = pd.DataFrame(pos).T.sort_index().fillna(method='ffill').fillna(0)
            self.benchmark_pos = pd.DataFrame(b_pos).T.sort_index().fillna(method='ffill').fillna(0)
            #TODO: refactor, right now depends on lots of assumption on naming
            self.pos = self.pos.merge(self.benchmark_pos, how='left', left_index=True, right_index=True,
                                      suffixes=('', self.BENCHMARK_SUFFIX))

            # load price data and merge price data
            diff_set = set(self.pos.columns).difference(self.price.columns)
            print("Diff set between position df and price df:", diff_set)
            self.pos = self.pos.merge(self.price.fillna(method='ffill'), how='left', left_index=True, right_index=True,
                                  suffixes=('', self.PRICE_SUFFIX))
            self.pos = self.pos.fillna(method='ffill')

            # add portfolio value
            df_v = self._get_portfolio_value(df_pos=self.pos, tickers=self.tickers +[self.CASH_COL],\
                                             port_value_col=self.PORTFOLIO_VAL_COL)
            self.pos = self.pos.join(df_v, how='left')

            # add portfolio value
            df_v = self._get_portfolio_value(df_pos=self.pos, tickers=[self.BENCHMARK_TICKER, self.CASH_BM_COL],\
                                             port_value_col=self.PORTFOLIO_VAL_BM_COL)
            self.pos = self.pos.join(df_v, how='left')

            self.success = True
        except:
            err_log = "Error processing trade log: "+traceback.format_exc()
            print(err_log)
            self.error = err_log
            self.success = False

    @classmethod
    def _update_position(cls, cur_pos, ticker, action, num, p):
        print("Updating Position from trade log: ", ticker, action, num, p)
        if ticker.lower().strip() == 'cash':
            if action.lower().strip().startswith('d'):
                cur_pos[cls.CASH_COL] = cur_pos[cls.CASH_COL] + num
                cur_pos[cls.CASH_DEP_COL] = cur_pos[cls.CASH_DEP_COL] + num
            if action.lower().strip().startswith('w'):
                cur_pos[cls.CASH_COL] = cur_pos[cls.CASH_COL] - num
                cur_pos[cls.CASH_DEP_COL] = cur_pos[cls.CASH_DEP_COL] - num
        else:
            a = get_action_sign(action)
            p_usd = p / USD_THB if is_local_ticker(ticker) else p
            cur_pos[ticker] = cur_pos[ticker] + a * num
            cur_pos[cls.CASH_COL] = cur_pos[cls.CASH_COL] - a * num * p_usd

    @classmethod
    def _update_benchmark_position(cls, cur_pos, ticker, benchmark_ticker, action, num, p, p_ben):
        #print("Updating Position on benchmark replicate port from trade log: ", date, ticker, benchmark_ticker, action, num, p)
        if ticker.lower().strip() == 'cash':
            if action.lower().strip().startswith('deposit'):
                cur_pos[cls.CASH_COL] = cur_pos[cls.CASH_COL] + num
                cur_pos[cls.CASH_DEP_COL] = cur_pos[cls.CASH_DEP_COL] + num
            if action.lower().strip().startswith('withdraw'):
                cur_pos[cls.CASH_COL] = cur_pos[cls.CASH_COL] - num
                cur_pos[cls.CASH_DEP_COL] = cur_pos[cls.CASH_DEP_COL] - num
        else:
            a = get_action_sign(action)
            p_usd = p / USD_THB if is_local_ticker(ticker) else p
            cur_pos[benchmark_ticker] = cur_pos[benchmark_ticker] + a * num * p_usd / p_ben
            cur_pos[cls.CASH_COL] = cur_pos[cls.CASH_COL] - a * num * p_usd

    @classmethod
    def _get_portfolio_value(cls, df_pos, tickers, port_value_col):
        cols = tickers
        p_cols = [f"{x}{cls.PRICE_SUFFIX}" for x in cols]
        df_value = df_pos[cols].multiply(df_pos[p_cols].rename(columns=dict(zip(p_cols, cols))))
        df_value.columns = [f"{x}{cls.VALUE_SUFFIX}" for x in cols]
        df_value[port_value_col] = df_value.sum(axis=1)
        return df_value

    def _get_price_data(self):
        """
        :return:
        df in the shape (num_date, num_tickers + num_benchmark + 1 (from cash)
        """
        if self.price is not None:
            return self.price
        min_date = self.begin_date - dt.timedelta(days=7)
        max_date = dt.date.today()
        date_ls = [min_date + dt.timedelta(days=k - 1) for k in range((max_date - min_date).days + 2)]
        min_date_str = dt.date.strftime(min_date, '%Y%m%d')
        max_date_str = dt.date.strftime(max_date, '%Y%m%d')
        tickers_ls = self.tickers + self.BENCHMARK_TICKERS
        dfs = YahooPrice.get_price_data(tickers_ls, start_date=min_date_str, end_date=max_date_str)
        df_price = dfs.loc[:, ["Date", "Adj Close"]].copy()
        df_price.columns = [''.join(col).strip().replace("Adj Close", "").replace("Date",self.DATE_COL) for col in df_price.columns]
        for ticker in tickers_ls:
            if is_local_ticker(ticker):
                df_price[ticker] = df_price[ticker]/USD_THB
        df_price[self.CASH_COL] = 1.0
        df_price[self.CASH_BM_COL] = 1.0
        df_price[self.DATE_COL] = df_price[self.DATE_COL].dt.date
        df_price = pd.DataFrame(index=date_ls).join(df_price.set_index(self.DATE_COL)).fillna(method='ffill')
        return df_price

    def get_portfolio_value_and_return(self):
        dfp = self.get_portfolio_value()
        dfp['gain'] = (dfp[self.PORTFOLIO_VAL_COL].diff() - dfp[self.CASH_DEP_COL].diff()).fillna(0)
        dfp['return'] = (dfp['gain'] / dfp[self.PORTFOLIO_VAL_COL].shift(1)).fillna(0)
        dfp['cum_gain'] = dfp['gain'].cumsum()
        dfp['cum_return'] = dfp['return'].cumsum()

        ##TODO: Implement this
        dfp['gain_bm'] = (dfp[self.PORTFOLIO_VAL_BM_COL].diff() - dfp[self.CASH_DEP_BM_COL].diff()).fillna(0)
        dfp['return_bm'] = (dfp['gain_bm'] / dfp[self.PORTFOLIO_VAL_BM_COL].shift(1)).fillna(0)
        dfp['cum_gain_bm'] = dfp['gain_bm'].cumsum()
        dfp['cum_return_bm'] = dfp['return_bm'].cumsum()
        return dfp

    def get_summary(self):
        res = dict()

        ## HISTORICAL VALUE
        df = self.get_portfolio_value_and_return()
        df['date_str'] = df.index.map(lambda x: x.strftime('%Y-%m-%d'))
        df['date_unix'] = pd.DatetimeIndex(df.index).astype(np.int64) // 10 ** 9

        res["Historical"] = dict()
        cols = [self.PORTFOLIO_VAL_COL,'gain','cum_gain','return', 'cum_return']
        bm_cols = [self.PORTFOLIO_VAL_BM_COL,'gain_bm','cum_gain_bm','return_bm', 'cum_return_bm']
        for col in cols+bm_cols:
            res["Historical"][col] = df[['date_unix',col]].values.tolist()

        ## CURRENT VALUE
        res["CurrentValue"] = df[cols].to_dict(orient='records')[-1]

        return res

