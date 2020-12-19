from portfolio import Portfolio
import datetime as dt
import pandas as pd

class TestPortfolio():

    def test_compute_portfolio_value(self):

        mock_log = [
            {"date": dt.date(2020, 1, 1), "ticker": "Cash", "action": "Deposit", "price": 1, "no.": 100000},
            {"date": dt.date(2020, 1, 1), "ticker":"CPALL.BK", "action": "B", "price":69, "no.":5},
            {"date": dt.date(2020, 1, 4), "ticker": "CPALL.BK", "action": "B", "price": 62, "no.": 10},
            {"date": dt.date(2020, 1, 7), "ticker": "CPALL.BK", "action": "S", "price": 70, "no.": 15},
            {"date": dt.date(2020, 1, 1), "ticker": "FB", "action": "B", "price": 140, "no.": 10},
            {"date": dt.date(2020, 1, 4), "ticker": "FB", "action": "S", "price": 145, "no.": 5},
            {"date": dt.date(2020, 1, 7), "ticker": "FB", "action": "B", "price": 70, "no.": 5},
            {"date": dt.date(2020, 1, 8), "ticker": "FB", "action": "S", "price": 145, "no.": 10},
            {"date": dt.date(2020, 1, 9), "ticker": "FB", "action": "B", "price": 145, "no.": 5},
            {"date": dt.date(2020, 1, 9), "ticker": "Cash", "action": "Withdraw", "price": 1, "no.": 10000}
        ]
        df_log = pd.DataFrame(mock_log)

        Port = Portfolio().from_log(df_log, initial_cash=100000)
        print(Port.tickers)
        dfp = Port.get_position()

        ## DEBUGGING
        # print('===================')
        # print(dfp.head(10).to_string())
        #print('===================')
        #print(dfp.tail().to_string())

        assert dfp.loc[dt.date(2020, 1, 8), 'CPALL.BK'] == 0
        assert dfp.loc[dt.date(2020, 1, 8), 'FB'] == 0
        assert dfp.loc[dt.date(2020, 12, 1), 'FB'] == 5

        ## Test correct USD unit
        assert 200 <= dfp.loc[dt.date(2020, 1, 1), 'FB_Price'] <= 206
        assert 60/31 <= dfp.loc[dt.date(2020, 1, 1), 'CPALL.BK_Price'] <= 70/30

    def test_validate_trade_log_column(self):
        df_log = pd.DataFrame(columns=['Date', 'ticker', 'Price', 'No.', 'Commission', 'action'])
        p = Portfolio()
        p._validate_input_log(df_log)
        assert p.success is None
        assert list(p.trade_log.columns) == ['date','ticker','price', 'num', 'commission','action']
        assert p.trade_log.shape == (0, 6)

        df_log = pd.DataFrame(columns=['Date','ticker','Price','No.', 'Commission'])
        p = Portfolio()
        p._validate_input_log(df_log)
        assert p.success is False

        df_log = pd.DataFrame(columns=['ticker', 'Price', 'No.', 'Commission'])
        p = Portfolio()
        p._validate_input_log(df_log)
        assert p.success is False

    def test_compute_port_from_log(self):
        ##TODO: Result look weird invstiage this
        df_log = pd.read_csv("static_files/mock_log.csv").head(20)
        p = Portfolio.from_log(df_log)
        assert p.success is True
        #print(p.get_portfolio_value().to_string())

    def test_compute_portfolio_value_return_cash_inflow(self):

        mock_log = [
            {"date": dt.date(2020, 1, 1), "ticker":"CPALL.BK", "action": "B", "price":69, "no.":5},
            {"date": dt.date(2020, 1, 2), "ticker": "Cash", "action": "Deposit", "no.": 100000},
            {"date": dt.date(2020, 1, 2), "ticker": "CPALL.BK", "action": "S", "price": 70, "no.": 5}
        ]
        df_log = pd.DataFrame(mock_log)

        Port = Portfolio()
        Port.process_trade_log(df_log, initial_cash=0, end_date=dt.date(2020,1,3))
        print(Port.tickers)
        dfp = Port.get_portfolio_value()
        print(dfp.to_string())

        ## test correct local currency
        assert -20 < dfp.iloc[0]['cash'] < 0
        assert -20 < dfp.iloc[0]['cash_bm'] < 0

        ## test correct cash deposit logic
        assert dfp.iloc[-1]['cash_dep_bm'] == 100000
        assert dfp.iloc[-1]['cash_dep']==100000
        assert dfp.iloc[-1]['cash'] > 100000

