from price import YahooPrice


class TestYahooPrice():

    def test_yahoo_price_benchmark(self):
        yh = YahooPrice.get_price_data(["SPY","CPALL.BK"], start_date='2020-01-01', end_date='2020-12-31')
        print(yh)