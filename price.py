from pandas_datareader import data as pdr

class YahooPrice:

    @classmethod
    def get_price_data(cls, tickers, start_date, end_date):
        """

        :param ticker: ticker of the stock in Yahoo
        :type ticker: str (eg. cpall, PTT)
        :param start_date: start date of the data
        :type start_date: str in the format 'YYYY-MM-DD'
        :param end_date: the last date of the data
        :type end_date str in the format 'YYYY-MM-DD'
        :return: pd.DataFrame
        """
        # DOWNLOADING DATA
        tickers_ls = [x.upper() for x in tickers]
        data = pdr.get_data_yahoo(
            tickers_ls,
            start=start_date, end=end_date
        ).reset_index()
        #data['yahoo_symbol'] = ticker_yh
        return data


