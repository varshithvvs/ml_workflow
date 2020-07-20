# -*- coding: utf-8 -*-
"""
ETL for Raw Data.

@author: Varshtih
"""

import dask.dataframe as dd


def etl_inputs():
    """ETL for Input Files."""
    # Load Input Files
    sales_15 = dd.read_csv(r"Data Files\2015_sales_data.csv")
    sales_16 = dd.read_csv(r"Data Files\2016_sales_data.csv")
    sales_17 = dd.read_csv(r"Data Files\2017_sales_data.csv")
    sales_18 = dd.read_csv(r"Data Files\2018_sales_data.csv")
    footfall = dd.read_csv(r"Data Files\foot_fall.csv")
    discount = dd.read_csv(r"Data Files\historical_discount.csv")
    product = dd.read_csv(r"Data Files\product_information.csv")
    city = dd.read_csv(r"Data Files\city_dict.csv")
    discount_test = dd.read_csv(r"Data Files\expected_discount.csv")
    test_data = dd.read_csv(r"Data Files\test_data.csv")

    # ETL the data to create train.csv and test.csv
    month_list_15_16 = [4, 5, 6, 7, 8, 9]
    month_list_17 = [4, 5, 6, 7, 8, 9, 11, 12]
    sales_15 = sales_15[dd.to_datetime(sales_15['date']).dt.month.isin(month_list_15_16)]
    sales_16 = sales_16[dd.to_datetime(sales_16['date']).dt.month.isin(month_list_15_16)]
    sales_17 = sales_17[dd.to_datetime(sales_17['date']).dt.month.isin(month_list_17)]
    footfall = footfall.melt(id_vars=['city']).rename(columns={'variable': 'date', 'value': 'footfall'})
    sales = dd.concat([sales_15, sales_16, sales_17, sales_18])
    discount.columns = list(discount.columns.str.lstrip('Discount_'))
    discount = discount.melt(id_vars=['date', 'product']).rename(columns={'variable': 'city', 'value': 'discount_flag'})
    discount_test.columns = list(discount_test.columns.str.lstrip('Discount_'))
    discount_test = discount_test.melt(id_vars=['date', 'product']).rename(columns={'variable': 'city', 'value': 'discount_flag'})
    sales = sales.groupby(['date', 'city', 'product']).sum().reset_index()
    footfall = footfall.groupby(['date', 'city']).sum().reset_index()
    footfall = footfall.merge(city, how='left', left_on='city', right_on='city')
    footfall['city'] = footfall['id']
    footfall = footfall.drop(['id'], axis=1)
    sales['date'] = dd.to_datetime(sales['date']).dt.strftime('%m-%d-%y')
    footfall['date'] = dd.to_datetime(footfall['date']).dt.strftime('%m-%d-%y')
    discount = discount.merge(city, how='left', left_on='city', right_on='city')
    discount['city'] = discount['id']
    discount = discount.drop(['id'], axis=1)
    discount['date'] = dd.to_datetime(discount['date']).dt.strftime('%m-%d-%y')
    discount_test = discount_test.merge(city, how='left', left_on='city', right_on='city')
    discount_test['city'] = discount_test['id']
    discount_test = discount_test.drop(['id'], axis=1)
    discount_test['date'] = dd.to_datetime(discount_test['date']).dt.strftime('%m-%d-%y')
    test_data['date'] = dd.to_datetime(test_data['date']).dt.strftime('%m-%d-%y')
    product = product.drop(['var_4', 'var_7'], axis=1)

    train = sales.merge(footfall, how='left', left_on=['date', 'city'], right_on=['date', 'city'])
    train = train.merge(discount, how='left', left_on=['date', 'city', 'product'], right_on=['date', 'city', 'product'])
    train = train.merge(product, how='left', left_on=['product'], right_on=['product'])
    # train = train[train['discount_flag'].notnull()]
    train['day'] = dd.to_datetime(train['date']).dt.day.astype(int)
    train['month'] = dd.to_datetime(train['date']).dt.month.astype(int)
    train['year'] = dd.to_datetime(train['date']).dt.year.astype(int)
    train['week_day'] = dd.to_datetime(train['date']).dt.weekday.astype(int)
    train['month_start'] = dd.to_datetime(train['date']).dt.is_month_start.astype(int)
    train['month_end'] = dd.to_datetime(train['date']).dt.is_month_end.astype(int)
#    train['quarter_start'] = dd.to_datetime(train['date']).dt.is_quarter_start.astype(int)
#    train['quarter_end'] = dd.to_datetime(train['date']).dt.is_quarter_end.astype(int)
    train = train.drop(['date'], axis=1)

    test = test_data.merge(discount_test, how='left', left_on=['date', 'city', 'product'], right_on=['date', 'city', 'product'])
    test = test.merge(product, how='left', left_on=['product'], right_on=['product'])
    test['day'] = dd.to_datetime(test_data['date']).dt.day.astype(int)
    test['month'] = dd.to_datetime(test_data['date']).dt.month.astype(int)
    test['year'] = dd.to_datetime(test_data['date']).dt.year.astype(int)
    test['week_day'] = dd.to_datetime(test_data['date']).dt.weekday.astype(int)
    test['month_start'] = dd.to_datetime(test['date']).dt.is_month_start.astype(int)
    test['month_end'] = dd.to_datetime(test['date']).dt.is_month_end.astype(int)
#    test['quarter_start'] = dd.to_datetime(test['date']).dt.is_quarter_start.astype(int)
#    test['quarter_end'] = dd.to_datetime(test['date']).dt.is_quarter_end.astype(int)
    test = test.drop(['date'], axis=1)

    return train, test
