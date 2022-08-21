import pandas as pd


def plt_collection_to_series(collection_p):
    arr_l = collection_p.get_offsets().data.T
    series_l = pd.Series(data=arr_l[1], index=arr_l[0])
    return series_l


def assert_list_near(left_list, right_list, tol=1e-8):
    assert len(left_list) == len(right_list)
    assert all([abs(left-right) < tol for left, right in zip(left_list, right_list)])


def flatten_list_of_lists(list_of_lists):
    return [element for sub_list in list_of_lists for element in sub_list]


def assert_listoflist_near(left_listol, right_listol, tol=1e-8):
    left_list = flatten_list_of_lists(left_listol)
    right_list = flatten_list_of_lists(right_listol)
    assert_list_near(left_list, right_list)
