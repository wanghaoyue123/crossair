import riptable as rt
from riptable.rt_stats import statx
import numpy as np
import pandas as pd

rt.TypeRegister.DisplayOptions.PRECISION = 4

# Fast array
arr = rt.FastArray([1,2,3,4,5]); rt.FA([1,2,3,4,5], dtype=float) # create from list

arr.describe(); statx(arr)
arr[1:3]; arr[[1,2,4]]
arr.sum(); arr.mean(); arr.median(); arr.var(); arr.std(); arr.max(); arr.argmax()
arr2 = rt.FA([1.0, 2.0, 3.0, rt.nan]) # Missing value
arr2.nansum(); arr2.nanmean(); arr2.nanmedian(); arr2.nanvar(); arr2.nanstd(); arr2.nanmax(); arr2.nanargmin()


# Create dateset
df = rt.Dataset({"A": [1,2,3,4], "B": [4,5,6,9]}) # from duct
rt.Dataset() #create empty
rt.zeros(10); rt.zeros(10, int); rt.ones(5)
rt.repeat(234.123, 10) 
rt.tile([1,2,3], 5)
rt.arange(10)
rt.Date.range('20190201', '20190208')
rt.Date.range('20190201', days=5, step=2)



# Column operations (each column is a FastArray)
df.C = ["cat", "cat", "dog", "dog"] # add/redefine a column
df.C = 100 # propogation
df["C"]; df[["A", "B"]] # another way to query columns
df.col_filter(["A", "B"]); df.col_filter(regex=".")
df.A[1] == df[1, "A"] == df[1, 0]
df[1:3, :]
del df.C # delete a column
df.col_rename("A", "AA"); df.col_rename("AA", "A")

# ## new
# df.get_ncols(); df.get_nrows()
# df.col_map({"old1": "new1", "old2":"new2"})
# df.dtypes
# df.imatrix_make()
# df.size # number of elements (row * col)
# df.total_size # size in bytes
# df.abs()
# rt.stack_rows([ds1, ds2, ds3])






# Basic info
df.dtypes; df.shape
df.head(5); df.tail(10); df.sample(20); df.describe();


# Files and transformations
df = rt.Dataset({"A": [1,2,3,4], "B": [4,5,6,9]})
arr = rt.FA([1,2,3,4,5])
df.to_pandas()
df.save("my_df.sds"); arr.save("my_arr.sds")
df2 = rt.load_sds("my_df.sds"); arr2 = rt.load_sds("my_arr.sds")
df.to_pandas().to_csv("my_df2.csv"); 
rt.Dataset(pd.read_csv("my_df2.csv"))
# print(type(arr2))



# Others
df = rt.Dataset({"A": [2,1,3,4], "B": [4,5,1,9]}) 
df.sort_copy("A"); df.sort_inplace(by="B", ascending=False)
df.B.issorted()


# Filter
arr = rt.FA([1,2,3,4,5,6])
df = rt.Dataset({"A": [1,2,3,4,5,6], "B": [4,5,6,9,-1,2]})
arr.filter(arr<=4)
df[df.A>3, :]; df.filter(df.A>3)
df.filter((df.A>=3) & (df.B>0))
res = rt.where(df.A>3, 100, 50) # res is a FA, with entry 100 where df.A>3, o.w. 50.
df.B.nansum(filter=(df.A>3))



# riptable structure
s = rt.Struct()
s.dataset = df; s.array = arr; s.name="sig"


# Date and time
# 3 classes: rt.Date, rt.DateTimeNano, rt.TimeSpan
dates = rt.Date(['20210101', '20210519', '20220308']) # each element is an "riptable.rt_datetime.DateScalar"
dates._fa # convert to an FA, each element is the number of days since 1970-1-1
rt.Date(['12/31/19', '6/30/19', '02/21/19'], format='%m/%d/%y')
dates.year; dates.month; dates.day_of_month; dates.day_of_week; dates.day_of_year; dates.monthyear # return an FastArray
dates.start_of_month; dates.start_of_week
dates.strftime('%b%y')
sp = dates.max() - dates.min() # returned type is a "DateSpan" object
rt.Date("20231002") + sp # DateSpan can be added to a Date
dates - dates.start_of_month # return an array of DateSpan

dtn = rt.DateTimeNano(['20210101 09:31:15', '20210519 05:21:17'], from_tz='GMT', to_tz='GMT') # from_tz and to_tz can choose from: NYC, DUBLIN, Australia/Sydney, GMT, UTC
dtn2 = rt.DateTimeNano(['12/31/19', '6/30/19'], format='%m/%d/%y', from_tz='NYC')
rt.DateTimeNano([1609511475000000000, 1621416077000000000], from_tz='NYC') # Given the number of nanoseconds since 1970-1-1
rt.DateTimeNano.random(200)
dtn._fa # A FastArray with intergers (number of nanoseconds from 1970-1-1 00:00:00)
rt.Date(dtn) # get the dates
dtn.strftime('%m/%d/%y %H:%M:%S'); dtn.strftime('%H:%M:%S');  # An array of strings of the specific format
dtn.time_since_midnight(); dtn.nanos_since_midnight(); dtn.time_since_start_of_year() # return type: TimeSpan
time_diff = dtn -dtn2 # Type: TimeSpan
dtn3 = dtn2+time_diff

rt.TimeSpan(['09:00', '10:45', '02:30', '15:00', '23:10'])
rt.TimeSpan([9, 10, 12, 14, 18], unit='h')




# Groupby
cat = rt.Categorical(["A", "B", "B", "A"]); rt.Cat(["A", "B", "B", "A"]) # <class 'riptable.rt_categorical.Categorical'>
arr = [1,2,5,2]
cat.sum(arr); cat.max(arr); cat.var(arr); cat.mean(arr); cat.median(arr); cat.nansum(arr); cat.nanmax(arr); cat.nanvar(arr); cat.nanmean(arr)  
# A Dataset containing the groups from the Categorical and the result of the operation we called on each group.
cat.count(); cat.count_uniques(arr); cat.first(arr); cat.last(arr); 
cat.apply_reduce(lambda x: x.min()+2, arr)
cat.apply_reduce(lambda x,y: x.max()*y.sum(), (arr, [1,2,3,4]))

cat.cumsum(arr); cat.cumprod(arr) # Same shape as arr, but within each group replaced by the cumsum
prev_val = cat.shift(arr) # # Same shape as arr, but within each group replaced by the previous value
cat.rolling_sum(arr, 2); cat.rolling_mean(arr,2); cat.rolling_nansum(arr,2)
cat.apply_nonreduce(lambda x: x.cumsum()+2, arr)
cat.apply_nonreduce(lambda x,y: x.cumsum()+y, (arr, [1,2,3,4]))

cat.sum(arr, transform=True) # Same length as arr; all elements with a group is the reduced value

df = rt.Dataset({"c1": [2,1,3,4], "c2": [4,5,1,9], "c3": ["p", "q", "r", "s"]}) 
cat.sum([df.c1, df.c2]); cat.sum(df) # Apply to multiple cols or the whole dataset. If a colum cannot apply the operation, that col is not returned.

cat.expand_array; cat._fa; cat.category_array; cat.unique()
rt.Categorical([1, 3, 2, 2, 1, 3, 3, 1], categories=['a','b','c']) # base is 1; integer 0 is reserved for filtered values
cat2 = rt.Cat(["A", "B", "A", "A", "B"], filter=[False, True, True, True, True])
cat2.sum([1,2,3,4,5]) # A: 7; B: 7

lab1 = ["A", "B", "A", "A", "B"]; lab2 = ["cat", "cat", "cat", "dog", "dog"]
# cat3 = rt.Cat([lab1, lab2]) # multi-key catergories

rt.cut(range(100), bins=5) # partition values into equal-width bins. Return a catergory
rt.cut(range(100), bins=[0,80,100]) # create our 2 bins: 0-80, 80-100
rt.qcut(range(100), q=3) # bins based on sample quantiles







# Concatenate Datasets
ds1 = rt.Dataset({'A': ['A0', 'A1', 'A2'], 'B': ['B0', 'B1', 'B2']})
ds2 = rt.Dataset({'A': ['A3', 'A4', 'A5'], 'B': ['B3', 'B4', 'B5'], 'C': ['C3', 'C4', 'C5'] })
rt.Dataset.concat_rows([ds1, ds2]) # up-down concat, filled with "" (for strings) or nan (for floats)
ds1 = rt.Dataset({'A': ['A0', 'A1', 'A2'], 'B': ['B0', 'B1', 'B2']})
ds2 = rt.Dataset({'C': ['C0', 'C1', 'C2'], 'D': ['D0', 'D1', 'D2']})
rt.Dataset.concat_columns([ds1, ds2], do_copy=True) 
# When do_copy==True, changes you make to values in the original Datasets do not change the values in your new, concatenated Dataset





# Merge
df1 = rt.Dataset({"name": ["Chris", "Alice", "Daniel"], "age":[23, 43, 19] } )
df2 = rt.Dataset({"name": ["Alice", "Daniel", "Bob"], "gender":["F", "M", "M"] } )
df3 = rt.Dataset({"NAME": ["Alice", "Daniel", "Bob"], "gender":["F", "M", "M"] } )
df4 = rt.Dataset({"name": ["Alice", "Daniel", "Bob"], "gender":["F", "M", "M"], "age":[0,0,0] } )
df5 = rt.Dataset({"name": ["Alice", "Daniel", "Bob", "Alice"], "gender":["F", "M", "M", "M"] } )

df1.merge_lookup(df2, on="name"); rt.merge_lookup(df1, df2, on="name") # Only retain records in df1, add info from df2
df1.merge_lookup(df3, left_on="name", right_on="NAME") # Will retain both cols "name" and "NAME"
df1.merge_lookup(df3, left_on="name", right_on="NAME", columns_right=["gender"]) # Choose which right cols to retain
df1.merge_lookup(df4, on="name", suffixes=["_1", "_2"]) # suffices is used when there are other cols with same name
df1.merge_lookup(df5, on="name", keep="last") # If there are duplicate keys in df5, "keep" can specify the first or the last to be merged

df1 = rt.Dataset({'Symbol': ['AAPL', 'AMZN', 'AAPL'],
                 'Venue': ['A', 'I', 'A'],
                 'Time': rt.TimeSpan(['09:30', '10:00', '10:20'])})
df2 = rt.Dataset({'Symbol': ['AMZN', 'AMZN', 'AMZN', 'AAPL', 'AAPL', 'AAPL'],
                      'Spot Price': [2000.0, 2025.0, 2030.0, 500.0, 510.0, 520.0],
                      'Time': rt.TimeSpan(['09:25', '09:30', '10:00', '10:00', '10:25', '10:25'])})
# merge_asof need that the "on" field is sorted (ascending)
df1.merge_asof(df2, on='Time', by='Symbol', direction='backward', matched_on=True) # match in the record in df2 that is closest before (in Time)
df1.merge_asof(df2, on='Time', by='Symbol', direction='forward', matched_on=True) # match in the record in df2 that is closest after (in Time)
df1.merge_asof(df2, on='Time', by='Symbol', direction='nearest', matched_on=True) # match in the record in df2 that is closest (in Time)






# Accum and pivot
cat1 = rt.Cat(["A", "B", "B", "A", "A", "B", "B"])
cat2 = rt.Cat(["T", "H", "H", "H", "T", "T", "H"])
fa1 = rt.FA([1,2,3,4,5,6,7])
fa2 = rt.FA([8,9,10,11,12,13,14])
ft1 = rt.FA([True, True, True, True, True, True, True])
rt.Accum2(cat1, cat2) # 2x2 main cells with catergories (A,T), (A,H), (B,T) and (B,H) resp: the counts of all values with given catergory combination
rt.Accum2(cat1, cat2).mean(fa1, filter=ft1) # # 2x2 main cells with catergories (A,T), (A,H), (B,T) and (B,H) resp: the mean of all values with given catergory combination
rt.accum_ratio(cat1, cat2, fa1, fa2, include_numer=True, include_denom=True) 
# 2x2 main cells: sum of values in fa1 (with given cat comb) / sum of values in fa2 (with given cat comb)
# It is the division of the following two tables:
# rt.Accum2(cat1, cat2).nansum(fa1)
# rt.Accum2(cat1, cat2).nansum(fa2)
rt.accum_ratiop(cat1, cat2, norm_by="C") # 2x2 main cells: the nansums of all values with catergories (A,T), (A,H), (B,T) and (B,H) resp. Normalized by column
rt.accum_ratiop(cat1, cat2, norm_by="R")
rt.accum_ratiop(cat1, cat2, norm_by="T")
ds = rt.Dataset()
ds.cat1 = cat1; ds.cat2 = cat2; ds.fa1 = fa1
pivot_table = ds.pivot(cat1, cat2, fa1)





rng = np.random.default_rng(12345)



