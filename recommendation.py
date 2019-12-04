import pickle
from pandas import DataFrame, get_dummies

df_demo = pickle.load(open('df_demo.sav','rb'))
df_similarity = pickle.load(open('similarity.sav','rb'))

def demo(x):
    table = []
    for item in x:
        new_table = df_demo[df_demo['product_category_name_english']==item][['product_id','review_score']].sort_values(by='review_score',ascending=False).head(5)
        table.append(new_table)
    return table

def similarity(x):
    top_list = []
    for item in x:
        top = list(df_similarity[item].sort_values(ascending=False)[1:4].index)
        top_list.append(top)
    table = []
    for item in top_list:
        for elemen in item:
            new_table = df_demo[df_demo['product_category_name_english']==elemen].sort_values(by='review_score',ascending=False).head(3)
            table.append(new_table)
    return table

# print(demo(['perfumery','health_beauty','housewares'])[0])
# print(similarity(['perfumery','health_beauty','housewares'])[0])