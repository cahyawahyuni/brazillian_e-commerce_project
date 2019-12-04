from dash import Dash
from dash.dependencies import Input, Output
import dash_html_components as html
import flask
import dash_core_components as dcc
from datas import prod_cat_col
from flask import Flask, render_template, request,url_for
from recommendation import demo,similarity
import pickle
import datetime
import dash_bootstrap_components as dbc



col = ['all','cool_stuff', 'pet_shop', 'moveis_decoracao', 'perfumaria',
       'ferramentas_jardim', 'utilidades_domesticas', 'telefonia',
       'beleza_saude', 'livros_tecnicos', 'fashion_bolsas_e_acessorios',
       'cama_mesa_banho', 'esporte_lazer', 'consoles_games',
       'moveis_escritorio', 'malas_acessorios', 'alimentos',
       'agro_industria_e_comercio', 'eletronicos',
       'informatica_acessorios', 'construcao_ferramentas_construcao',
       'audio', 'bebes', 'construcao_ferramentas_iluminacao',
       'brinquedos', 'papelaria', 'industria_comercio_e_negocios',
       'relogios_presentes', 'automotivo', 'eletrodomesticos',
       'moveis_cozinha_area_de_servico_jantar_e_jardim', 'climatizacao',
       'casa_conforto', 'telefonia_fixa', 'portateis_casa_forno_e_cafe',
       'fraldas_higiene', 'sinalizacao_e_seguranca',
       'instrumentos_musicais', 'eletroportateis',
       'construcao_ferramentas_jardim', 'artes', 'casa_construcao',
       'livros_interesse_geral', 'artigos_de_festas',
       'construcao_ferramentas_seguranca', 'cine_foto',
       'fashion_underwear_e_moda_praia', 'fashion_roupa_masculina',
       'alimentos_bebidas', 'bebidas', 'moveis_sala', 'market_place',
       'musica', 'fashion_calcados', 'flores', 'eletrodomesticos_2',
       'fashion_roupa_feminina', 'pcs', 'livros_importados',
       'artigos_de_natal', 'moveis_quarto', 'casa_conforto_2',
       'portateis_cozinha_e_preparadores_de_alimentos', 'dvds_blu_ray',
       'cds_dvds_musicais', 'artes_e_artesanato',
       'moveis_colchao_e_estofado', 'tablets_impressao_imagem',
       'construcao_ferramentas_ferramentas', 'fashion_esporte',
       'la_cuisine', 'pc_gamer', 'seguros_e_servicos',
       'fashion_roupa_infanto_juvenil']

epoch = datetime.datetime.utcfromtimestamp(0)

def unix_time_millis(dt):
    return (dt - epoch).total_seconds() #* 1000.0

df_ex_sales = pickle.load(open('df_ex_sales.sav','rb'))
df_ex_sales_monthly = pickle.load(open('df_ex_sales_monthly.sav','rb'))
df_sp_train = pickle.load(open('df_sp_train.sav','rb'))
df_sp_test = pickle.load(open('df_sp_test.sav','rb'))
y_hat_avg = pickle.load(open('y_hat_avg.sav','rb'))
data_forecast = pickle.load(open('data_forecast.sav','rb'))
data_satu = pickle.load(open('data_satu.sav','rb'))
data_dua = pickle.load(open('data_dua.sav','rb'))
data_tiga = pickle.load(open('data_tiga.sav','rb'))
data_empat = pickle.load(open('data_empat.sav','rb'))
data_lima = pickle.load(open('data_lima.sav','rb'))
eng_word_satu = pickle.load(open('eng_word_satu.sav','rb'))
eng_word_dua = pickle.load(open('eng_word_dua.sav','rb'))
eng_word_tiga = pickle.load(open('eng_word_tiga.sav','rb'))
eng_word_empat = pickle.load(open('eng_word_empat.sav','rb'))
eng_word_lima = pickle.load(open('eng_word_lima.sav','rb'))

app = Flask(__name__)
app_dash = Dash(__name__, server=app, url_base_pathname='/pathname/')
app_dash2 = Dash(__name__, server=app, url_base_pathname='/pathname2/')
app_dash3 = Dash(__name__, server=app, url_base_pathname='/pathname3/')

app_dash2.layout = html.Div([
    html.Div([html.H1('Sales Prediction using Time Series Forecasting')],
    style={'height':'13%'}),
    html.Div([
        html.Div([dcc.Graph(id='my_graph2')],style={'height':'80%',
        'width':'48%',
        'border-style': 'inset',
        'display': 'inline-block'}),
        html.Div([dcc.Graph(id='my_graph3')],style={'height':'80%',
        'width':'48%',
        'border-style': 'inset',
        'display': 'inline-block'}),
        html.Div([dcc.RadioItems(id='my_radio2',options=[
                    {'label': 'All', 'value': 'all'},
                    {'label': 'Train', 'value': 'train'},
                    {'label': 'Test', 'value': 'test'},
                    {'label': 'Test & Sarima', 'value': 'sarima'}],
                    value='all',labelStyle={'display':'inline-block', 'margin':'5px'})],style={'height':'10%',
        'width':'48%',
        'display': 'inline-block'}),
        html.Div([dcc.RangeSlider(id='my_slider2',
                        min=unix_time_millis(data_forecast['Date'].min()),
                        max=unix_time_millis(data_forecast['Date'].max()),
                        value=[unix_time_millis(data_forecast['Date'].min()),unix_time_millis(data_forecast['Date'].max())]
                    )],style={'height':'10%',
        'width':'48%',
        'display': 'inline-block'})
    ],style={'height':'58%',
        'padding': '60px'}),
    html.Div(style={'height':'10%'})
    ],
    style={
            'text-align':'center', 
            'width':'100%'
            })

app_dash3.layout = html.Div([
    html.Div([html.H1('Sentiment Analysis using NLP')],
    style={'height':'13%'}),
    html.Div([
        html.Div([dcc.Graph(id='my_graph4')],style={
        'width':'60%',
        'display': 'inline-block',
        'border-style': 'outset'
        }),
        html.Div([html.Table([
        html.Tr([html.Td(id='a')]),
        html.Tr([html.Td(id='b')]),
        html.Tr([html.Td(id='c')]),
        html.Tr([html.Td(id='d')]),
        html.Tr([html.Td(id='e')]),
        html.Tr([html.Td(id='f')]),
        html.Tr([html.Td(id='g')]),
        html.Tr([html.Td(id='h')]),
        html.Tr([html.Td(id='i')]),
        html.Tr([html.Td(id='j')]),
        html.Tr([html.Td(id='k')]),
        html.Tr([html.Td(id='l')]),
        html.Tr([html.Td(id='m')]),
        html.Tr([html.Td(id='n')]),
        html.Tr([html.Td(id='o')]),
        html.Tr([html.Td(id='p')]),
        html.Tr([html.Td(id='q')]),
        html.Tr([html.Td(id='r')]),
        html.Tr([html.Td(id='s')]),
        html.Tr([html.Td(id='t')]),
    ])],style={
        'width':'10%',
        'border-style': 'outset',
        'display': 'inline-block',
        'backgroundColor': '#F5F2F5',
        'text-align':'left',
        'align':'center'}),
        html.Div([dcc.RadioItems(id='my_radio3',options=[
                    {'label': 'Rating 1', 'value': 'satu'},
                    {'label': 'Rating 2', 'value': 'dua'},
                    {'label': 'Rating 3', 'value': 'tiga'},
                    {'label': 'Rating 4', 'value': 'empat'},
                    {'label': 'Rating 5', 'value': 'lima'}],
                    value='satu',labelStyle={'display':'inline-block', 'margin':'5px'})],style={
        'width':'60%',
        'display': 'inline-block'}),
        html.Div(style={'height':'10%',
        'width':'10%',
        'display': 'inline-block'})
    ],style={
        # 'border-style': 'solid',
        'padding': '60px'}),
    html.Div(style={'height':'10%'
        })
    ],
    style={
            'text-align':'center',
            'width':'100%',
            })

app_dash.layout = html.Div([
    html.Div([
        html.Div([html.H1('Brazillian E-Commerce'),
        html.H2('Overview')],style={
        'text-align':'center', 
        'margin-left':'27px',
        'margin-right':'27px'}),
        html.Div([
                html.Div(style={
                    'height': '80px',
                    # 'border-style': 'inset', 
                    # 'backgroundColor': '#E1DDE0',
                    'margin':'15px'}),
                html.Div([
                html.H3('Filter by graph periode :'),
                dcc.RadioItems(id='my_radio',options=[
                    {'label': 'Daily Graph', 'value': 'DG'},
                    {'label': 'Monthly Graph', 'value': 'MG'}],
                    value='DG',labelStyle={'display':'inline-block', 'margin':'5px'}),
                html.H3('Filter by construction date (or select range in histogram):'),
                html.Div(dcc.RangeSlider(id='my_slider',
                        min=unix_time_millis(df_ex_sales['order_purchase_timestamp'].min()),
                        max=unix_time_millis(df_ex_sales['order_purchase_timestamp'].max()),
                        value=[unix_time_millis(df_ex_sales['order_purchase_timestamp'].min()),unix_time_millis(df_ex_sales['order_purchase_timestamp'].max())]
                    ),style={'width': '90%'}),
                html.H3('Filter by category :'),
                html.Div(dcc.Dropdown(
                        id='my_dropdown',
                        options=[{'label': i, 'value': i} for i in (col)],
                        value='all'
                    ),style={'width': '90%'})],style={
                'width':'90%',
                'height':'300px',
                'border-style': 'outset',
                'text-align':'left', 
                'padding-left' : '10px', 
                'margin':'2px',
                'margin-right':'50px',
                'backgroundColor': '#FAF8F9',
                'display': 'inline-block'})
                        ],
                style={'width':'30%',
                'height':'500px',
                'text-align':'left', 
                'padding-left' : '10px', 
                'margin':'2px',
                'margin-right':'50px',
                'display': 'inline-block'}),
        html.Div([
            html.Div([
                html.Div([
                    html.H3('4074'),
                    html.H5('No. City Coverage')
                ],style={'width':'18%',
                    'height': '80px',
                    'border-style': 'outset', 
                    'display': 'inline-block',
                    'backgroundColor': '#FAF8F9',
                    'margin':'15px'}),
                html.Div([
                    html.H3('73'),
                    html.H5('No. Product Category')
                ],style={'width':'18%',
                    'height': '80px',
                    'border-style': 'outset', 
                    'display': 'inline-block',
                    'backgroundColor': '#FAF8F9',
                    'margin':'15px'}),
                html.Div([
                    html.H3('32951'),
                    html.H5('No. Product ID')
                ],style={'width':'18%',
                    'height': '80px',
                    'border-style': 'outset', 
                    'display': 'inline-block',
                    'backgroundColor': '#FAF8F9',
                    'margin':'15px'}),
                html.Div([
                    html.H3('3088'),
                    html.H5('No. Seller ID')
                ],style={'width':'18%',
                    'height': '80px',
                    'border-style': 'outset', 
                    'display': 'inline-block',
                    'backgroundColor': '#FAF8F9',
                    'margin':'15px'})
            ],style={
                'margin':'2px'}),
            html.Div([dcc.Graph(id='my_graph')],style={ 
                'border-style': 'outset', 
                'margin':'2px'})],
            style={
            'width':'60%',
            'text-align':'center', 
            'display': 'inline-block',
            'vertical-align': 'top'}),
        html.Div([html.Iframe(id = 'city_cust', srcDoc=open('city_cust.html', 'r').read(),width='100%', height='100%')],style={
        'width':'45%',
        'height':'400px',
        'border-style': 'outset',  
        'text-align':'center', 
        'margin':'2px',
        'margin-right':'50px',
        'margin-top':'30px',
        'display': 'inline-block'}),
        html.Div([html.Iframe(id = 'city_seller', srcDoc=open('city_seller.html', 'r').read(),width='100%', height='100%')],style={
        'width':'45%',
        'height':'400px',
        'border-style': 'outset',  
        'text-align':'center', 
        'margin':'2px',
        'margin-top':'30px',
        'display': 'inline-block'}),
        html.Div([
                html.Div([html.H1('Customer Mapping')]),
                html.Div([html.Iframe(id = 'map_cust', srcDoc=open('map_1.html', 'r').read(),width='100%', height='100%')],style={
                    'height': '350px',
                    'border-style': 'inset'})],
        style={
        'width':'45%',
        'text-align':'center', 
        'margin':'2px',
        'margin-right':'50px',
        'margin-top':'30px',
        'display': 'inline-block'}),
        html.Div([
                html.Div([html.H1('Seller Mapping')]),
                html.Div([html.Iframe(id = 'map_seller', srcDoc=open('map2.html', 'r').read(),width='100%', height='100%')],style={
                    'height': '350px',
                    'border-style': 'outset'})],
        style={
        'width':'45%',
        'text-align':'center', 
        'margin':'2px',
        'margin-top':'30px',
        'display': 'inline-block'}),
        html.Div([html.Iframe(id = 'top_prod', srcDoc=open('graph3.html', 'r').read(),width='100%', height='100%')],style={
        'width':'45%',
        'height':'400px',
        'border-style': 'outset',  
        'text-align':'center', 
        'margin':'2px',
        'margin-right':'50px',
        'margin-top':'30px',
        'display': 'inline-block'})
    ], style={
        'text-align':'center'
        # 'backgroundColor': '#F5F2F5'
        })
    ])


@app_dash.callback(
    Output('my_graph','figure'),
    [Input('my_slider','value'),
    Input('my_radio','value'),
    Input('my_dropdown','value')])
def update_graph(slider_value,radio_value,dropdown_value):
    if radio_value == 'DG':
        if dropdown_value == 'all':
            dff = df_ex_sales.groupby(by='order_purchase_timestamp').sum().reset_index()
            df = dff[(dff['order_purchase_timestamp']>datetime.datetime.fromtimestamp(slider_value[0]))&(dff['order_purchase_timestamp']<datetime.datetime.fromtimestamp(slider_value[1]))]
            return {
                'data' : [dict(
                    x=df['order_purchase_timestamp'],
                    y=df['order_item_id'],
                    type='bar'
                )],
                'layout' : dict(
                    xaxis={'title':'order_purchase_timestamp'},
                    yaxis={'title':'order_item_id'},
                    
                )
            }
        else:
            dff = df_ex_sales[df_ex_sales['product_category_name']== dropdown_value ]
            df = dff[(dff['order_purchase_timestamp']>datetime.datetime.fromtimestamp(slider_value[0]))&(dff['order_purchase_timestamp']<datetime.datetime.fromtimestamp(slider_value[1]))]
            return {
                'data' : [dict(
                    x=df['order_purchase_timestamp'],
                    y=df['order_item_id'],
                    type='bar'
                )],
                'layout' : dict(
                    xaxis={'title':'order_purchase_timestamp'},
                    yaxis={'title':'order_item_id'}
                )
            }
    elif radio_value == 'MG':
        if dropdown_value == 'all':
            dff = df_ex_sales_monthly.groupby(by='order_purchase_timestamp').sum().reset_index()
            df = dff[(dff['order_purchase_timestamp']>datetime.datetime.fromtimestamp(slider_value[0]))&(dff['order_purchase_timestamp']<datetime.datetime.fromtimestamp(slider_value[1]))]
            return {
                'data' : [dict(
                    x=df['order_purchase_timestamp'],
                    y=df['order_item_id'],
                    type='bar'
                )],
                'layout' : dict(
                    xaxis={'title':'order_purchase_timestamp'},
                    yaxis={'title':'order_item_id'}             
                )
            }
        else:
            dff = df_ex_sales_monthly[df_ex_sales_monthly['product_category_name']== dropdown_value ]
            df = dff[(dff['order_purchase_timestamp']>datetime.datetime.fromtimestamp(slider_value[0]))&(dff['order_purchase_timestamp']<datetime.datetime.fromtimestamp(slider_value[1]))]
            return {
                'data' : [dict(
                    x=df['order_purchase_timestamp'],
                    y=df['order_item_id'],
                    type='bar'
                )],
                'layout' : dict(
                    xaxis={'title':'order_purchase_timestamp'},
                    yaxis={'title':'order_item_id'},
                    
                )
            }

@app_dash2.callback(
    Output('my_graph2','figure'),
    [Input('my_radio2','value')]
    )
def update_graph2(radio2_value):
    if radio2_value == 'all':
        return {'data' : [dict(
                    x=df_sp_train['order_purchase_timestamp'],
                    y=df_sp_train['order_item_id'],
                    name='train'
                ),
                dict(
                    x=df_sp_test['order_purchase_timestamp'],
                    y=df_sp_test['order_item_id'],
                    name='test'),
                dict(
                    x=y_hat_avg['order_purchase_timestamp'],
                    y=y_hat_avg['SARIMA'],
                    name='sarima')],
                'layout' : dict(
                    xaxis={'title':'year'},
                    yaxis={'title':'order_item_id'})}
    elif radio2_value == 'train':
        return {'data' : [dict(
                    x=df_sp_train['order_purchase_timestamp'],
                    y=df_sp_train['order_item_id'],
                    name='train'
                )],
                'layout' : dict(
                    xaxis={'title':'year'},
                    yaxis={'title':'order_item_id'})}
    elif radio2_value == 'test':
        return {'data' : [
                dict(
                    x=df_sp_test['order_purchase_timestamp'],
                    y=df_sp_test['order_item_id'],
                    name='test')],
                'layout' : dict(
                    xaxis={'title':'year'},
                    yaxis={'title':'order_item_id'})}
    elif radio2_value == 'sarima':
        return {'data' : [
                dict(
                    x=df_sp_test['order_purchase_timestamp'],
                    y=df_sp_test['order_item_id'],
                    name='test'),
                dict(
                    x=y_hat_avg['order_purchase_timestamp'],
                    y=y_hat_avg['SARIMA'],
                    name='sarima')],
                'layout' : dict(
                    xaxis={'title':'year'},
                    yaxis={'title':'order_item_id'})}

@app_dash2.callback(
    Output('my_graph3','figure'),
    [Input('my_slider2','value')]
    )
def update_graph3(slider2_value):
    dff = data_forecast
    df = dff[(dff['Date']>datetime.datetime.fromtimestamp(slider2_value[0]))&(dff['Date']<datetime.datetime.fromtimestamp(slider2_value[1]))]
    return {
        'data' : [dict(
            x=df['Date'],
            y=df['order']
        )],
        'layout' : dict(
            xaxis={'title':'Date'},
            yaxis={'title':'Predict Order'},  
        )
    }

@app_dash3.callback(
    Output('my_graph4','figure'),
    [Input('my_radio3','value')]
    )
def update_graph4(radio3_value):
    if radio3_value == 'satu':
        return {
                'data' : [dict(
                    x=data_satu['word'],
                    y=data_satu['appear'],
                    type='bar'
                )],
                'layout' : dict(
                    xaxis={'title':'word'},
                    yaxis={'title':'appear'},
                )
            }
    elif radio3_value == 'dua':
        return {
                'data' : [dict(
                    x=data_dua['word'],
                    y=data_dua['appear'],
                    type='bar'
                )],
                'layout' : dict(
                    xaxis={'title':'word'},
                    yaxis={'title':'appear'},
                )
            }
    elif radio3_value == 'tiga':
        return {
                'data' : [dict(
                    x=data_tiga['word'],
                    y=data_tiga['appear'],
                    type='bar'
                )],
                'layout' : dict(
                    xaxis={'title':'word'},
                    yaxis={'title':'appear'},
                )
            }
    elif radio3_value == 'empat':
        return {
                'data' : [dict(
                    x=data_empat['word'],
                    y=data_empat['appear'],
                    type='bar'
                )],
                'layout' : dict(
                    xaxis={'title':'word'},
                    yaxis={'title':'appear'},
                )
            }
    elif radio3_value == 'lima':
        return {
                'data' : [dict(
                    x=data_lima['word'],
                    y=data_lima['appear'],
                    type='bar'
                )],
                'layout' : dict(
                    xaxis={'title':'word'},
                    yaxis={'title':'appear'},
                )
            }

@app_dash3.callback(
    [Output('a','children'),
    Output('b','children'),
    Output('c','children'),
    Output('d','children'),
    Output('e','children'),
    Output('f','children'),
    Output('g','children'),
    Output('h','children'),
    Output('i','children'),
    Output('j','children'),
    Output('k','children'),
    Output('l','children'),
    Output('m','children'),
    Output('n','children'),
    Output('o','children'),
    Output('p','children'),
    Output('q','children'),
    Output('r','children'),
    Output('s','children'),
    Output('t','children')],
    [Input('my_radio3','value')]
    )
def update_data(radio_value3):
    if radio_value3=='satu':
        return eng_word_satu
    elif radio_value3=='dua':
        return eng_word_dua
    elif radio_value3=='tiga':
        return eng_word_tiga
    elif radio_value3=='empat':
        return eng_word_empat
    elif radio_value3=='lima':
        return eng_word_lima

@app.route('/plotly_dashboard')
def dashboard():
    return flask.redirect('/pathname/')

@app.route('/sales_forecast')
def forecast():
    return flask.redirect('/pathname2/')

@app.route('/sentiment_analysis')
def sentiment():
    return flask.redirect('/pathname3/')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommendation', methods=['GET','POST'])
def recommendation():
    if request.method == 'POST':
        data = request.form
        data = data.to_dict()
        data = list(data.keys())
        hasil1= demo(data)
        hasil2=similarity(data)
        return render_template('result.html', pred=hasil1, data=data, simi=hasil2)
    return render_template('recommendation.html',data_cat=prod_cat_col)

if __name__ == '__main__':
    app.run(debug=True, port=4444)