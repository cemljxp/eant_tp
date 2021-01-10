import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import dash_daq as daq
import dash_bootstrap_components as dbc

from datetime import date

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from dash.dependencies import Input, Output
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error

import pickle
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
external_stylesheets = ['https://gitcdn.link/repo/cemljxp/eant_tp/main/style.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
                meta_tags=[{'name': 'viewport','content': 'width=device-width, initial-scale=1.0'}])

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# 1.- Predicción de demanda
# Se lee el modelo
with open('modelo.pickle', 'rb') as archivo:
    model_etr_simp = pickle.load(archivo)
# Se crea el vector con las variables predictoras
val_pred = [[0, 0, 0, 0, 0, 0, 0, 0, 0]]
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# 2.- Datos
df_demanda = pd.read_csv('https://raw.githubusercontent.com/cemljxp/eant_tp/main/APP_VAR_MW_CABA_2017_2020.csv')
L_Date = list(df_demanda['Date'])
L_MW = list(df_demanda['MW'])
L_Temp_avg = list(df_demanda['Temp_avg'])
L_Temp_min = list(df_demanda['Temp_min'])
L_Temp_max = list(df_demanda['Temp_max'])
L_hPa = list(df_demanda['hPa'])
L_Hum = list(df_demanda['Hum'])
L_Wind_avg = list(df_demanda['Wind_avg'])
L_Wind_max = list(df_demanda['Wind_max'])

fig = make_subplots(rows=5, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.01)

fig.add_trace(go.Scatter(name='hPa', x=L_Date, y=L_hPa, line=dict(color='gold', width=1)),
              row=1, col=1)

fig.add_trace(go.Line(name='Temp_avg', x=L_Date, y=L_Temp_avg, line=dict(color='lawngreen', width=2)),
              row=2, col=1)

fig.add_trace(go.Scatter(name='Temp_min',x=L_Date, y=L_Temp_min, line=dict(color='deepskyblue', width=1, dash='dashdot')),
              row=2, col=1)

fig.add_trace(go.Scatter(name='Temp_max',x=L_Date, y=L_Temp_max, line=dict(color='red', width=1, dash='dashdot')),
              row=2, col=1)

fig.add_trace(go.Scatter(name='MW',x=L_Date, y=L_MW,line=dict(color='blue', width=1.5)),
              row=3, col=1)

fig.add_trace(go.Scatter(name='Hum',x=L_Date, y=L_Hum,line=dict(color='deeppink', width=1.5)),
              row=4, col=1)

fig.add_trace(go.Scatter(name='Wind_avg',x=L_Date, y=L_Wind_avg, line=dict(color='orange', width=1.5)),
              row=5, col=1)

fig.add_trace(go.Scatter(name='Wind_max',x=L_Date, y=L_Wind_max, line=dict(color='magenta', width=1, dash='dot')),
              row=5, col=1)
fig.update_yaxes(title_text="Pres. Atmosf. (hPa)", range=[970, 1050],row=1, col=1)
fig.update_yaxes(title_text="Temp. Max, Avg, Min (°C)", row=2, col=1)
fig.update_yaxes(title_text="Demanda (MW)", range=[900, 2300], row=3, col=1)
fig.update_yaxes(title_text="Humedad (%)", range=[0, 110],row=4, col=1)
fig.update_yaxes(title_text="Vel. Viento Max, Avg (km/h)", row=5, col=1)

fig.update_layout(height=1000, width=1500, title_text="Visualización de los Datos", margin=dict(l=20, r=20, t=40, b=20))
fig.update_layout({'plot_bgcolor': 'black','paper_bgcolor': 'whitesmoke',})
#fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# 3.- Outliers
df_demanda2 = df_demanda[df_demanda['Year']!=2020]
df_demanda2["Year"] =df_demanda2["Year"].astype(str)
fig1 = px.scatter(df_demanda2,
                 x='Date',
                 y='MW',
                 title="Demanda Eléctrica C.A.B.A. 2017-2019",
                 color="Year",
                 color_discrete_sequence=px.colors.qualitative.Set1,
                 labels = {'Date':'Fecha', 'MW':'Potencia (MW)', 'Year':'Año'})
fig1.update_layout({'plot_bgcolor': 'black','paper_bgcolor': 'whitesmoke',}, margin=dict(l=20, r=20, t=40, b=20))
fig1.update_xaxes(showgrid=False)
fig1.update_yaxes(range=[800,2400], tick0=200, dtick=200)
fig1.update_layout(height=500, width=1000)

df_demanda3 = df_demanda[(df_demanda['MW']>1200) & (df_demanda['Year']!=2020)]
df_demanda3["Year"] =df_demanda3["Year"].astype(str)
fig2 = px.scatter(df_demanda3,
                 x='Date',
                 y='MW',
                 title="Demanda Eléctrica C.A.B.A. 2017-2019",
                 color="Year",
                 color_discrete_sequence=px.colors.qualitative.Set1,
                 labels = {'Date':'Fecha', 'MW':'Potencia (MW)', 'Year':'Año'})
fig2.update_layout({'plot_bgcolor': 'black','paper_bgcolor': 'whitesmoke',}, margin=dict(l=20, r=20, t=40, b=20))
fig2.update_xaxes(showgrid=False)
fig2.update_yaxes(range=[1200,2400], tick0=200, dtick=200)
fig2.update_layout(height=500, width=1000)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# 4.- Modelos
df_res = pd.read_csv('https://raw.githubusercontent.com/cemljxp/eant_tp/main/APP_RES_MW_CABA_2017_2020.csv')
L_Date = list(df_res['Date'])
L_MW = list(df_res['MW'])
L_MW_pred_knr = list(df_res['MW_pred_knr'])
L_MW_pred_lr = list(df_res['MW_pred_lr'])
L_MW_pred_dtr = list(df_res['MW_pred_dtr'])
L_MW_pred_abr = list(df_res['MW_pred_abr'])
L_MW_pred_rfr = list(df_res['MW_pred_rfr'])
L_MW_pred_etr = list(df_res['MW_pred_etr'])
L_MW_pred_gbr = list(df_res['MW_pred_gbr'])
L_MW_pred_mlpr = list(df_res['MW_pred_mlpr'])
L_MW_pred_vr = list(df_res['MW_pred_vr'])
L_MW_pred_sr = list(df_res['MW_pred_sr'])

fig3 = go.Figure()

fig3.add_trace(go.Scatter(x=L_Date, y=L_MW, name='MW Real',
                         line=dict(color='blue', width=2)))

fig3.add_trace(go.Scatter(x=L_Date, y=L_MW_pred_knr, name='KNeighbors',
                         line=dict(color='firebrick', width=1.2)))

fig3.add_trace(go.Scatter(x=L_Date, y=L_MW_pred_lr, name='LinearReg',
                         line=dict(color='deeppink', width=1.2)))

fig3.add_trace(go.Scatter(x=L_Date, y=L_MW_pred_dtr, name='DecisionTree',
                         line=dict(color='lime', width=1.2)))

fig3.add_trace(go.Scatter(x=L_Date, y=L_MW_pred_abr, name='AdaBoost',
                         line=dict(color='burlywood', width=1.2)))

fig3.add_trace(go.Scatter(x=L_Date, y=L_MW_pred_rfr, name='RandomForest',
                         line=dict(color='greenyellow', width=1.2)))

fig3.add_trace(go.Scatter(x=L_Date, y=L_MW_pred_etr, name='ExtraTrees',
                         line=dict(color='yellow', width=1.2)))

fig3.add_trace(go.Scatter(x=L_Date, y=L_MW_pred_gbr, name='GradientBoosting',
                         line=dict(color='crimson', width=1.2)))

fig3.add_trace(go.Scatter(x=L_Date, y=L_MW_pred_mlpr, name='MultiLayerPerceptron',
                         line=dict(color='gold', width=1.2)))

fig3.add_trace(go.Scatter(x=L_Date, y=L_MW_pred_vr, name='VotingReg',
                         line=dict(color='red', width=1.2)))

fig3.add_trace(go.Scatter(x=L_Date, y=L_MW_pred_sr, name='StackingReg',
                         line=dict(color='aqua', width=1.2)))

fig3.update_layout(height=600, width=1500, title_text="Comparación de Modelos", margin=dict(l=20, r=20, t=80, b=20))
fig3.update_layout({'plot_bgcolor': 'black','paper_bgcolor': 'black','font_color': 'white'})
#fig3.update_xaxes(showgrid=False)
fig3.update_yaxes(showgrid=False)
fig3.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1))

df_tp = pd.read_csv('https://raw.githubusercontent.com/cemljxp/eant_tp/main/APP_TOL_PRE_MW_CABA_2017_2020.csv')

l_tol_knr = list(df_tp['tol_knr'])
l_tol_lr = list(df_tp['tol_lr'])
l_tol_dtr = list(df_tp['tol_dtr'])
l_tol_abr = list(df_tp['tol_abr'])
l_tol_rfr = list(df_tp['tol_rfr'])
l_tol_etr = list(df_tp['tol_etr'])
l_tol_gbr = list(df_tp['tol_gbr'])
l_tol_mlpr = list(df_tp['tol_mlpr'])
l_tol_vr = list(df_tp['tol_vr'])
l_tol_sr = list(df_tp['tol_sr'])

l_pre_knr = list(df_tp['pre_knr'])
l_pre_lr = list(df_tp['pre_lr'])
l_pre_dtr = list(df_tp['pre_dtr'])
l_pre_abr = list(df_tp['pre_abr'])
l_pre_rfr = list(df_tp['pre_rfr'])
l_pre_etr = list(df_tp['pre_etr'])
l_pre_gbr = list(df_tp['pre_gbr'])
l_pre_mlpr = list(df_tp['pre_mlpr'])
l_pre_vr = list(df_tp['pre_vr'])
l_pre_sr = list(df_tp['pre_sr'])

fig4 = go.Figure()

fig4.add_trace(go.Scatter(x=l_tol_knr, y=l_pre_knr, name='KNeighbors',
                         line=dict(color='firebrick', width=2.5)))

fig4.add_trace(go.Scatter(x=l_tol_lr, y=l_pre_lr, name='LinearReg',
                         line=dict(color='deeppink', width=2.5)))

fig4.add_trace(go.Scatter(x=l_tol_dtr, y=l_pre_dtr, name='DecisionTree',
                         line=dict(color='lime', width=2.5)))

fig4.add_trace(go.Scatter(x=l_tol_abr, y=l_pre_abr, name='AdaBoost',
                         line=dict(color='burlywood', width=2.5)))

fig4.add_trace(go.Scatter(x=l_tol_rfr, y=l_pre_rfr, name='RandomForest',
                         line=dict(color='greenyellow', width=2.5)))

fig4.add_trace(go.Scatter(x=l_tol_etr, y=l_pre_etr, name='ExtraTrees',
                         line=dict(color='yellow', width=2.5)))

fig4.add_trace(go.Scatter(x=l_tol_gbr, y=l_pre_gbr, name='GradientBoosting',
                         line=dict(color='crimson', width=2.5)))

fig4.add_trace(go.Scatter(x=l_tol_mlpr, y=l_pre_mlpr, name='MultiLayerPerceptron',
                         line=dict(color='gold', width=2.5)))

fig4.add_trace(go.Scatter(x=l_tol_vr, y=l_pre_vr, name='VotingReg',
                         line=dict(color='red', width=2.5)))

fig4.add_trace(go.Scatter(x=l_tol_sr, y=l_pre_sr, name='StackingReg',
                         line=dict(color='aqua', width=2.5)))

fig4.update_layout(height=600, width=1500, title_text="Precisión de los modelos Vs el Toleterancia de Error",
                    xaxis_title='Toleterancia (%)',
                    yaxis_title='Precisión (%)',
                    showlegend=True)
fig4.update_layout({'plot_bgcolor': 'black','paper_bgcolor': 'black','font_color': 'white'})
#fig4.update_xaxes(showgrid=False)
fig4.update_yaxes(showgrid=False)

df_rmse = pd.read_csv('https://raw.githubusercontent.com/cemljxp/eant_tp/main/APP_RMSE_MW_CABA_2017_2020.csv')
MW_avg = df_demanda3["MW"].mean()

fig5 = px.bar(df_rmse, x='model', y='rmse', height=500, width=1000, color='rmse', color_continuous_scale='jet_r')
fig5.update_layout({'plot_bgcolor': 'black','paper_bgcolor': 'black','font_color': 'white'})
fig5.update_layout(xaxis_tickangle=-90)
fig5.update_layout(title_text='Cross-Validation RMSE (%)- MW avg = {:.2f}'.format(MW_avg))
fig5.update_yaxes(title_text="RMSE (%)", range=[0, 11])
fig5.update_xaxes(title_text="Modelos")

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# 5.- COVID-19
df20 = pd.read_csv('https://raw.githubusercontent.com/cemljxp/eant_tp/main/APP_PRED20_MW_CABA_2017_2020.csv')

df20_01= df20[df20['Month']=='January']
df20_02= df20[df20['Month']=='February']
df20_03= df20[df20['Month']=='March']
df20_04= df20[df20['Month']=='April']
df20_05= df20[df20['Month']=='May']
df20_06= df20[df20['Month']=='June']
df20_07= df20[df20['Month']=='July']
df20_08= df20[df20['Month']=='August']
df20_09= df20[df20['Month']=='September']
df20_10= df20[df20['Month']=='October']
df20_11= df20[df20['Month']=='November']


L_Date20_01 = list(df20_01['Date'])
L_Date20_02 = list(df20_02['Date'])
L_Date20_03 = list(df20_03['Date'])
L_Date20_04 = list(df20_04['Date'])
L_Date20_05 = list(df20_05['Date'])
L_Date20_06 = list(df20_06['Date'])
L_Date20_07 = list(df20_07['Date'])
L_Date20_08 = list(df20_08['Date'])
L_Date20_09 = list(df20_09['Date'])
L_Date20_10 = list(df20_10['Date'])
L_Date20_11 = list(df20_11['Date'])

L_MW20_01 = list(df20_01['MW'])
L_MW20_02 = list(df20_02['MW'])
L_MW20_03 = list(df20_03['MW'])
L_MW20_04 = list(df20_04['MW'])
L_MW20_05 = list(df20_05['MW'])
L_MW20_06 = list(df20_06['MW'])
L_MW20_07 = list(df20_07['MW'])
L_MW20_08 = list(df20_08['MW'])
L_MW20_09 = list(df20_09['MW'])
L_MW20_10 = list(df20_10['MW'])
L_MW20_11 = list(df20_11['MW'])

L_MW_pred20_01 = list(df20_01['MW_pred'])
L_MW_pred20_02 = list(df20_02['MW_pred'])
L_MW_pred20_03 = list(df20_03['MW_pred'])
L_MW_pred20_04 = list(df20_04['MW_pred'])
L_MW_pred20_05 = list(df20_05['MW_pred'])
L_MW_pred20_06 = list(df20_06['MW_pred'])
L_MW_pred20_07 = list(df20_07['MW_pred'])
L_MW_pred20_08 = list(df20_08['MW_pred'])
L_MW_pred20_09 = list(df20_09['MW_pred'])
L_MW_pred20_10 = list(df20_10['MW_pred'])
L_MW_pred20_11 = list(df20_11['MW_pred'])

# Variables de Visualización de Barras de Error
ey_vis = True # Activa/Desactiva las barras de error
ey_pval = 9.2 # Porcentaje de Barras de Error



l_mse = []
l_rmse = []
y = [L_MW20_01, L_MW20_02, L_MW20_03, L_MW20_04, L_MW20_05, L_MW20_06, L_MW20_07, L_MW20_08, L_MW20_09, L_MW20_10, L_MW20_11]
y_pred = [L_MW_pred20_01, L_MW_pred20_02, L_MW_pred20_03, L_MW_pred20_04, L_MW_pred20_05, L_MW_pred20_06, L_MW_pred20_07, L_MW_pred20_08, L_MW_pred20_09, L_MW_pred20_10, L_MW_pred20_11]

for i in range (0,11):
    l_mse.append(mean_squared_error(y[i], y_pred[i]))
    l_rmse.append(np.sqrt(l_mse[i]))

mes = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre']

import plotly.graph_objects as go
fig7 = go.Figure([go.Bar(x = mes, y = l_rmse,
                        marker=dict(color=l_rmse, colorbar=dict(title="Potencia (MW)"),colorscale="jet"))])

fig7.update_layout({'plot_bgcolor': 'black','paper_bgcolor': 'black','font_color': 'white'})
fig7.update_layout(height=400, width=1000)
fig7.update_xaxes(showgrid=False)
#fig.update_yaxes(showgrid=False)
fig7.update_layout(xaxis_tickangle=0)
fig7.update_layout(title_text='RMSE de la Demanda Eléctrica 2020 (MW)', legend_title="Legend Title")
fig7.update_yaxes(title_text="Variación de Demanda Eléctrica (MW)", range=[0, 300])

l_hab = []
for r in l_rmse:
    l_hab.append(r*1000000/514)

fig8 = go.Figure([go.Bar(x = mes, y = l_hab,
                        marker=dict(color=l_hab, colorbar=dict(title="Habitantes x 1.000"),colorscale="jet"))])

fig8.update_layout({'plot_bgcolor': 'black','paper_bgcolor': 'black','font_color': 'white'})
fig8.update_layout(height=400, width=1000)
fig8.update_xaxes(showgrid=False)
#fig.update_yaxes(showgrid=False)
fig8.update_layout(xaxis_tickangle=0)
fig8.update_layout(title_text='Reducción Equivalente de Demanda Promedio de Habitantes', legend_title="miles de Habitante")
fig8.update_yaxes(title_text="Habitantes")
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
tabs_styles = {
    'height': '10px'
}
tab_style = {
    'borderBottom': '1px solid #ffffff',
    'padding': '4px',
    'fontWeight': 'bold',
    'color': 'black',
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '4px',
    'fontWeight': 'bold'
}

theme = {
    'dark': False,
    'detail': '#007439',
    'primary': '#00EA64',
    'secondary': '#6E6E6E'
}
#------------------------------------------------------------------------------
app.layout = html.Div([
    html.Br(),
    html.H2('Predicción de Demanda Electrica CABA e Impacto del COVID-19', style={'textAlign':'center'}),
    dcc.Tabs([
        dcc.Tab(label='Predicción de Demanda', style=tab_style, selected_style=tab_selected_style,
                children=[
                html.Br(),
                dbc.Row([
                    dbc.Col(html.H3('Demanda Predicha (MW)', className="text-center"))
                ]),
                dbc.Row([
                dbc.Col(daq.LEDDisplay(id='LED_display',
                            value=8888.88, size=70,
                            color="#119DFF",
                            backgroundColor="#000000", className="text-center")),
                ]),
                html.Br(),
                html.Br(),

                dbc.Row([
                    dbc.Col(),
                    dbc.Col(id='date_pick_val', style={'textAlign':'center'}),
                    dbc.Col(),
                    dbc.Col(dcc.DatePickerSingle(id='date_pick', date=date(2020, 1, 1), style={'textAlign':'center'})),
                    dbc.Col(daq.BooleanSwitch(id='sw_holiday', on=True, label = 'Feriado No Laborable', color = '#119DFF', style={'textAlign':'center'})),
                    dbc.Col(),
                    dbc.Col(),
                    dbc.Col(),
                ]),

                html.Br(),
                html.Br(),

                dbc.Row([
                    dbc.Col(html.H5('Temperatura Promedio (°C)'), style={'textAlign':'center'}),
                    dbc.Col(html.H5('Temperatura Máxima (°C)'), style={'textAlign':'center'}),
                    dbc.Col(html.H5('Temperatura Mínima (°C)'), style={'textAlign':'center'}),
                ]),

                dbc.Row([
                    dbc.Col(dcc.Slider(id='temp_avg',
                        min = round(min(L_Temp_avg))-5,
                        max = round(max(L_Temp_avg))+5,
                        step=0.1,
                        value=round(sum(L_Temp_avg)/len(L_Temp_avg)),
                        updatemode='drag',)
                        ),
                    dbc.Col(dcc.Slider(id='temp_max',
                        min = 10,
                        max = round(max(L_Temp_max))+5,
                        step=0.1,
                        value=round(max(L_Temp_max)),
                        updatemode='drag',)
                        ),
                    dbc.Col(dcc.Slider(id='temp_min',
                        min = round(min(L_Temp_min))-5,
                        max = 35,
                        step=0.1,
                        value=round(min(L_Temp_min)),
                        updatemode='drag',)
                        ),
                ]),

                dbc.Row([
                    dbc.Col(id='temp_avg_val', style={'textAlign':'center'}),
                    dbc.Col(id='temp_max_val', style={'textAlign':'center'}),
                    dbc.Col(id='temp_min_val', style={'textAlign':'center'}),
                ]),

                html.Br(),
                html.Br(),
                html.Br(),
                dbc.Row([
                    dbc.Col(html.H5('Velocidad de Viento Promedio (km/h)'), style={'textAlign':'center'}),
                    dbc.Col(html.H5('Humedad Relativa (%)'), style={'textAlign':'center'}),
                    dbc.Col(html.H5('Presión Atmosférica (hPa)'), style={'textAlign':'center'}),
                ]),

                dbc.Row([
                    dbc.Col(dcc.Slider(id='wind_avg',
                        min = 0,
                        max = round(max(L_Wind_avg)),
                        step=0.1,
                        value=round(sum(L_Wind_avg)/len(L_Wind_avg)),
                        updatemode='drag',)
                        ),
                    dbc.Col(dcc.Slider(id='hum',
                        min = 0,
                        max = 100,
                        step=0.1,
                        value=round(sum(L_Hum)/len(L_Hum)),
                        updatemode='drag',)
                        ),
                    dbc.Col(dcc.Slider(id='hpa',
                        min = round(min(L_hPa))-5,
                        max = round(max(L_hPa))+5,
                        step=0.1,
                        value=round(sum(L_hPa)/len(L_hPa)),
                        updatemode='drag',)
                        ),
                ]),

                dbc.Row([
                    dbc.Col(id='wind_avg_val', style={'textAlign':'center'}),
                    dbc.Col(id='hum_val', style={'textAlign':'center'}),
                    dbc.Col(id='hpa_val', style={'textAlign':'center'}),
                ]),
            ]
            ),

        dcc.Tab(label='Resumen', style=tab_style, selected_style=tab_selected_style,
                children=
                    [
                    html.Br(),
                    html.H4('Problema:'),
                    dcc.Markdown
                    ('''
                    La demanda eléctrica de la Ciudad de Buenos Aires está relacionada a factores
                    climáticos tales como la temperatura, humedad y velocidad del viento promedio
                    que se experimentan en la ciudad, durante las estaciones de Primavera y Verano
                    se incrementa la demanda a medida que se incrementa la temperatura promedio
                    y durante Otoño e Invierno disminuye a medida que disminuye la temperatura
                    promedio, este patrón de comportamiento permite planificar mantenimientos
                    preventivos durante las épocas de menor consumo con la finalidad de garantizar
                    una mayor confiabilidad y disponibilidad del suministro de energía eléctrica
                    durante los periodos de mayor consumo.

                    A partir del **20 de Marzo 2020**, se inicia una cuarentena obligatoria en
                    varias regiones de la Argentina, lo que obligó a varios grupos industriales
                    y comerciales de diferentes sectores a paralizar sus labores cotidianas y
                    por ende a reducir su consumo de energía eléctrica, por otro lado, hubo sectores
                    que incrementaron su consumo de energía eléctrica debido al incrememento de
                    sus actividades diarias, como por ejemplo el sector del área de la salud,
                    medicina o asistencia médica, otro grupo que incrementó su consumo promedio
                    fueron los consumidores residenciales, motivado a la necesidad de pasar mayor
                    tiempo en sus hogares debido a la cuarentena obligatoria o la implementación
                    del trabajo a distancia (Home Office), esta dicotomía genera las siguientes
                    interrogantes: **¿Como fué afectada la demanda de energía eléctrica de la
                    Ciudad de Buenos Aires durante el año 2020 debido a la cuarentena obligatoria?**
                    y **¿Cuanto se redujo o incrementó la demanda de energía eléctrica en la Ciudad
                    de Buenos Aires durante la cuarentena obligatoria?**.
                    '''),
                    html.Br(),
                    html.H4('Alcance:'),
                    dcc.Markdown
                    ('''
                    Este trabajo empleará **técnicas de Machine Learning para evaluar distintos
                    modelos de aprendisaje supervisado**, utilizando el lenguaje **Python**,
                    para la **creación de un modelo que permita predecir el comportamiento de
                    la demanda eléctrica de la Ciudad de Buenos Aires** en función de variables
                    metereológicas (Temperatura, Humedad, Presión, Velocidad de Viento) y variables
                    asociadas al calendario (Día de la Semana, Mes, Feriados No Laborables),
                    se utilizará dicho modelo para **cuantificar el impacto promedio mensual**
                    de la cuarentena obligatoria, producto de la pandemia mundial asociada al
                    COVID-19, en el consumo de potencia eléctrica diaria (MW) de la Ciudad Autónoma
                    de Buenos Aires.

                    _______________________________________________________________________________
                    **Trabajo Final del Programa de Ciencia de Datos con R y Python,
                    Escuela Argentina de Nuevas Tecnologías (EANT)**

                    Integrantes: **Carlos Martinez, Gustavo Prieto**

                    **Palabras Claves**: *Potencia Eléctrica, Predicción de Demanda, Predicción de Consumo, COVID-19, CABA, Argentina, Machine Learning, Python, Dash, Aprendisaje Supervisado*

                    15/01/2021
                    '''),
            ]),

        dcc.Tab(label='Variables',style=tab_style, selected_style=tab_selected_style,
                children=
                    [
                    html.Br(),
                    html.H6('En esta sección se muestran las graficas de todas las variables.'),
                    dcc.Graph(figure=fig,style={'width': '100%','padding-left':'3%', 'padding-right':'3%'}),
                    html.Br(),
                    html.H6('Observamos que la curva de demanda anual tiene tres (03) maximos:'),
                    html.H6('* A principios del mes de Enero y finales del mes de Diciembre debido al del Verano.'),
                    html.H6('* A mediados del mes de Julio cuando se alcanzan los mínimos de tempertura durante el Invierno.'),
                    html.Br(),
            ]),

        dcc.Tab(label='Outliers', style=tab_style, selected_style=tab_selected_style,
                children=
                    [
                    html.Br(),
                    html.H6('Sí solo vemos los datos del periodo 2017-2019 podremos identificar los Outliers de este conjunto de datos que usaremos para entrenar y probar los modelos predictivios. Recordemos que los valores del año 2020 están afectados por la cuarentenea obligatoria debido a la pandemia mundial de COVID-19.'),
                    html.Br(),
                    dcc.Graph(figure=fig1,style={'width': '100%','padding-left':'20%', 'padding-right':'25%'}),
                    html.Br(),
                    html.H6('Se identifican dos (02) outliers:'),
                    html.H6('* 16 de Junio 2019 (Dia de la falla que afectó a toda la Argentina y paises vecinos)'), dcc.Markdown('''[Falla Argentina, Uruguay y Paraguay](https://es.wikipedia.org/wiki/Apag%C3%B3n_el%C3%A9ctrico_de_Argentina,_Paraguay_y_Uruguay_de_2019)'''),
                    html.H6('* 25 de Diciembre 2019'),
                    html.H6('Se decidió eliminar todos aquellos puntos con una potencia inferior a 1.200 MW'),
                    html.Br(),
                    dcc.Graph(figure=fig2,style={'width': '100%','padding-left':'20%', 'padding-right':'25%'}),
                    html.Br(),
            ]),

        dcc.Tab(label='Modelos Evaluados', style=tab_style, selected_style=tab_selected_style,
                children=
                    [
                    html.Br(),
                    html.H6('En esta sección se comparan los resultados de los modelos de predicción evaluados.'),
                    html.Br(),
                    dcc.Graph(figure=fig3,style={'width': '100%','padding-left':'3%', 'padding-right':'3%'}),
                    dcc.Markdown ('''Nota: Se pueden encender/apagar los resultados haciendo click en la leyenda'''),
                    html.Br(),
                    html.H6('De la comparación gráfica de predicciones observamos que los modelos con mejores resultados son: ExtraTreeRegressor, RandomForestRegressor y VotingRegressor (conformado por ExtraTreeRegressor y RandomForestRegressor)'),
                    html.Br(),
                    html.H6('Cuantificar la precisión (observaciones predichas correctamente) usando solo la comparación grafica de los modelos, para seleccionar el mejor  modelo no es facil, en tal sentido se propone un métrica que cuantifique la precisión basada en bandas de tolerancia, es decir, cuantificar todas las predicciones que se encuentran dentro del rango del valor predicho +/- una tolerancia y así poder comparar el desempeño de cada modelo.'),
                    html.Br(),
                    dcc.Graph(figure=fig4,style={'width': '100%','padding-left':'3%', 'padding-right':'3%'}),
                    dcc.Markdown ('''Nota: Se pueden encender/apagar los resultados haciendo click en la leyenda'''),
                    html.H6('Los modelos que alcanzan un 99% de precisión con menos del 10% de tolerancia son: ExtraTreeRegressor (9,2%), RandomForestRegressor (9,6%) y VotingRegressor (9,8%) (conformado por ExtraTreeRegressor y RandomForestRegressor'),
                    html.Br(),
                    html.H6('Comparando los valores de RMSE obtenidos por Cross-Validation para todos lo modelos'),
                    dcc.Graph(figure=fig5,style={'width': '100%','padding-left':'20%', 'padding-right':'25%'}),
                    html.Br(),
                    html.H6('Los modelos que tienen menor RMSE (%) son: VotingRegressor (4,55%), RandomForestRegressor (4,65%) y ExtraTreeRegressor (4,67%)'),
                    html.Br(),
                    html.H6('De los resultados presentados, se selecciona el ExtraTreeRegressor como mejor modelo para predicir la demanda en este caso de estudio.'),
                    html.Br(),
            ]),

        dcc.Tab(label='COVID-19', style=tab_style, selected_style=tab_selected_style,
                children=
                    [
                    html.Br(),
                    html.H6('En esta sección se muestra el impacto de la cuarentena obligatoria a partir del 20 de Marzo 2020 sobre la demanda de potencia eléctrica en la Ciudad de Buenos Aires'),
                    #dcc.Graph(figure=fig6,style={'width': '100%','padding-left':'1%', 'padding-right':'1%'}),
                    daq.BooleanSwitch(id='sw_err', on=True, label = 'Barras de Error', color = '#119DFF', style={'padding-left':'1%', 'padding-right':'90%'}),
                    dbc.Row([
                        dbc.Col(
                            dcc.Slider(id='err_sld', min=0, max=10, step=0.2, value=9.2,
                                marks={0: {'label': '0%'}, 1: {'label': '1%'}, 2: {'label': '2%'},
                                    3: {'label': '3%'}, 4: {'label': '4%'}, 5: {'label': '5%'},
                                    6: {'label': '6%'}, 7: {'label': '7%'}, 8: {'label': '8%'},
                                    9: {'label': '9%'}, 10: {'label': '10%'}})
                            ),
                        dbc.Col(),
                        ]),
                    html.Div(id='sld_err_val', style={'width': '100%','padding-left':'1%', 'padding-right':'1%'}),
                    dcc.Graph(id='graph6', style={'width': '100%','padding-left':'1%', 'padding-right':'1%'}),
                    html.Br(),
                    html.H6('A partir del 20 de Marzo se empieza a notar el impacto de la cuarentena en la demanda de potencia eléctrica, es el primer punto donde la demanda real es inferior al valor mínimo de la banda inferior de tolerancia de la demanda predicha.'),
                    html.Br(),
                    dcc.Graph(figure=fig7,style={'width': '100%','padding-left':'18%', 'padding-right':'25%'}),
                    html.Br(),
                    html.H6('Para tener un orden de magnitud de cuan grande fué el impacto en la demanda se recurrió al informe de resultados del Consumo de Energía en la Ciudad de Buenos Aires 2013, Informe N° 663, publicado en marzo de 2014'),
                    dcc.Markdown('''[Consumo de Energía en la Ciudad de Buenos Aires 2013, Informe N° 663](https://www.estadisticaciudad.gob.ar/eyc/wp-content/uploads/2015/04/ir_2014_663.pdf)'''),
                    html.H6('En este informe se indica que el consumo promedio de energía eléctrica por habitante en CABA es de 4.500 kWh lo que equivale a un consumo promedio continuo por habitante de 514 W, en tal sentido graficamos la cantidad de habitantes promedios que representa la variación de la demanda'),
                    html.Br(),
                    dcc.Graph(figure=fig8,style={'width': '100%','padding-left':'18%', 'padding-right':'25%'}),
                    html.Br(),
                    html.H6('El impacto respecto a la cantidad de personas es equivalente a una reducción de 552 mil habitantes en promedio de consumo de energía eléctrica, lo que equivaldría al 18% de la población de acuerdo al censo de 2010 (3.075.646 hab).'),
                    html.Br(),
                    html.H6('Se nota que la recuperación de la demanda de consumo de energía eléctrica a partir del mes Mayo 2020 coincide con las primeras medidas de flexibilización de la cuarentenea'),
                    html.Br(),
                    html.Br(),
            ]),

        dcc.Tab(label='Tab 7', style=tab_style, selected_style=tab_selected_style,
                children=
                    [
                    html.H3('Tab content 7'),
            ]),

        ]),
    ])


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
@app.callback(
    [Output('temp_avg_val', 'children'),
    Output(component_id='temp_max', component_property='min'),
    Output(component_id='temp_min', component_property='max'),
    Output('temp_max_val', 'children'),
    Output('temp_min_val', 'children'),
    Output('wind_avg_val', 'children'),
    Output('hum_val', 'children'),
    Output('hpa_val', 'children'),
    Output(component_id='LED_display', component_property='value')],
    [dash.dependencies.Input('temp_avg', 'value'),
    dash.dependencies.Input('temp_max', 'value'),
    dash.dependencies.Input('temp_min', 'value'),
    dash.dependencies.Input('wind_avg', 'value'),
    dash.dependencies.Input('hum', 'value'),
    dash.dependencies.Input('hpa', 'value'),
    dash.dependencies.Input('sw_holiday', 'on'),
    Input('date_pick', 'date')])
def update_predict(temp_avg, temp_max, temp_min, wind_avg, hum, hpa, sw_holiday, date_pick):
    temp_max_min = temp_avg + 1
    temp_min_max = temp_avg - 1
    if sw_holiday == True:
        val_pred[0][0] = 1
    else:
        val_pred[0][0] = 0
    val_pred[0][1] = temp_avg
    val_pred[0][2] = temp_max
    val_pred[0][3] = temp_min
    val_pred[0][4] = hpa
    val_pred[0][5] = hum
    val_pred[0][6] = wind_avg
    date_object = date.fromisoformat(date_pick)
    date_p = date.weekday(date_object)
    if date_p == 5:
        val_pred[0][7] = 1
    else:
        val_pred[0][7] = 0
    if date_p == 6:
        val_pred[0][8] = 1
    else:
        val_pred[0][8] = 0
    y_pred = model_etr_simp.predict(val_pred)
    led_val = round(y_pred[0],2)
    return '{} °C '.format(temp_avg), temp_max_min, temp_min_max, '{} °C '.format(temp_max), '{} °C '.format(temp_min), '{} km/h '.format(wind_avg), '{}% '.format(hum), '{} hPa '.format(hpa), led_val
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
@app.callback(
    Output('graph6', 'figure'),
    Output('sld_err_val', 'children'),
    [dash.dependencies.Input('sw_err', 'on')],
    [dash.dependencies.Input('err_sld', 'value')])
def update_bars(swerr, err_sld):
    if swerr == True:
        ey_vis = True
    else:
        ey_vis = False
    ey_pval = err_sld
    fig6 = make_subplots(rows=4, cols=3,
                        vertical_spacing=0.03)
    fig6.add_trace(go.Scatter(name='MW_Ene_20', x=L_Date20_01, y=L_MW20_01, line=dict(color='blue', width=1.8)),
                  row=1, col=1)
    fig6.add_trace(go.Scatter(name='MW_pred_Ene_20', x=L_Date20_01, y=L_MW_pred20_01,
                             line=dict(color='red', width=1.5, dash='dot'),
                             error_y = dict(type='percent', value = ey_pval, thickness=1, width=2, visible=ey_vis)),
                  row=1, col=1)
    fig6.add_trace(go.Scatter(name='MW_Feb_20', x=L_Date20_02, y=L_MW20_02, line=dict(color='blue', width=1.8)),
                  row=1, col=2)
    fig6.add_trace(go.Scatter(name='MW_pred_Feb_20', x=L_Date20_02, y=L_MW_pred20_02,
                             line=dict(color='red', width=1.5, dash='dot'),
                             error_y=dict(type='percent', value=ey_pval, thickness=1, width=2, visible=ey_vis)),
                  row=1, col=2)

    fig6.add_trace(go.Scatter(name='MW_Mar_20', x=L_Date20_03, y=L_MW20_03, line=dict(color='blue', width=1.8)),
                  row=1, col=3)
    fig6.add_trace(go.Scatter(name='MW_pred_Mar_20', x=L_Date20_03, y=L_MW_pred20_03,
                             line=dict(color='red', width=1.5, dash='dot'),
                             error_y=dict(type='percent', value=ey_pval, thickness=1, width=2, visible=ey_vis)),
                  row=1, col=3)

    fig6.add_trace(go.Scatter(name='MW_Abr_20', x=L_Date20_04, y=L_MW20_04, line=dict(color='blue', width=1.8)),
                  row=2, col=1)
    fig6.add_trace(go.Scatter(name='MW_pred_Abr_20', x=L_Date20_04, y=L_MW_pred20_04,
                             line=dict(color='red', width=1.5, dash='dot'),
                             error_y=dict(type='percent', value=ey_pval, thickness=1, width=2, visible=ey_vis)),
                  row=2, col=1)

    fig6.add_trace(go.Scatter(name='MW_May_20', x=L_Date20_05, y=L_MW20_05, line=dict(color='blue', width=1.8)),
                  row=2, col=2)
    fig6.add_trace(go.Scatter(name='MW_pred_May_20', x=L_Date20_05, y=L_MW_pred20_05,
                             line=dict(color='red', width=1.5, dash='dot'),
                             error_y=dict(type='percent', value=ey_pval, thickness=1, width=2, visible=ey_vis)),
                  row=2, col=2)

    fig6.add_trace(go.Scatter(name='MW_Jun_20', x=L_Date20_06, y=L_MW20_06, line=dict(color='blue', width=1.8)),
                  row=2, col=3)
    fig6.add_trace(go.Scatter(name='MW_pred_Jun_20', x=L_Date20_06, y=L_MW_pred20_06,
                             line=dict(color='red', width=1.5, dash='dot'),
                             error_y=dict(type='percent', value=ey_pval, thickness=1, width=2, visible=ey_vis)),
                  row=2, col=3)

    fig6.add_trace(go.Scatter(name='MW_Jul_20', x=L_Date20_07, y=L_MW20_07, line=dict(color='blue', width=1.8)),
                  row=3, col=1)
    fig6.add_trace(go.Scatter(name='MW_pred_Jul_20', x=L_Date20_07, y=L_MW_pred20_07,
                             line=dict(color='red', width=1.5, dash='dot'),
                             error_y=dict(type='percent', value=ey_pval, thickness=1, width=2, visible=ey_vis)),
                  row=3, col=1)

    fig6.add_trace(go.Scatter(name='MW_Ago_20', x=L_Date20_08, y=L_MW20_08, line=dict(color='blue', width=1.8)),
                  row=3, col=2)
    fig6.add_trace(go.Scatter(name='MW_pred_Ago_20', x=L_Date20_08, y=L_MW_pred20_08,
                             line=dict(color='red', width=1.5, dash='dot'),
                             error_y=dict(type='percent', value=ey_pval, thickness=1, width=2, visible=ey_vis)),
                  row=3, col=2)

    fig6.add_trace(go.Scatter(name='MW_Sep_20', x=L_Date20_09, y=L_MW20_09, line=dict(color='blue', width=1.8)),
                  row=3, col=3)
    fig6.add_trace(go.Scatter(name='MW_pred_Sep_20', x=L_Date20_09, y=L_MW_pred20_09,
                             line=dict(color='red', width=1.5, dash='dot'),
                             error_y=dict(type='percent', value=ey_pval, thickness=1, width=2, visible=ey_vis)),
                  row=3, col=3)

    fig6.add_trace(go.Scatter(name='MW_Oct_20', x=L_Date20_10, y=L_MW20_10, line=dict(color='blue', width=1.8)),
                  row=4, col=1)
    fig6.add_trace(go.Scatter(name='MW_pred_Otc_20', x=L_Date20_10, y=L_MW_pred20_10,
                             line=dict(color='red', width=1.5, dash='dot'),
                             error_y=dict(type='percent', value=ey_pval, thickness=1, width=2, visible=ey_vis)),
                  row=4, col=1)

    fig6.add_trace(go.Scatter(name='MW_Nov_20', x=L_Date20_11, y=L_MW20_11, line=dict(color='blue', width=1.8)),
                  row=4, col=2)
    fig6.add_trace(go.Scatter(name='MW_pred_Nov_20', x=L_Date20_11, y=L_MW_pred20_11,
                             line=dict(color='red', width=1.5, dash='dot'),
                             error_y=dict(type='percent', value=ey_pval, thickness=1, width=2, visible=ey_vis)),
                  row=4, col=2)

    fig6.update_yaxes(title_text="Demanda (MW)",row=1, col=1)
    fig6.update_yaxes(title_text="Demanda (MW)", row=2, col=1)
    fig6.update_yaxes(title_text="Demanda (MW)", row=3, col=1)
    fig6.update_yaxes(title_text="Demanda (MW)", row=4, col=1)
    fig6.update_yaxes(title_text="Demanda (MW)", row=5, col=1)
    fig6.update_yaxes(title_text="Demanda (MW)", row=6, col=1)

    fig6.update_layout(height=1200, width=1540,
                      title_text="Demanda Real Vs Demanda Predicha 2020")
    fig6.update_layout({'plot_bgcolor': 'black','paper_bgcolor': 'whitesmoke',}, margin=dict(l=30, r=30, t=130, b=20))
    fig6.update_xaxes(showgrid=False)
    fig6.update_yaxes(showgrid=False)
    fig6.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1))
    return fig6, 'Barras de Error: {}% '.format(err_sld)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
