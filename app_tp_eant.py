import dash
import dash_html_components as html
import dash_core_components as dcc
from datetime import date

import pandas as pd

from dash.dependencies import Input, Output
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# 2.- Datos
df_demanda = pd.read_csv('APP_VAR_MW_CABA_2017_2020.csv')
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
fig1.update_layout({'plot_bgcolor': 'black','paper_bgcolor': 'whitesmoke',})
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
fig2.update_layout({'plot_bgcolor': 'black','paper_bgcolor': 'whitesmoke',})
fig2.update_xaxes(showgrid=False)
fig2.update_yaxes(range=[1200,2400], tick0=200, dtick=200)
fig2.update_layout(height=500, width=1000)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

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
    'borderBottom': '1px solid #d6d6d6',
    'padding': '2px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '2px'
}
#------------------------------------------------------------------------------
app.layout = html.Div([
html.H3('Predicción de Demanda Electrica CABA e Impacto del COVID-19', style={'textAlign':'center'}),
        dcc.Tabs(id="tabs", value='tab-1',
        children=[
        dcc.Tab(label='Introducción', value='tab-1', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Datos', value='tab-2', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Outliers', value='tab-3', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Tab 4', value='tab-4', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Tab 5', value='tab-5', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Tab 6', value='tab-6', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Tab 7', value='tab-7', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Tab 8', value='tab-8', style=tab_style, selected_style=tab_selected_style),
    ]),
    html.Div(id='tabs-content')
])
#------------------------------------------------------------------------------
@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
        html.H6('Problema:'),
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
        html.H6('Alcance:'),
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
        html.H1(''),
        html.H1(''),
        html.H1(''),
        html.H1(''),
        html.H1(''),
        ])

    elif tab == 'tab-2':
        return html.Div([
    dcc.Graph(figure=fig,style={'width': '100%','padding-left':'3%', 'padding-right':'3%'}),
    html.H6('Observamos que la curva de demanda anual tiene tres (03) maximos:'),
    html.H6('* A principios del mes de Enero y finales del mes de Diciembre debido al del Verano.'),
    html.H6('* A mediados del mes de Julio cuando se alcanzan los mínimos de tempertura durante el Invierno.'),
        ])

    elif tab == 'tab-3':
        return html.Div([
            html.H6('Sí solo vemos los datos del periodo 2017-2019 podremos identificar los Outliers de este conjunto de datos que usaremos para entrenar y probar los modelos predictivios. Recordemos que los valores del año 2020 están afectados por la cuarentenea obligatoria debido a la pandemia mundial de COVID-19.'),
            dcc.Graph(figure=fig1,style={'width': '100%','padding-left':'15%', 'padding-right':'25%'}),
            html.H6('Se identifican dos (02) outliers:'),
            html.H6('* 16 de Junio 2019 (Dia de la falla que afectó a toda la Argentina y paises vecinos)'), dcc.Markdown('''[Falla Argentina, Uruguay y Paraguay](https://es.wikipedia.org/wiki/Apag%C3%B3n_el%C3%A9ctrico_de_Argentina,_Paraguay_y_Uruguay_de_2019)'''),
            html.H6('* 25 de Diciembre 2019'),
            html.H6('Se decidió eliminar todos aquellos puntos con una potencia inferior a 1.200 MW'),
            dcc.Graph(figure=fig2,style={'width': '100%','padding-left':'15%', 'padding-right':'25%'}),

        ])

    elif tab == 'tab-4':
        return html.Div([
            html.H3('Tab content 4'),

        ])

    elif tab == 'tab-5':
        return html.Div([
            html.H3('Tab content 5'),

        ])

    elif tab == 'tab-6':
        return html.Div([
            html.H3('Tab content 6'),

        ])

    elif tab == 'tab-7':
        return html.Div([
            html.H3('Tab content 7'),

        ])

    elif tab == 'tab-8':
        return html.Div([
            html.H3('Tab content 8'),
            dcc.DatePickerSingle(
    id='date-picker-single',
    date=date(2020, 1, 1)),
        ])
if __name__ == '__main__':
    app.run_server(debug=True)
