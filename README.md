# Chatbot-Project-Using-Dash-App
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
!pip install -q dash dash-core-components dash-html-components dash-table
import dash
from dash import dcc,html
from dash.dependencies import Input,Output
import random


data=pd.read_csv('chatbot_dataset.csv')

nltk.download('punkt')
data['Question']=data['Question'].apply(lambda x:' '.join(nltk.word_tokenize(x.lower())))

x_train,x_test,y_train,y_test=train_test_split(data['Question'],data['Answer'],test_size=0.2,random_state=42)
model=make_pipeline(TfidfVectorizer(),MultinomialNB())
model.fit(x_train,y_train)

def get_response(question):
  question=' '.join(nltk.word_tokenize(question.lower()))
  answer=model.predict([question])
  return answer


app=dash.Dash(__name__)

app.layout=html.Div([
      html.H1("Chatbot",style={'textAlign':'center'}),
      dcc.Textarea(
          id='user-input',
          value='Type your question here...',
          style={'width':'100%','height':'100px'}
    ),
      html.Button('Submit',id='submit-button',n_clicks=0),
      html.Div(id='chatbot-output',style={'padding':'10px'})

])


@app.callback(
    Output('chatbot-output','children'),
    Input('submit-button','n_clicks'),
    [dash.dependencies.State('user-input','value')]
)

def update_output(n_clicks,user_input):
  if n_clicks>0:
    response=get_response(user_input)
    return html.Div([
        html.P(f"you: {user_input}",style={'mqrgin': '10px'}),
        html.P(f"Bot: {response}",style={'margin':'10px','backgroundColor':'#f0f0f0','padding':'10px'})
    ])
  return "Ask me something"

if __name__=='__main__':
  app.run_server(debug=True)




