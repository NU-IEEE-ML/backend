import flask
import xgboost
import pandas as pd

bst = xgboost.Booster({'nthread' : 4})
bst.load_model('airbnb_model2.bin')
    
app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        bedrooms = flask.request.form['bedrooms']
        accom = flask.request.form['accom']
        gym = flask.request.form['gym']

        input_variables = pd.DataFrame([[bedrooms, accom, gym]],
                                       columns=['bedrooms', 'accommodations', 'gym'],
                                       dtype=float)
        d_input = xgboost.DMatrix(input_variables)
        prediction = bst.predict(d_input)[0]
        return flask.render_template('main.html', original_input={'Bedrooms':bedrooms,
                                                                  'Accomadates':accom,
                                                                  'Gym':gym},
                                     result=prediction,)
    

if __name__ == '__main__':
    app.run()
