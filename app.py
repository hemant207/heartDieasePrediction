import pickle
import numpy as np
import pandas as pd
from flask import Flask, Response,request,redirect,app,jsonify,url_for,render_template
from flask_sqlalchemy import SQLAlchemy
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField,PasswordField,BooleanField
from wtforms.validators import InputRequired,Email,Length


#retriving model from the pickel file
hear_model = pickle.load(open("random_forest.pkl","rb"))

#creating flask app name app
app = Flask(__name__)
Bootstrap(app)


#congigration of the database with the app and make connection with database
app.config['SECRET_KEY'] = 'thisisit'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///G:/project_github/heartDieasePrediction/database.db'
db = SQLAlchemy(app)
app.app_context().push()

#craeting function for the diferent type of forms login(),signup()

class LoginForm(FlaskForm):
    email = StringField(u'email', validators=[InputRequired(),Email(message='invalid email'),Length(max=30)])
   # username = StringField(u'UserName', validators=[validators.required(), validators.length(max=10)])
    password = PasswordField(u'UserName',validators=[InputRequired(),Length(min=6)])

class SignupForm(FlaskForm):
    email = StringField(u'email', validators=[InputRequired(),Email(message='invalid email'),Length(max=30)])
    username = StringField(u'UserName', validators=[InputRequired(),Length(max=10)])
    password = PasswordField(u'UserName', validators=[InputRequired(), Length(min=6)])



#creating model for the user to store in database
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True) # primary keys are required by SQLAlchemy
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    username = db.Column(db.String(1000))
   
db1 = {"abc@gmail.com":"123"}

#creating routes for the website.
@app.route('/')
def index():
    return render_template('abc.html')

@app.route('/dash')
def dash():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():

    req = request.get_json()
    print("hi from abc")
    d = req['data']
    print(d)

    return jsonify(d)

@app.route('/predict',methods=["POST"])
def predict():
    data = [[x for x in request.form.values()]]
    output = hear_model.predict(data)[0]

    return render_template("home.html",pred_text = "you are heart diease chances is {} ".format(output))

@app.route('/login',methods=['Post','Get'])
def login():
    form1 = LoginForm()

    if form1.validate_on_submit():
        user=User.query.filter_by(email=form1.email.data).first()
        if user:
            if user.password == form1.password.data:
                return redirect(url_for("dash"))
            return redirect(url_for("login"))

    return render_template("login.html", form = form1)
    
@app.route('/signup',methods=['Post','Get'])
def signup():
    
    form = SignupForm()
    if form.validate_on_submit():
        new_user = User(username=form.username.data,email=form.email.data,password=form.password.data)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for("login"))
    return render_template('sign_up.html',form=form)
    


    

if __name__=="__main__":
    app.run(debug=True)