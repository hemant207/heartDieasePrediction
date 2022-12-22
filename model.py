import sqlalchemy from 

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True) # primary keys are required by SQLAlchemy
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    f_name = db.Column(db.String(1000))
    l_name = db.Column(db.String(1000))