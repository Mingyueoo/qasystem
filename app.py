from flask_bootstrap import Bootstrap5
from flask import Flask, render_template, redirect, url_for, request, flash
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm, CSRFProtect
from wtforms import StringField, SubmitField, TextAreaField, HiddenField
from wtforms.validators import DataRequired, Length, Optional
import os
from functions import load_model, run_code_generator

# Create the Flask app
app = Flask(__name__)
app.secret_key = 'tO$&!|0wkamvVia0?n$NqIRVWOG'

# Database configuration
db_name = 'answers.db'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, db_name)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_path
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
bootstrap = Bootstrap5(app)
csrf = CSRFProtect(app)


# Global variables to hold the model and tokenizer
tokenizer = None
model = None
# tokenizer, model = load_model()




# Define the AnswerModel
class AnswerModel(db.Model):
    __tablename__ = 'PROMPTCODE'
    id = db.Column(db.Integer, primary_key=True)
    prompt = db.Column(db.String(500), nullable=False)
    code = db.Column(db.Text, nullable=False)

    def __init__(self, prompt, code):
        self.prompt = prompt
        self.code = code


# Define the forms
class PromptForm(FlaskForm):
    prompt = StringField('Input prompt', validators=[DataRequired(), Length(10, 500)])
    submit = SubmitField('Submit')


class AnswerForm(FlaskForm):
    prompt = StringField('Input prompt', validators=[DataRequired(), Length(10, 500)])
    answer = TextAreaField('The answer', validators=[Optional(), Length(max=1024)])


class AddRecord(FlaskForm):
    id_field = HiddenField()
    prompt = StringField('Input prompt')
    code = TextAreaField('The answer')
    submit = SubmitField('Submit')


# Ensure the tables are created
with app.app_context():
    db.create_all()
    print(f"Tables created or confirmed in database at {db_path}")



# Routes
@app.route('/', methods=['GET', 'POST'])
def index():

    global tokenizer, model
    # if tokenizer is None or model is None:
    #     return "Error loading model or tokenizer1.", 500
    # Load the tokenizer and model only once
    if tokenizer is None or model is None:
        tokenizer, model = load_model()  # Assuming load_model() returns (tokenizer, model)

    form = PromptForm()
    if form.validate_on_submit():
        prompt = form.prompt.data
        return redirect(url_for('results', prompt=prompt))
    return render_template("index.html", form=form)


@app.route("/results", methods=['GET', 'POST'])
def results():
    global tokenizer, model
    if tokenizer is None or model is None:
        return "Error loading model or tokenizer2.", 500
    if request.method == 'POST':
        prompt = request.form.get('prompt')
        if prompt:
            code = run_code_generator(prompt, tokenizer, model)
            form = AnswerForm(prompt=prompt, answer=code)
            # Create a new record and add it to the database
            record = AnswerModel(prompt=prompt, code=code)
            try:
                db.session.add(record)
                db.session.commit()
                print(f"Record added: {record.prompt}, {record.code}")
            except Exception as e:
                print(f"Error adding record: {e}")
                db.session.rollback()
            return render_template("result.html", form=form, prompt=prompt, answer=code)
    return redirect(url_for('index'))


@app.route('/read_record')
def read_record():
    records = db.session.execute(db.select(AnswerModel).order_by(AnswerModel.id)).scalars().all()
    return render_template('list.html', records=records)


if __name__ == '__main__':
    app.run(debug=True)
