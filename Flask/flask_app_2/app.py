"""The app asks for a name and says hello"""
from flask import request, render_template, Flask
from wtforms import Form, TextAreaField, validators

app = Flask(__name__)


class HelloForm(Form):
    say_hello = TextAreaField('', [validators.DataRequired()])


@app.route('/')
def index():
    form = HelloForm(request.form)
    return render_template('first_app.html', form=form)


@app.route('/hello', methods=['POST'])
def hello():
    form = HelloForm(request.form)
    if request.method == 'POST' and form.validate():
        name = request.form['say_hello']
        return render_template('hello.html', name=name)
    return render_template('first_app.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)
