# coding: utf-8
import sys
import random
sys.path.append('../')
from flask import Flask, request, render_template, make_response
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_moment import Moment
from wtforms.validators import Required
from wtforms import SelectField, TextAreaField, SubmitField
from models import hierarchical_30w,gru_30w
from util import get_law_cons, random_sample
from Lawclf import pred


app = Flask(__name__)
law_cons_dict = get_law_cons()
samples = random_sample()
clf = gru_30w()

bootstrap = Bootstrap(app)
moment = Moment(app)
app.config['SECRET_KEY'] = 'hard to guess string'

class NnSelected(FlaskForm):
    input_text = TextAreaField(u'',validators=[Required()],render_kw={"rows": 5})
    submit=SubmitField(u'搜索')


@app.route('/', methods=['GET', 'POST'])
def index():
	form = NnSelected()
	if form.validate_on_submit():
		input_text = form.input_text.data
		res = pred(clf, input_text)[-1][1]
		for i in res:
			print(res)
		laws = [law_cons_dict.get(i[0], u'抱歉，没有找到法条内容') for i in res]
		print(len(res))
		res = list(map(lambda x: '%s %.3lf' % (x[0],x[1]), res))
		return render_template('index.html', form=form, res=list(zip(res, laws)))
	else:
		form.input_text.data = random.choice(samples)
		return render_template('index.html', form=form)

		
@app.route('/random')
def random_text():
	return random.choice(samples)


from werkzeug.contrib.fixers import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app)
if __name__ == '__main__':
	app.run(debug=0, host='0.0.0.0', port=1024)
