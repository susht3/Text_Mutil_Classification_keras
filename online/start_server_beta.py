# coding: utf-8
import sys
sys.path.append('../')
import random
import requests
import json
from flask import Flask, request, render_template, make_response, abort, redirect
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_moment import Moment
from wtforms.validators import Required
from wtforms import SelectField, TextAreaField, SubmitField
from models import gru_9w
from util import get_law_cons, random_sample
from Lawclf import pred


app = Flask(__name__)
law_cons_dict = get_law_cons()
samples = random_sample()

clf_9w = gru_9w()

clf_versions = [clf_9w]
#clf_versions = [[clf_hierarchical_6w]
#	[clf_m1_30, clf_m0_30, clf_m2_30, clf_m3_30]]

bootstrap = Bootstrap(app)
moment = Moment(app)

app.config['SECRET_KEY'] = 'hard to guess string'

class NnSelected(FlaskForm):
	model_selected = SelectField(u'模型选择', coerce=int, choices=[])
	input_text = TextAreaField(u'', validators=[Required()], render_kw={"rows": 5})
	submit=SubmitField(u'搜索')


@app.route('/', methods=['GET', 'POST'])
def home():
	form = NnSelected()
	model_names = ['gru_9w']
	form.model_selected.choices = [(idx, i) for idx,i in enumerate(model_names)]
	if form.validate_on_submit():
		model_selected = form.model_selected.data
		input_text = form.input_text.data
			
		clfs = clf_versions
		# clfs = []
		combine_res = pred(clfs[model_selected], input_text)
		for idx, res in enumerate(combine_res):
			law_con = [law_cons_dict.get(i[0], u'抱歉，没有找到法条内容') for i in res[1]]
			#if len(res[1]) > 0 and type(res[1][0][1]) is unicode:
			#	law_name = map(lambda x: '%s %s' % (x[0],x[1][:5]), res[1])
			#else:
			law_name = list(map(lambda x: '%s %.3lf' % (x[0],x[1]), res[1]))
			combine_res[idx] = (res[0], list(zip(law_name, law_con)))
		return render_template('main.html', form=form, combine_res=combine_res)
	else:
		return render_template('main.html', form=form)
		

@app.route('/random')
def random_text():
	return random.choice(samples)


from werkzeug.contrib.fixers import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app)
if __name__ == '__main__':
	app.run(debug=0, host='0.0.0.0', port=1024)
