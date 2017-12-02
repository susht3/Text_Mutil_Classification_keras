from models import m1_30w
from Lawclf import pred
import json


if __name__ == '__main__':
	clf = m1_30w()
	with open('../train_data/credit_card_fraud_show', 'w') as writer:
		with open('../train_data/credit_card_fraud_train') as f:
			cnt = 0
			for i in f:
				uni_i = i.decode('utf-8')
				input_text = json.loads(uni_i)['input_text']
				res = pred(clf, input_text)[-1][1]
				if len(res) > 0:
					writer.write(i)
					cnt += 1
					# print cnt, len(res)
				else:
					print input_text