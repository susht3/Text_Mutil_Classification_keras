import word2vec
import json

def count():
	f = open('../train_data/economic_fraud_train').readlines()
	f2 = open('40.txt','w')
	n1 = 0
	n2 = 0
	for line in f:
		text = json.loads(line)['input_text']
		seq = text.split('，')
		s = len(seq)
		if(s<=40):
			n1 += s
		else:
			n2 += s

		n = ''
		for i in seq:
			n += str(len(i)) + ' '
		#print(str(s)+ ' seqs: '+n)
		f2.write(str(s)+' seqs: '+n + '\n')

	print('n1: '+str(n1))
	print('n2: '+str(n2))
	


def main():
	word_path = '../dict/words_6w.txt'
	#phrase_path = '../dict/phrases_6w.txt'
	bin_path = '../dict/vector_6w.bin'
	#word2vec.word2phrase(word_path, phrase_path, verbose=True)
	word2vec.word2vec(word_path, bin_path, cbow=1,size=128,min_count=1,verbose=True)

	model = word2vec.load(bin_path)
	words = list(model.vocab)
	for i in range(10):
		print(words[i])
	print(model.vectors.shape)
	#print(model.vectors)
	#print(model['本'][:20])
	'''
	min_count = 0
	word2vec_model_path = '../dict/word2vec.txt'
	sentences = [['你', '好'], ['喂', '哦'，'哈']]
	model = gensim.models.Word2Vec(text, min_count)
	model.save(word2vec_model_path)
	'''

if __name__ == '__main__':
        #main()
	count()
