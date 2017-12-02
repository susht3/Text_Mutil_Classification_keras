# coding:utf-8
import json
import re

tree = {}
tiao_tag = False
tiao_con = []
tiao_name = None
kuan_con = []
kuan_nb = 0
xiang_nb = 0
s = u'一二三四五六七八九十'

with open('xinyongka.txt') as f:
	for i in f:
		i = i.decode('utf-8')
		if len(i) == 0 or i == '\r\n':
			continue
		if re.search(ur'第*[条章节](之一)?[　 ]', i) is not None:
			tiao_tag = False
			if len(tiao_con) > 0:
				tree[tiao_name] = ''.join(tiao_con)
			if len(kuan_con) > 0:
				kuan_name = tiao_name + u'第%s款'%s[kuan_nb]
				tree[kuan_name] = ''.join(kuan_con)
		if re.search(ur'第*条(之一)?[　 ]', i) is not None:
			tiao_tag = True
			tiao_name = u'《最高人民法院、最高人民检察院关于办理妨害信用卡管理刑事案件具体应用法律若干问题的解释》' + filter(lambda x: len(x)>0, i.split(u' '))[0].replace(u'　','')
			tiao_con = []
			kuan_nb = 0
			kuan_con = []
			xiang_nb = 0
		if tiao_tag == True:
			tiao_con.append(i)
			if re.search(ur'　　\([一二三四五六七八九十]\)', i) is not None or re.search(ur'第*条(之一)?[　 ]', i) is not None:
				kuan_con.append(i)
				if re.search(ur'　　\([一二三四五六七八九十]\)', i) is not None:
					xiang_name = tiao_name + u'第%s款'%s[kuan_nb] + u'第（%s）项'%s[xiang_nb]
					tree[xiang_name] = i
					xiang_nb += 1
			else:
				kuan_name = tiao_name + u'第%s款'%s[kuan_nb]
				tree[kuan_name] = ''.join(kuan_con)
				kuan_nb += 1
				kuan_con = [i]
				xiang_nb = 0
		# print i.encode('utf-8')
		# print kuan_nb
		# print xiang_nb

	if len(tiao_con) > 0:
		tree[tiao_name] = ''.join(tiao_con)



# print len(tree)

# with open('xinyongka', 'w') as f:
# 	f.write(json.dumps(tree))

tree = json.loads(open('xinyongka').readline())
# for i in sorted(tree):
# 	print i
# 	print tree[i]
# print len(tem)
print tree[u'《最高人民法院、最高人民检察院关于办理妨害信用卡管理刑事案件具体应用法律若干问题的解释》第五条']
