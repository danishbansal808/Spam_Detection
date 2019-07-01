import sys
import nltk
nltk.download
import pandas as pd
import string
import tkinter as tk

from nltk.corpus import stopwords
stop_words=stopwords.words('english')
li=['not','aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",'against','off','no','nor','too','don',"don't"]
for i in li:
    stop_words.remove(i)
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB,MultinomialNB

ps=PorterStemmer()
cv=CountVectorizer()

df = pd.read_csv('dataset/sms.txt',sep='\t',names=['label','messages'] )

def clean_text(msg):
    '''
    1:remove punctuation
    2:remove stopwords
    3:steming
    '''
    new_msg=[w for w in msg if w not in string.punctuation]
    new_msg2=''.join(new_msg)
    tmp_list=[]
    for ww in new_msg2.split():        
        if(ww.lower() not in stop_words):
            tmp_list.append(ww.lower())
    new_msg3=' '.join(tmp_list)
    new_msg4=[ps.stem(w) for w in new_msg3.split()]
    return ' '.join(new_msg4)
df['messages']=df.messages.apply(clean_text)
sparse_mtr=cv.fit_transform(df.messages)
X=sparse_mtr.toarray()
y=df.label	
gnb=GaussianNB()
gnb.fit(X,y)

def vp_start_gui():
    global val, w, root
    root = tk.Tk()
    top = Toplevel1 (root)
    root.mainloop()

def predict(top):
	test=clean_text(top.Entry1.get())
	test_X=cv.transform([test])
	pred=gnb.predict(test_X.toarray())
	top.Label2.configure(text=pred[0])

class Toplevel1:
	def __init__(self, top=None):
		_bgcolor = '#d9d9d9'  # X11 color: 'gray85'
		_fgcolor = '#000000'  # X11 color: 'black'
		_compcolor = '#d9d9d9' # X11 color: 'gray85'
		_ana1color = '#d9d9d9' # X11 color: 'gray85'
		_ana2color = '#ececec' # Closest X11 color: 'gray92'
		font14 = "-family {Segoe UI} -size 14 -weight bold -slant roman -underline 0 -overstrike 0"
		font15 = "-family {Segoe UI} -size 12 -weight bold -slant roman -underline 0 -overstrike 0"
		font18 = "-family {Segoe UI} -size 28 -weight bold -slant roman -underline 1 -overstrike 0"
		top.geometry("856x450+209+115")
		top.title("Spam Prediction Project")
		top.configure(background="#d9d9d9")
		self.Label1 = tk.Label(top)
		self.Label2 = tk.Label(top)
		self.Label1.place(relx=0.188, rely=0.244, height=31, width=185)
		self.Label1.configure(background="#d9d9d9")
		self.Label1.configure(font=font14)
		self.Label1.configure(text='''Enter your message:''',fg='blue')
		
		self.Entry1 = tk.Entry(top)
		self.Entry1.place(relx=0.427, rely=0.244,height=30, relwidth=0.276)
		self.Entry1.configure(background="white")
		self.Entry1.configure(width=264)
		self.Entry1.configure('',font=font14)
		
		self.Button1 = tk.Button(top,command=lambda:predict(self))
		self.Button1.place(relx=0.387, rely=0.4, height=35)
		self.Button1.configure(background="#d9d9d9")
		self.Button1.configure(font=font15)
		self.Button1.configure(text='''Prediction''')

      
		self.Label2.place(relx=0.388, rely=0.556, height=31, width=103)
		self.Label2.configure(background="#d9d9d9")
		self.Label2.configure(disabledforeground="#a3a3a3")
		self.Label2.configure(font=font14)
		self.Label2.configure(foreground="#000000")
		self.Label2.configure(text='''''',fg='blue')

		
		self.Label3 = tk.Label(top)
		self.Label3.place(relx=0.282, rely=0.067, height=38, width=317)
		self.Label3.configure(activeforeground="#0000FF")
		self.Label3.configure(background="#d9d9d9")
		self.Label3.configure(disabledforeground="#a3a3a3")
		self.Label3.configure(font=font18)
		self.Label3.configure(foreground="#000000")
		self.Label3.configure(text='''Spam Detection''',fg='red')
		self.Label3.configure(width=317)

if __name__ == '__main__':
    vp_start_gui()




