import tkinter
from tkinter import *
import numpy as np
import urllib.request
import tkinter.scrolledtext as tkst
from tkinter.ttk import *
from bs4 import BeautifulSoup
import pandas as pd
from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import load_model


TRAIN_DATA_PARA2SENT="/home/manish/Downloads/MRPC_DIR/test.tsv"
#TRAIN_LABEL_PARA2SENT="semeval-2014_task-3/SemEval-2014_Task-3/keys/training/paragraph2sentence.train.gs.tsv"
#MODEL_FILE='/home/manish/TAMU_FALL_2019/RESEARCH/second_model_train/third_model.h5'
MODEL_FILE='/home/manish/TAMU_FALL_2019/RESEARCH/second_model_train/second_model.h5'
####################################################
#Input 
docs_input=[
"Naftali (Tali) \n Tishby Physicist, professor of computer science and computational neuroscientist. The Ruth and Stan Flinkman professor of Brain Research \n I work at the interfaces between computer science, physics, and biology which provide some of the most challenging problems in today’s science and technology. We focus on organizing computational principles that govern information processing in biology, at all levels. To this end, we employ and develop methods that stem from statistical physics, information theory and computational learning theory, to analyze biological data and develop biologically inspired algorithms that can account for the observed performance of biological systems. We hope to find simple yet powerful computational mechanisms that may characterize evolved and adaptive systems, from the molecular level to the whole computational brain and interacting populations.",
"Jurgen Schmidhuber's home page \n Since age 15 or so, the main goal of professor Jürgen Schmidhuber has been to build a self-improving Artificial Intelligence (AI) smarter than himself, then retire. His lab's Deep Learning Neural Networks (since 1991) such as Long Short-Term Memory (LSTM) have transformed machine learning and AI, and are now (2017) available to billions of users through the world's most valuable public companies. He also generalized algorithmic information theory and the many-worlds theory of physics, and introduced the concept of Low-Complexity Art, the information age's extreme form of minimal art. He is recipient of numerous awards, and Chief Scientist of the company NNAISENSE, which aims at building the first practical general purpose AI. He is also advising various governments on AI strategies. Teaching Master's in Artifical Intelligence (Fall 2019)",
"Hector Zenil \n I also lead the Algorithmic Nature Group, the Paris-based lab that started the Online Algorithmic Complexity Calculator and the Human Randomness Perception and Generation Project (triggering wide media coverage), an inverse Turing test performed on ~3.4K people.​  Previously, I was a Research Associate at the Behavioural and Evolutionary Theory Lab at the Department of Computer Science at the University of Sheffield in the UK before joining the Department of Computer Science at the University of Oxford as a Senior Researcher (faculty member) and director of Oxford Immune Algorithmics. I am also a member of the Mexican National System of Researchers (SNI II), and an elected member of the London Mathematical Society in the UK. ",
"Aditya Sharma \n Information Theory of Deep Learning In this post, I will try to summarize the findings and research done by Prof. Naftali Tishby which he shares in his talk on Information Theory of Deep Learning at Stanford University recently. There have been many previous versions of the same talk so don’t be surprised if you have already seen one of his talks on the same topic. Most of the summary will be based on his research and I will try to include some relevant mathematics (not everything) for better understanding of the concepts",
"David MacKay \n The Back Cover \n Information theory and inference, often taught separately, are here united in one entertaining textbook. These topics lie at the heart of many exciting areas of contemporary science and engineering - communication, signal processing, data mining, machine learning, pattern recognition, computational neuroscience, bioinformatics, and cryptography. This is an extraordinary and important book, generous with insight and rich with detail in statistics, information theory, and probabilistic modeling across a wide swathe of standard, creatively original, and delightfully quirky topics. David MacKay is an uncompromisingly lucid thinker, from whom students, faculty and practitioners all can learn. ",
"Yaser S. Abu-Mostafa \n Professor of Electrical Engineering and Computer Science \n Machine learning applies to any situation where there is data that we are trying to make sense of, and a target function that we cannot mathematically pin down. The spectrum of applications is huge, going from financial forecasting to medical diagnosis to industrial inspection to recommendation systems, to name a few. The field encompasses neural networks, statistical inference, and data mining. He is also interested in using quantum information theory to shed new light on fundamental techniques in theoretical computer science such as semidefinite programming and approximation algorithms.",
"American football, referred to as football in the United States and Canada and also known as gridiron,[nb 1] is a team sport played by two teams of eleven players on a rectangular field with goalposts at each end. The offense, which is the team controlling the oval-shaped football, attempts to advance down the field by running with or passing the ball, while the defense, which is the team without control of the ball, aims to stop the offense's advance and aims to take control of the ball for themselves. The offense must advance at least ten yards in four downs, or plays, and otherwise they turn over the football to the defense; if the offense succeeds in advancing ten yards or more, they are given a new set of four downs. Points are primarily scored by advancing the ball into the opposing team's end zone for a touchdown or kicking the ball through the opponent's goalposts for a field goal. The team with the most points at the end of a game wins. ",
"American football is the most popular sport in the United States.The National Football League has the highest average attendance of any sports league in the world. In the United States the game is most often referred to as simply football .There is no single national governing body for American football in the United States or a continental governing body for North America. There is an international governing body, the International Federation of American Football, or IFAF. Befitting its status as a popular sport, football is played in leagues of different size, age and quality, in all regions of the country. Organized football is played almost exclusively by men and boys, although a few amateur and semi-professional women's leagues have begun play in recent years. A team / academy may be referred to as a 'football program' – not to be confused with football program. ",
"Popular Classic Must Reads Books Pride and Prejudice (Paperback) Jane Austen. The Great Gatsby (Paperback) F. 1984 (Kindle Edition) George Orwell, To Kill a Mockingbird (Paperback) Harper Lee. ...The Catcher in the Rye (Paperback) J.D. Salinger. ...Wuthering Heights (Paperback) Emily Brontë ...Lord of the Flies (Paperback) William Golding. Little Women by Louisa May Alcott, Jane Eyre by Charlotte Brontë, Fahrenheit 451 (Kindle Edition) by Ray Bradbury, The Odyssey (Paperback) by Homer, Brave New World (Paperback)  by Aldous Huxley, Of Mice and Men (Paperback) by John Steinbeck, Anna Karenina by Leo Tolstoy, A Tale of Two Cities (Paperback) by Charles Dickens, The Scarlet Letter (Paperback) by Nathaniel Hawthorne, Les Misérables (Mass Market Paperback) by Victor Hugo , Animal Farm by George Orwell",
"100 must-read classic books, as chosen by our readers. They broke boundaries and challenged conceptions. We asked our readers for their must-reads; from timeless non-fiction to iconic bestsellers, these are their essential recommends. The Great Gatsby F. Scott Fitzgerald  Buy the book 1. The Great Gatsby by F. Scott Fitzgerald.The greatest, most scathing dissection of the hollowness at the heart of the American dream. Hypnotic, tragic, both of its time and completely relevant. One Hundred Years of Solitude,Gabriel Garcia Marquez . Magic realism at its best. Both funny and moving, this book made me reflect for weeks on the inexorable march of time.",
"We’ve already recommended our picks for the best 50 books of the past 50 years, but now we’re diving deeper into our literary history, temporally speaking. These are our picks for the 50 most essential classic books. You know, the ones that everyone should get around to reading sooner, rather than later. These have meant a great deal to readers throughout the centuries, and they distinguish themselves as firsts and bests, sure, but also unexpected, astonishing, and boundary-breaking additions to the canon. That’s why we’re still reading them. Everyone has his or her own definition of a literary classic, and our choices span the centuries, from the 8th century B.C. to the English Renaissance to the mid-20th century. (We’ve even included a book from the 1990s, as we’re convinced it’s going to go down in history as a classic.) ",
"One of my aims is to begin catching up on all the reading I’ve neglected for, well, the majority of my life. So, I started by googling several combinations of ‘books to read before you die,’ ‘100 most important books,’ ‘books everyone should should read in a lifetime,’ and so on. I discovered that quite a few reputable (and a few not-so-reputable) sources have published such a list. Nice, but it still leaves me at a loss for what to do next. Which list do I go with? After carefully reading through what was on offer I decided to take the collective wisdom from the various sources by painstakingly comparing (well, I hired ‘Vi’ from Vietnam via Elance to painstakingly compare) all of the lists to determine how much overlap existed between them. Here are the 8 lists I started with, amalgamated, and called The Guardian’s The 100 greatest novels of all time.",
"From Heathcliff to Bovary to Becky Sharp, meet some of the greatest literary creations of all time in our list of 20 essential classic novels to read before you die.They've withstood the passage of time to bewitch countless readers all over the world. But, with such a wealth of great literature, it can be hard to know where to start. Fret not! We've sifted through them to come up with a list of works no book-lover should miss our 20 Classic Books you have to read before you die.",
"As in England, where the sport first developed, early football in the United States was relatively disorganized and often quite violent. Different towns and schools played by their own sets of rules, but they all involved two sides of a dozen or more men on foot rather than horseback—hence the sport's name—attempting to direct a ball toward goals at opposite ends of the field. With the rising popularity of interscholastic competition, football gradually became more formalized on both sides of the Atlantic during the second half of the nineteenth century. In 1863, English proponents of the relatively non-violent, no-handling version of the game created the Football Association, whose distinctive soccer rules have since become the world's most popular football code. One year earlier, Gerritt Smith Miller established the first formal football club in the United States, Boston Common's Oneida Club."]




#####################################################

class Gyata:
    def __init__(self, master):
        self.master = master
        master.title("GUI TinySearch")

        ####################
        # Initializations
        #self.bc = BertClient(port=5555,port_out=5556,check_version=False)
        # Make socket 
        self.bc = BertClient(ip='128.194.142.14',port=5555,port_out=5556,check_version=False)
        #self.train_df = pd.read_csv(TRAIN_DATA_PARA2SENT,engine='python',sep='\t')
        # Load the nn model
        self.model = load_model(MODEL_FILE)
        # Encode the docs
        self.docs_vec=self.bc.encode(docs_input)

        self.text_html = tkst.ScrolledText(master,width=200)
        #self.text_html.insert(1.0,self.train_data[10:30,3])
        for i in range(len(docs_input)):
            self.text_html.insert(END,"Document ")
            self.text_html.insert(END,i+1)
            self.text_html.insert(END,"\n")
            self.text_html.insert(END,docs_input[i])
            self.text_html.insert(END,"\n\n\n\n")
            
        ####################
        self.valid_url = None
        self.query_response = StringVar()
        self.query_response.set("")

        self.message_url = "Enter the website link"
        self.label_text_url = StringVar()
        self.label_text_url.set(self.message_url) 
        self.label_url = Label(master,textvariable=self.label_text_url)

        vcmd = master.register(self.validate) # we have to wrap the command
        self.url_entry=Entry(master,validate="key",validatecommand=(vcmd,'%P'))

        #self.url_entry = self.url_input_box.get("1.0","end-1c")       

        
        self.message = "Enter the string you want to find "
        self.label_text = StringVar()
        self.label_text.set(self.message)
        self.label_query = Label(master, textvariable=self.label_text,wraplength=600,justify="left")

        self.label_train_data_text = StringVar()
        self.label_train_data_text.set("Row number of train data")
        self.label_train_data=Label(master,textvariable=self.label_train_data_text)

        self.query_entry = Entry(master)
        self.train_data_entry = Entry(master)

        # Scroll Area to display results
         
        #self.text_data = tkst.ScrolledText(master)
        self.data_col1= StringVar()
        self.data_col2= StringVar()
        self.data_col3= StringVar()

        self.label_data_col1=Label(master,textvariable=self.data_col1,wraplength=600,justify="left")
        self.label_data_col2=Label(master,textvariable=self.data_col2,wraplength=300,justify="right")
        self.label_data_col3=Label(master,textvariable=self.data_col3,wraplength=100,justify="center")


        self.find_button = Button(master, text="Find", command=self.find_query)
        self.search_again_button = Button(master, text="Search again", command=self.reset, state=DISABLED)
        #self.train_data_button = Button(master, text="See Training Data", command=self.read_data)

        self.label_query_response = Label(master,textvariable=self.query_response,wraplength=600,justify="left");
 
        # organization on screen

        #self.label_url.grid(row=0, column=0, sticky=W)
        #self.url_entry.grid(row=0, column=1, columnspan=2, sticky=W+E)
        self.label_query.grid(row=1, column=0, sticky=W)
        self.query_entry.grid(row=1, column=1, columnspan=2, sticky=W+E)
        #self.label_train_data.grid(row=0,column=6,sticky=W)
        #self.train_data_entry.grid(row=0,column=7)
        self.find_button.grid(row=2, column=0,sticky=W)
        self.search_again_button.grid(row=2, column=1,sticky=W)
        #self.train_data_button.grid(row=2, column=6)
        self.text_html.grid(row=3,column=0,columnspan=20,rowspan=15,sticky=W+E)
        #self.label_data_col1.grid(row=3,column=5,columnspan=5,rowspan=3,stick=W)
        #self.label_data_col2.grid(row=3,column=12,columnspan=3,rowspan=3,stick=W)
        #self.label_data_col3.grid(row=3,column=16,columnspan=1,rowspan=3,stick=W)

        #self.text_data.grid(row=3,column=6,columnspan=15,rowspan=10,sticky=S)
        self.label_query_response.grid(row=20,column=0,sticky=W)




    def validate(self, new_text):
        if not new_text: # the field is being cleared
            self.valid_url = None
            return True
        
        self.valid_url = new_text
        return True
       
    def tokenize(self,parsedhtml):
        
        # remove () by replacing them with space
        parsedhtml = parsedhtml.replace('(',' ').replace(')',' ').replace(':',' ').replace(',',' ')
        # remove :: and [] and /  and -- and  . by replacing them with space
        parsedhtml = parsedhtml.replace('[',' ').replace(']',' ').replace('::',' ').replace('/',' ').replace('--',' ').replace('.',' ')
        # split on space and save in tokens_list
        tokens_list=parsedhtml.split() 
        
        return tokens_list 

    def find_query(self):
       
       
        query=self.bc.encode([self.query_entry.get()])[0]
        queryT = np.asarray(query,np.float32) 
        queryT = queryT.reshape(1,768)
        #print(np.shape(queryT))
        # we are going to calculate the relevance score for each 
        # doc and query pair and store in score
        '''score = []
        print(np.shape(self.docs_vec))
        for doc in self.docs_vec:
            docT = np.asarray(doc,np.float32)
            docT = docT.reshape(1,768)
            #print(queryT)
            #print(docT)
            #print(np.shape([queryT,docT]))
            score.append( self.model.predict([queryT,docT]).tolist()[0][0] )
        print(score)
        '''
        score = np.sum(query * self.docs_vec, axis=1) / np.linalg.norm(self.docs_vec, axis=1)
        score /= np.linalg.norm(query)
        #score = np.sum(query * self.doc_vec, axis=1)
        #score = []
        #for i in range(len(self.doc_vec)):
        #score=cosine_similarity(query,self.doc_vec)
        
        topk_idx = np.argsort(score)[::-1][:5]
        l=[]
        for idx in topk_idx:
            l.append('%s \n\n\n' % (docs_input[idx]))
	    
            #rint('> %s\t%s' % (score[idx], self.docs_list[idx]))

        
        self.query_response.set(l)
        
        self.find_button.configure(state=DISABLED)
        self.search_again_button.configure(state=NORMAL)
    
    def reset(self):
        self.query_entry.delete(0, END)
        self.url_entry.delete(0, END)
        #self.text_html.delete('1.0',END)

        self.message = "Enter the website link"
        self.label_text_url.set(self.message)
       
        self.message= "Enter the string you want to find "
        self.label_text.set(self.message)

        self.query_response.set("")

        self.find_button.configure(state=NORMAL)
        self.search_again_button.configure(state=DISABLED)
'''
    def read_data(self):
        row = int(self.train_data_entry.get())
        self.data_col1.set("")
        self.data_col2.set("")
        self.data_col3.set("")
        #self.text_data.delete('1.0', END)
        #self.text_data.insert(1.0,self.train_data[row,:])
        self.data_col1.set(self.train_data[row,0])
        self.data_col2.set(self.train_data[row,1])
        self.data_col3.set(self.train_label[row,0])
'''        
root = Tk()
root.style = Style()
root.style.theme_use("clam")
root.geometry("1600x600")
my_gui = Gyata(root)
root.mainloop()
