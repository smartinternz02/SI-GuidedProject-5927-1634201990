from flask import Flask, request, render_template
# Flask-It is our framework which we are going to use to run/serve our application.
#request-for accessing file which was uploaded by the user on our application.
import pandas as pd # used for data manipulation
import pickle


app = Flask(__name__) # initializing a flask app
model=pickle.load(open("book.pkl",'rb'))  #loading the model

#loading the updated dataset
us_canada_user_rating_pivot=pd.read_csv("us_canada_user_rating_pivot1.csv",
                                        encoding = "ISO-8859-1", index_col='Book-Title')


@app.route('/')# route to display the home page
def home():
    return render_template('home.html')#rendering the home page


@app.route('/extractor')
def extractor():
    return render_template('extractor.html')

#extractor page
@app.route('/keywords',  methods=['POST'])# route to show the predictions in a web UI
def keywords():
    typ=request.form['type']
    output=request.form['output']
    #if typ=="text":
        #output=re.sub("[^a-zA-Z.,]"," ",output)
    print(output)
    distances,indices=model.kneighbors(
        us_canada_user_rating_pivot.loc[output,:].values.reshape(1, -1), n_neighbors = 6)
    keyword=[]
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('Recommendations for {0}:\n'.format(output))
        else:
            keyword.append('{0}: {1}'.format(i, 
                                us_canada_user_rating_pivot.index[indices.flatten()[i]]))
    
   # showing the prediction results in a UI
    return render_template('keywords.html',keyword=keyword)

    
if __name__ == "__main__":
   # running the app
    app.run(debug=False)
