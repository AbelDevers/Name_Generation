# Name_Generation
-------------------------------- Table of content     --------------------------------------
1. Project presentaiton
2. How to use the code
3. Non-exhaustive list of countries used
-------------------------------- Project Presentation --------------------------------------
This small project has been developped with a friend working towards a PhD in linguistic. 

We were both curious to see if recurrents neural network could efficiently understand the logics of sounds and writing underlying different languages. 

To do so, I have developped this small python code using tensorflow to generate random names of cities sounding french, german, english, japaneze, etc... 

Using a large free-database found online, with tens of thousands of names per country, we first vectorized the input. For each symbole, we assigned a vector with 0 in all spaces and 1 in the only relevent one. For example, a "a" is [1 0 0 ... 0] and b is [0 1 0 0 ... 0]. 
[0 .... 25 26 27 28 29] size 30
[a .... z  -  '  space nothing]
For each letter in any name we concatenate 20 of these vectors for the twenty previous letters (in the forward model) and twenty of the following letters in the backward models. So a word with four letters, will be four data points (one per letter) and each data point will be 20 concataneted 30 size vectors (600 size).

We then trained two models using a very standard recurrent ltsm neural network. On predicitng the next letter based on the previous (forward model) and one predicitong the previous letters based on the next (backward).

To generate new names, we have (quite long) procedure. First we generate random letters. Then, using the forward model. we update each letters, by taking pseudo randomly one new letter. We compute the output value for each letter, take the five most likely, standardize the probability and randomly select among those five. We then use the backward model, first to assign a probability to each letter and then to select a new letter for all letter with a likelihood below some treshold. We repete the operation, using alternatively the backward and forward model to update the letters bellow some likelihood threshold. At each step, we keep each generated names and their respective likelihood. 

Finaly, we keep only the top 20% (highest likelhood) of the generated name for a particular length of name. We start over with the same or a new lenght. 

-------------------------------- How to use the code ------------------------------------------
To generate some new name you have to do the following steps 
0. pip or conda
1. Vectorize backward data
2. Vectorize forward data
3. Train neural network
4. Call the pseudo random generation routine

0 pip or conda
To use this project you will need tensorflow and pickle, with python 3.5. You will also need to download all file as present in the git and keep them as such when running the code. 

1 & 2, vectorizing the data. 
Using the code createdata.py and createdata_backward.py, both case update the cc_flips variable to the country you want (see list below) and simply run the code. Quite a long process. 
At the end, the code will save a pickle with the vectorized data for the country corresponding to the cc_flips. Please note that the file can be quite heavy for some data heavy countries such as Russia or China. 
There is probably a way to be more efficient here, the idea of this project is just to do some exploration. 

3 Train neural networ
Usint the rrn_train.py in the same direcotry as the newly obtained pickles file (step 1 & 2). Update the cc_flips value to the desired country. Run the code. Ten epoch tend to be more than enough, but in some case, more might be a good idea, especially with very non-linear language. 

4 Call the pseudo random generation routine
Again, update the cc_flips value in the create_name_rnn.py file and launch the code. A new Output.txt file will be generated with the new randomized name. 

---------------------------------------- List of countries, cc_flips, example ----------------------


list of country indeces used (cc_flips value) 
GM - germany
UK - united kingdom
JA - Japan
IR - Iran
IT - Italia
RS - Russia
EZ - Czeech Republic
PO - Porutgal
SP - Spain
GR - Greece
SW - Sweeden

NO - Norway
DA - Danemark
