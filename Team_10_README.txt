

README to execute the code

OBJECTIVE - To analyse demand with respect to different variables and to suggest ways to improve demand.

Data - Main datasets were downloaded from Kaggle - https://www.kaggle.com/nikdavis/steam-store-games
            Data scraped from steam website using dataset 1 containing weblinks to each game

Dependencies -
1) pandas
2) numpy
3) sklearn
4) wordcloud
5) nltk
6) skimage
7) heapq

Steps carried out during pre-processing:

1)   First all the necessary libraries were imported.
2)   Then the datasets downloaded from Kaggle were imported using pandas.
3)   Then all the necessary columns which are to be utilized were created as empty.
4)   Further, two new columns named total_ratings and positive_percent were created for our analysis.
5)   We analysed mean reviews and owners to help us convert owners to continuous from categorical variable. 
6)   After that oweners were converted from categorical to continuous variables.
7)   Scraping was carried out usng selenium to find if any free game has Downloadable Content in it.
8)   After scraping dumy variables were created for categorical columns i.e. for - 
			a) action
			b) In-App Purchases
			c) windows
			d) mac
			e) linux

9)   Image processing was done using dataset 2 containing html path to cover image/icon image of each game	in order to
      extrct contrast and brightness of images to check whether these variables had any effect on downloads and hence demand
10) Sentimental analysis was carried out to find the most prequently used words in the discription of the game to find out the 
       leading trend in the industry 
11) Natural language processing was done using NLTK to find the most popular topics in the description of the games
12) All the datasets were then saved, treated in minitab and analysiss was carried on minitab. 

