import streamlit as st
import pandas as pd
import numpy as np
# import altair as alt
# import pydeck as pdk

# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# from sqlalchemy import create_engine

from PIL import Image

st.title("Beyond Parasite")
st.markdown(
"""
This is a recommender for Korean Movies.
""")

# st.subheader('Texi Driver')
# image = Image.open('Texi_Driver.jpg')
# st.image(image,caption='Texi Driver',use_column_width=True)
# 'A widowed father and taxi driver who drives a German reporter from Seoul to Gwangju to cover the 1980 uprising, soon finds himself regretting his decision after being caught in the violence around him.'

@st.cache
def load_data(DATA_URL):
    data = pd.read_csv(DATA_URL)
    return data

df = load_data("https://beyondparasite.s3-us-west-1.amazonaws.com/data/imdb_mat_f.csv")

# year options
year_min, year_max = st.slider('Year', 1960, 2019, (2010, 2019))

# genre options
genre_options=st.multiselect('Genre',df['genre1'].unique())
# st.write('You selected:',genre_options)

new_df = df[np.logical_and(df['year'] >= year_min, df['year'] <= year_max)]
if len(genre_options)>0:
    new_df = new_df[new_df.genre1.isin(genre_options)]
else:
    pass

# if len(new_df)>0:
#     st.write(new_df)
# else:
#     st.write('Oops! Your selections return no result.')

# recommender

# DATA_URL3 = (
#     "https://beyondparasite.s3-us-west-1.amazonaws.com/data/k_ratings_f.csv"
# )
# k_ratings_f = pd.read_csv(DATA_URL3)
#
# k_moviemat = k_ratings_f.pivot_table(index='userId',columns='title',values='rating_x')
# ratings_final = pd.DataFrame(k_ratings_f.groupby('title')['rating_x'].mean())
# ratings_final['num of ratings'] = pd.DataFrame(k_ratings_f.groupby('title')['rating_x'].count())
# def corr_list(movie_nm):
#     user_ratings = k_moviemat[movie_nm]
#     similar = k_moviemat.corrwith(user_ratings)
#     corr = pd.DataFrame(similar, columns=['Correlation'])
#     corr.dropna(inplace=True)
#     corr = corr.join(ratings_final['num of ratings'])
#     corr_list = corr[corr['num of ratings']>100].sort_values('Correlation',ascending=False)
#     return corr_list

# movie_options = [" - "] + list(set(new_df['title']))
# option1 = st.selectbox('What is your favorite Korean movie?',movie_options)
# st.write(option1)

# pd.set_option('display.max_colwidth', 1000)

# df2 = load_data("https://movie2010s.s3-us-west-1.amazonaws.com/data/youtube_videoid_2019.csv")

#"Select your level:"
st.sidebar.subheader('Level')
level = st.sidebar.selectbox('', ('Beginner','Intermediate','Advanced'))
#level = st.sidebar.selectbox('Level', ('Beginner','Intermediate','Advanced'))

st.sidebar.subheader('Your genre preference')
#st.sidebar.markdown('Please select your genre preference:')

Action = st.sidebar.slider('Action', 1, 5, 3)
Drama = st.sidebar.slider('Drama', 1, 5, 3)
Crime = st.sidebar.slider('Crime', 1, 5, 3)
Comedy = st.sidebar.slider('Comedy', 1, 5, 3)
Thriller = st.sidebar.slider('Thriller', 1, 5, 3)
Horror = st.sidebar.slider('Horror', 1, 5, 3)
Mystery = st.sidebar.slider('Mystery', 1, 5, 3)
Animation = st.sidebar.slider('Animation', 1, 5, 3)

# recommender

original_df = load_data("https://beyondparasite.s3-us-west-1.amazonaws.com/data/original_df.csv")
intermediate_df = load_data("https://beyondparasite.s3-us-west-1.amazonaws.com/data/intermediate_df.csv")
advanced_df = load_data("https://beyondparasite.s3-us-west-1.amazonaws.com/data/advanced_df.csv")
beginner_df = load_data("https://beyondparasite.s3-us-west-1.amazonaws.com/data/beginner_df.csv")
df_movie=original_df[['movieId','title']].drop_duplicates(subset=None, keep='first')

original_mat_prediction = load_data("https://beyondparasite.s3-us-west-1.amazonaws.com/data/original_mat_prediction.csv")
intermediate_mat_prediction = load_data("https://beyondparasite.s3-us-west-1.amazonaws.com/data/intermediate_mat_prediction.csv")
advanced_mat_prediction = load_data("https://beyondparasite.s3-us-west-1.amazonaws.com/data/advanced_mat_prediction.csv")
beginner_mat_prediction = load_data("https://beyondparasite.s3-us-west-1.amazonaws.com/data/beginner_mat_prediction.csv")

original_userId = original_df.groupby('userId')['movieId'].count().sort_values(ascending=False)
original_userId = pd.DataFrame({'userId':original_userId.index,'ratings_count':original_userId.values})
selected_userId = original_userId.copy().loc[original_userId['ratings_count'] >= 30]
# st.write(selected_userId['ratings_count'].quantile(0.9))

umap100 = load_data("https://beyondparasite.s3-us-west-1.amazonaws.com/data/UMAP_100.csv")
umap100 = umap100.merge(selected_userId.drop('ratings_count',axis=1),on='userId')
# st.write(len(umap100))

from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
cosine_sim = cosine_similarity(umap100[['Action','Animation','Comedy','Crime','Drama','Horror','Mystery','Thriller']],
            [[Action,Animation,Comedy,Crime,Drama,Horror,Mystery,Thriller]])

userIndex = umap100.loc[cosine_sim.argmax()]['userId'].astype(int)


def recommend_movies(preds_df, userIndex, movies_df, original_ratings_df, num_recommendations=5):

    user_row_number = preds_df[preds_df['userId'] == userIndex].index.item()
    sorted_user_predictions = preds_df.iloc[user_row_number].drop('userId').sort_values(ascending=False)
    # user_data = original_ratings_df[original_ratings_df.index==user_row_number]
    user_data = original_ratings_df[original_ratings_df['userId']==userIndex]
    recommendations = pd.DataFrame(sorted_user_predictions).reset_index().astype(int).merge(
                    df_movie[~df_movie.movieId.isin(user_data['movieId'])],left_on='index',right_on='movieId').rename(
                    columns = {user_row_number: 'Predictions'})#.iloc[:num_recommendations]

    return user_data, recommendations;

if level == 'Beginner':
    (user, recommendations) = recommend_movies(original_mat_prediction,userIndex,df_movie,original_df.reset_index())
elif level == 'Intermediate':
    (user, recommendations) = recommend_movies(beginner_mat_prediction,userIndex,df_movie,beginner_df.reset_index())
else:
    (user, recommendations) = recommend_movies(intermediate_mat_prediction,userIndex,df_movie,intermediate_df.reset_index())
    # (user, recommendations) = recommend_movies(advanced_mat_prediction,userIndex,df_movie,advanced_df.reset_index(),num_recommendations=6)

num_recommendations = 6
recommendations_final = recommendations.merge(new_df,on='movieId')
# st.write(recommendations_final)

import math

if level == 'Beginner':
    recommendations_final['Predictions']=recommendations_final['Predictions']*np.log10(recommendations_final['vote'])
    recommendations_final['Predictions']=recommendations_final.apply(lambda n:np.log10(n['vote']) if n['Predictions']==0 else n['Predictions'],axis=1)
    recommendations_final.sort_values(by=['Predictions'],ascending=False,inplace=True)
elif level == 'Advanced':
    recommendations_final['Predictions']=recommendations_final['Predictions']/np.log10(recommendations_final['vote'])
    # recommendations_final['Predictions']=recommendations_final.apply(lambda n:np.log10(n['vote']) if n['Predictions']==0 else n['Predictions'],axis=1)
    recommendations_final.sort_values(by=['Predictions'],ascending=False,inplace=True)
else:
    pass

image = []
caption = []
if len(recommendations_final) < 1:
    #st.text("Opps, no recommendations found!")
    st.markdown(
    """
    Opps, no recommendations found!
    """)
    length = 0
else:
    length = min(num_recommendations,len(recommendations_final))
    for i in range(length):
        image.append(recommendations_final.iloc[i].poster)
        caption.append(recommendations_final.iloc[i].title_x+"\n"+recommendations_final.iloc[i].story)

# st.write(length)
# st.image([image[0],image[1],image[2]],[caption[0],caption[1],caption[2]],width=219)

if length == 6:
    st.image([image[0],image[1],image[2]],[caption[0],caption[1],caption[2]],width=219)
    st.image([image[3],image[4],image[5]],[caption[3],caption[4],caption[5]],width=219)
elif length == 5:
    st.image([image[0],image[1],image[2]],[caption[0],caption[1],caption[2]],width=219)
    st.image([image[3],image[4]],[caption[3],caption[4]],width=219)
elif length == 4:
    st.image([image[0],image[1],image[2]],[caption[0],caption[1],caption[2]],width=219)
    st.image([image[3]],[caption[3]],width=219)
elif length == 3:
    st.image([image[0],image[1],image[2]],[caption[0],caption[1],caption[2]],width=219)
elif length == 2:
    st.image([image[0],image[1]],[caption[0],caption[1]],width=219)
elif length == 1:
    st.image([image[0]],[caption[0]],use_column_width=True)

st.write(userIndex)
st.write(user)
st.write(recommendations_final)
#st.write(recommendations_final[['index','Predictions','movieId','title_x']][:length])

# st.write(recommendations_final)

# (user, recommendations) = recommend_movies(
#     original_mat_prediction,4311,df_movie,
#     original_df.reset_index(),num_recommendations=6)

# st.write(recommendations.merge(df,on='movieId'))

# # first row
# image1 = recommendations_final.loc[0].poster
# caption1 = recommendations_final.loc[0].story
#
# # image1 = recommendations_final.merge(new_df,on='movieId').loc[0].poster
# # caption1 = recommendations_final.merge(new_df,on='movieId').loc[0].story
# #st.image(image1,caption=caption1,width=219)
#
# image2 = recommendations_final.loc[1].poster
# caption2 = recommendations_final.loc[1].story
# # image2 = recommendations_final.merge(new_df,on='movieId').loc[1].poster
# # caption2 = recommendations_final.merge(new_df,on='movieId').loc[1].story
# #st.image(image2,caption=caption2,width=219)
#
# image3 = recommendations_final.loc[2].poster
# caption3 = recommendations_final.loc[2].story
# # image3 = recommendations_final.merge(new_df,on='movieId').loc[2].poster
# # caption3 = recommendations_final.merge(new_df,on='movieId').loc[2].story
# #st.image(image3,caption=caption1,width=219)
#
# # second row
# image4 = recommendations_final.loc[3].poster
# caption4 = recommendations_final.loc[3].story
# # image4 = recommendations_final.merge(new_df,on='movieId').loc[3].poster
# # caption4 = recommendations_final.merge(new_df,on='movieId').loc[3].story
# #st.image(image4,caption=caption1,width=219)
#
# image5 = recommendations_final.loc[4].poster
# caption5 = recommendations_final.loc[4].story
# # image5 = recommendations_final.merge(new_df,on='movieId').loc[4].poster
# # caption5 = recommendations_final.merge(new_df,on='movieId').loc[4].story
# #st.image(image5,caption=caption2,width=219)
#
# image6 = recommendations_final.loc[5].poster
# caption6 = recommendations_final.loc[5].story
# # image6 = recommendations_final.merge(new_df,on='movieId').loc[5].poster
# # caption6 = recommendations_final.merge(new_df,on='movieId').loc[5].story
# #st.image(image6,caption=caption1,width=219)
#
# st.image([image1,image2,image3],[caption1,caption2,caption3],width=219)
# st.image([image4,image5,image6],[caption4,caption5,caption6],width=219)


# if option1 != " - ":
#     # image = Image.open(df.loc[df['title']==option1]['poster'].to_string())
#     image_url = df.loc[df['title']==option1].poster.item()
#     # st.image(image_url,use_column_width=True)
#     st.image(image_url,width=219)
#     # top3 = corr_list(option1).head(3)
#     # st.write(top3)
# else:
#     st.write("Select a movie to start!")

# 'A widowed father and taxi driver who drives a German reporter from Seoul to Gwangju to cover the 1980 uprising, soon finds himself regretting his decision after being caught in the violence around him.'

# for top in top3:
#     st.write(df[df['title']==top.index[0]]['story'].to_string())

# c = alt.Chart(df[(
#             df['Production Budget']>=1000000)
#             #& (df['Rank']<=20)
#             & (df['Genre']==genre)
#             ],
#     height=500).mark_circle().encode(
#     x="Production Budget",y="Ratio",size='Opening Weekend Revenue',color="Theatrical Distributor",
#     tooltip=['Production Budget', 'Ratio', 'Title']
#     ).interactive()
# st.altair_chart(c, use_container_width=True)


# from scipy import spatial
# cosine_sim2 = []
# for index, row in umap100.iterrows():
#     cosine_sim2.append(1-spatial.distance.cosine(row[['Action','Animation','Comedy','Crime','Drama','Horror','Mystery','Thriller']],
#                 [[Action,Animation,Comedy,Crime,Drama,Horror,Mystery,Thriller]]))
#
# cosine_sim2 = np.asarray(cosine_sim2)
#
# st.write(cosine_sim2)
# st.write(cosine_sim2.argmax())

# options = [" - "] + list(set(df2['Title']))
# option1 = st.sidebar.selectbox('What is your favorite 2019 movie?',options)
# # 'You Option 1:', option1
#
# remaining_options = [" - "] + list(set(df2['Title'])-set([option1]))
# option2 = st.sidebar.selectbox('2nd:',remaining_options)
# # 'You Option 2:', option2
#
# remaining_options2 = [" - "] + list(set(df2['Title'])-set([option1])-set([option2]))
# option3 = st.sidebar.selectbox('3rd:',remaining_options2)
# # 'You Option 3:', option3

# option2 = st.sidebar.selectbox(
#     'Which 2019 movie do you choose?',
#      df2['Title'])
# 'You Option 2:', option2
#
# option3 = st.sidebar.selectbox(
#     'Which 2019 movie do you choose?',
#      df2['Title'])
# 'You Option 3:', option3
