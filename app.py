import streamlit as st
import pickle
import pandas as pd

# Page config (optional but good)
st.set_page_config(page_title="IPL Win Predictor", layout="centered")

# Load model
pipe = pickle.load(open('pipe.pkl','rb'))

# Teams and cities
teams = ['Sunrisers Hyderabad','Mumbai Indians','Royal Challengers Bangalore',
'Kolkata Knight Riders','Kings XI Punjab','Chennai Super Kings',
'Rajasthan Royals','Delhi Capitals']

cities = ['Hyderabad','Bangalore','Mumbai','Indore','Kolkata','Delhi',
'Chandigarh','Jaipur','Chennai','Cape Town','Port Elizabeth',
'Durban','Centurion','East London','Johannesburg','Kimberley',
'Bloemfontein','Ahmedabad','Cuttack','Nagpur','Dharamsala',
'Visakhapatnam','Pune','Raipur','Ranchi','Abu Dhabi',
'Sharjah','Mohali','Bengaluru']

# Title
st.title('🏏 IPL Win Predictor')

# Team selection
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select Batting Team', sorted(teams))

with col2:
    bowling_team = st.selectbox('Select Bowling Team', sorted(teams))

# City
selected_city = st.selectbox('Select Host City', sorted(cities))

# Target
target = st.number_input('Target Score', min_value=1)

# Match situation
col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Current Score', min_value=0)

with col4:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0)

with col5:
    wickets_out = st.number_input('Wickets Fallen', min_value=0, max_value=10)

# Prediction
if st.button('Predict Probability'):

    runs_left = target - score
    balls_left = 120 - int(overs * 6)
    balls_left = max(balls_left, 1)

    wickets = 10 - wickets_out

    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left

    input_df = pd.DataFrame({
        'batting_team':[batting_team],
        'bowling_team':[bowling_team],
        'city':[selected_city],
        'runs_left':[runs_left],
        'balls_left':[balls_left],
        'wickets':[wickets],
        'total_runs_x':[target],
        'crr':[crr],
        'rrr':[rrr]
    })

    result = pipe.predict_proba(input_df)

    loss = result[0][0]
    win = result[0][1]

    # Output
    st.subheader("Win Probability")

    st.success(f"{batting_team}: {round(win*100)}% chance to win")
    st.error(f"{bowling_team}: {round(loss*100)}% chance to win")

    st.progress(int(win * 100))