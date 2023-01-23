import time
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import SessionState

session = SessionState.get(run_id=0)

def predict_disease(input_data, scaler):
    """get probabilities of each label"""
    user_input = np.array([input_data]).astype(np.float64)
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)
    prediction_proba = model.predict_proba(user_input_scaled)

    # st.write("prediction: " + str(prediction))

    return int(prediction), prediction_proba


def main():
    st.set_page_config(page_title="Growing Pain Predictor", page_icon=":hospital:")

    st.markdown("<h1 style='text-align: center; color: black;'>Test I am here to facilitate Growing Pain (GP) diagnosis by using machine learning.</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'>Please fill the form below and click the predict button to see how likely your patient has GP.</h2>", unsafe_allow_html=True)

    st.markdown("<br/><hr>", unsafe_allow_html=True)

    # Here is the list of features to be sent to the model
    # st.write(model.features)

    inputs = []

    # Create a two-column form layout
    c1, c2 = st.beta_columns([1, 1])

    with c1:
        currentAge = st.number_input("Current Age", min_value=0, max_value=216, help="in months")
        height = st.number_input("Height", min_value=0, max_value=200, help="in centimeters")
        frequencyOfPain = st.selectbox("Frequency of Pain", ('', '1 Day a Week', '1 Day in 2 Weeks',
           '1 Day in 2-3 Days', '1 Day in 2-3 Weeks',
           '1 Day in 6 Months', '1 Week a Month',
           '1-2 Days a Month', '1-2 Days a Week',
           '1-2 Days in 2 Weeks', '1-2 Days in 3-4 Months',
           '2 Days a Month', '2 Days a Week',
           '2-3 Days a Month', '2-3 Days a Week',
           '2-3 Days in 2 Weeks', '3 Days a Week',
           '3-4 Days a Month', '3-4 Days a Week',
           '4 Days a Week', '4-5 Days a Week',
           '5 Days a Month', '5 Days a Week',
           '5-6 Days a Week', '6 Days a Week',
           'Everyday'))
        APR = st.radio("APR", ('Normal', 'High'))

    with c2:
        durationOfPain = st.number_input("Duration of Pain", min_value=0.0, max_value=200.0, step=0.1, help="in months")
        weight = st.number_input("Weight", min_value=0, max_value=150, help="in kilograms")
        assocWithTimeOfTheDay = st.selectbox("Associated with Time of the Day", ('', 'Morning', 'Evening', 'No Difference'))

        localizationOfPain = st.multiselect("Localization of Pain", ('LocalizationOnLeg',
           'LocalizationOnKnee', 'LocalizationOnCalf', 'LocalizationOnAnkle',
           'LocalizationOnThigh', 'LocalizationOnHip', 'LocalizationOnSacroiliac',
           'LocalizationOnFoot', 'LocalizationOnWrist', 'LocalizationOnArm',
           'LocalizationOnForeArm', 'LocalizationOnWaist', 'LocalizationOnHeel',
           'LocalizationOnNeck', 'LocalizationOnToe', 'LocalizationOnFinger',
           'LocalizationOnElbow', 'LocalizationOnHand', 'LocalizationOnFootTop',
           'LocalizationOnBack', 'LocalizationOnShoulder'), help="You can select more than one option.")

    # Create a three-column form layout
    c3, c4, c5 = st.beta_columns([1, 1, 1])

    with c3:
        unilatOrBilat = st.radio("Affected sides", ('Unilateral', 'Bilateral'))
        awakeningAtNight = st.radio("Awakening at Night?", ('Yes', 'No'))
        responseToMassage = st.radio("Response to Massage?", ('Yes', 'No'))
    with c4:
        historyOfArthritis = st.radio("History of Arthritis?", ('Yes', 'No'))
        morningStiffness = st.radio("MorningStiffness?", ('Yes', 'No'))
        limping = st.radio("Limping?", ('Yes', 'No'))
    with c5:
        limitationOfActivities = st.radio("Limitation of Activities?", ('Yes', 'No'))
        familyHistoryOfGrowingPain = st.radio("FamilyHistory of Growing Pain?", ('Yes', 'No'))
        physicalExamination = st.radio("Physical Examination?", ('Normal', 'Abnormal'))

    st.markdown("<hr>", unsafe_allow_html=True)

    left_button, right_button, _ = st.beta_columns([1,1,6])
    predicted = False
    with left_button:
        if st.button("Predict"):
            st.spinner()

            # For debugging purposes
            # st.write(currentAge)
            # st.write(durationOfPain)
            # st.write(localizationOfPain)
            # st.write(unilatOrBilat)
            # st.write(awakeningAtNight)
            # st.write(responseToMassage)
            # st.write(historyOfArthritis)
            # st.write(morningStiffness)
            # st.write(limping)
            # st.write(limitationOfActivities)
            # st.write(familyHistoryOfGrowingPain)
            # st.write(physicalExamination)
            # st.write(APR)
            # st.write(weight)
            # st.write(height)
            # st.write(BMI)
            # st.write(frequencyOfPain)
            # st.write(assocWithTimeOfTheDay)

            BMI = weight / (height / 100)**2

            # Create a list of values to use for prediction
            inputs.append(currentAge)
            inputs.append(durationOfPain)

            # localizationOfPain is a categorical variable
            inputs.append(1 if 'LocalizationOnLeg' in localizationOfPain else 0 )
            inputs.append(1 if 'LocalizationOnKnee' in localizationOfPain else 0 )
            inputs.append(1 if 'LocalizationOnCalf' in localizationOfPain else 0 )
            inputs.append(1 if 'LocalizationOnAnkle' in localizationOfPain else 0 )
            inputs.append(1 if 'LocalizationOnThigh' in localizationOfPain else 0 )
            inputs.append(1 if 'LocalizationOnHip' in localizationOfPain else 0 )
            inputs.append(1 if 'LocalizationOnSacroiliac' in localizationOfPain else 0 )
            inputs.append(1 if 'LocalizationOnFoot' in localizationOfPain else 0 )
            inputs.append(1 if 'LocalizationOnWrist' in localizationOfPain else 0 )
            inputs.append(1 if 'LocalizationOnArm' in localizationOfPain else 0 )
            inputs.append(1 if 'LocalizationOnForeArm' in localizationOfPain else 0 )
            inputs.append(1 if 'LocalizationOnWaist' in localizationOfPain else 0 )
            inputs.append(1 if 'LocalizationOnHeel' in localizationOfPain else 0 )
            inputs.append(1 if 'LocalizationOnNeck' in localizationOfPain else 0 )
            inputs.append(1 if 'LocalizationOnToe' in localizationOfPain else 0 )
            inputs.append(1 if 'LocalizationOnFinger' in localizationOfPain else 0 )
            inputs.append(1 if 'LocalizationOnElbow' in localizationOfPain else 0 )
            inputs.append(1 if 'LocalizationOnHand' in localizationOfPain else 0 )
            inputs.append(1 if 'LocalizationOnFootTop' in localizationOfPain else 0 )
            inputs.append(1 if 'LocalizationOnBack' in localizationOfPain else 0 )
            inputs.append(1 if 'LocalizationOnShoulder' in localizationOfPain else 0 )

            inputs.append(0 if unilatOrBilat == "Unilateral" else 1)
            inputs.append(0 if awakeningAtNight == "No" else 1)
            inputs.append(0 if responseToMassage == "No" else 1)
            inputs.append(0 if historyOfArthritis == "No" else 1)
            inputs.append(0 if morningStiffness == "No" else 1)
            inputs.append(0 if limping == "No" else 1)
            inputs.append(0 if limitationOfActivities == "No" else 1)
            inputs.append(0 if familyHistoryOfGrowingPain == "No" else 1)
            inputs.append(0 if physicalExamination == "Normal" else 1)
            inputs.append(0 if APR == "Normal" else 1)
            inputs.append(BMI)

            # frequencyOfPain is a categorical variable
            inputs.append(1 if frequencyOfPain == "1 Day a Week" else 0)
            inputs.append(1 if frequencyOfPain == "1 Day in 2 Weeks" else 0)
            inputs.append(1 if frequencyOfPain == "1 Day in 2-3 Days" else 0)
            inputs.append(1 if frequencyOfPain == "1 Day in 2-3 Weeks" else 0)
            inputs.append(1 if frequencyOfPain == "1 Day in 6 Months" else 0)
            inputs.append(1 if frequencyOfPain == "1 Week a Month" else 0)
            inputs.append(1 if frequencyOfPain == "1-2 Days a Month" else 0)
            inputs.append(1 if frequencyOfPain == "1-2 Days a Week" else 0)
            inputs.append(1 if frequencyOfPain == "1-2 Days in 2 Weeks" else 0)
            inputs.append(1 if frequencyOfPain == "1-2 Days in 3-4 Months" else 0)
            inputs.append(1 if frequencyOfPain == "2 Days a Month" else 0)
            inputs.append(1 if frequencyOfPain == "2 Days a Week" else 0)
            inputs.append(1 if frequencyOfPain == "2-3 Days a Month" else 0)
            inputs.append(1 if frequencyOfPain == "2-3 Days a Week" else 0)
            inputs.append(1 if frequencyOfPain == "2-3 Days in 2 Weeks" else 0)
            inputs.append(1 if frequencyOfPain == "3 Days a Week" else 0)
            inputs.append(1 if frequencyOfPain == "3-4 Days a Month" else 0)
            inputs.append(1 if frequencyOfPain == "3-4 Days a Week" else 0)
            inputs.append(1 if frequencyOfPain == "4 Days a Week" else 0)
            inputs.append(1 if frequencyOfPain == "4-5 Days a Week" else 0)
            inputs.append(1 if frequencyOfPain == "5 Days a Month" else 0)
            inputs.append(1 if frequencyOfPain == "5 Days a Week" else 0)
            inputs.append(1 if frequencyOfPain == "5-6 Days a Week" else 0)
            inputs.append(1 if frequencyOfPain == "6 Days a Week" else 0)
            inputs.append(1 if frequencyOfPain == "Everyday" else 0)

            # assocWithTimeOfTheDay is a categorical variable
            if assocWithTimeOfTheDay == "Evening":
                inputs.append(0) # Morning
                inputs.append(0) # No Difference
            elif assocWithTimeOfTheDay == "Morning":
                inputs.append(1) # Morning
                inputs.append(0) # No Difference
            else:
                inputs.append(0) # Morning
                inputs.append(1) # No Difference

            # For debugging purposes
            # st.write(inputs)

            predicted_class, probas = predict_disease(inputs, scaler)
            # st.write(predicted_class)
            # st.write(probas)
            probas = [x * 100 for x in probas]
            predicted = True

    with right_button:
        if st.button("Clear"):
            session.run_id += 1

    if predicted:

        st.markdown("<br/>", unsafe_allow_html=True)

        probabilityOfGP = round(probas[0][1], 2)
        if probabilityOfGP < 50:
            st.info('It is **unlikely** that your patient has GP. Because, my calculations show **{}%** probability.'.format(str(probabilityOfGP)))
        elif probabilityOfGP <= 75 and probabilityOfGP >= 50:
            st.warning('Sorry, I must stay neutral. Your patient may or may not have GP. Because, my calculations show **{}%** probability for GP.'.format(str(probabilityOfGP)))
        elif probabilityOfGP <= 90 and probabilityOfGP > 75:
            st.success('It is **likely** that your patient may have GP. Because, my calculations show **{}%** probability.'.format(str(probabilityOfGP)))
        else:
            st.success('It is **very likely** that your patient has GP. Because, my calculations show **{}%** probability.'.format(str(probabilityOfGP)))

        st.error('By the way, do not forget that I am just a prototype! Don\'t take my word for it.')

if __name__ == '__main__':
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    main()
