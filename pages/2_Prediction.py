# Standard ML Models for comparison
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

# Splitting data into training/testing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error

# Data Handling
import pandas as pd
import numpy as np

# Displaying the data in webpage
from PIL import Image
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import streamlit as st
import matplotlib.pyplot as plt


# Set page configuration
st.set_page_config(page_title='My Project', page_icon=':tada:', layout='wide')
st.sidebar.write("## STUDENT PERFORMANCE ANALYSIS SYSTEM")
st.sidebar.image("images/student.jpg", use_column_width=True)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        # Consolas,'Courier New',monospace,Hack,'cascadia code'

def main():
    with st.container():
        st.subheader("Machine Learning Project")
        st.title("STUDENT PERFORMANCE ANALYSIS SYSTEM")
        st.write("Performance evaluation of students is essential to check the feasibility of improvement. Regular evaluation not only improves the performance of the student but also it helps in understanding where the student is lacking. It takes a lot of manual effort to complete the evaluation process as even one college may contain thousands of students. This paper proposed an automated solution for the performance evaluation of the students using machine learning.")

    st.title("Student Grade Prediction")

    # Load the data into a Pandas dataframe
    training_data = pd.read_csv("datasets/csv/training_data.csv", sep=';')
    
    # Training the Machine learning model
    training_data = training_data[['G1', 'G2', 'G3', 'absences', 'failures', 'studytime']]
    predict = 'G3'
    X = np.array(training_data.drop(['G3'], axis=1))
    y = np.array(training_data[predict])

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Selecting the subject
    st.subheader("Select the Subject")
    subject = st.selectbox("Select below",
                 ['Artificial Intelligence', 
                  'Computer Graphics', 
                  'Computer Networks', 
                  'Management Information System', 
                  'Web Technology']
    )

    # choosing the Machine learning model for the prediction
    st.subheader("Select the Model for prediction")
    choose = st.selectbox("Select below",
               ['LinearRegression', 
                'RandomForestRegressor', 
                'ElasticNet', 
                'ExtraTreesRegressor', 
                'SVR', 
                'GradientBoostingRegressor']                 
    )

    # Train the model using different Machine Learning algorithms
    match choose:
        case 'LinearRegression':
            model = LinearRegression()
            model.fit(X_train, y_train)
        case 'RandomForestRegressor':
            model = RandomForestRegressor(n_estimators=100, random_state=0)
            model.fit(X, y)   
        case 'ElasticNet':
            model = ElasticNet(alpha=1.0, l1_ratio=0.5)
            model.fit(X_train, y_train)
        case 'ExtraTreesRegressor':
            model = ExtraTreesRegressor(n_estimators=100)
            model.fit(X_train, y_train)
        case 'SVR':
            model = SVR(kernel='rbf', degree=3, C=1.0, gamma='auto')
            model.fit(X_train, y_train)
        case 'GradientBoostingRegressor':
            model = GradientBoostingRegressor(n_estimators=50)
            model.fit(X_train, y_train)
        
        
    # Evaluate the model
    st.subheader("Accuracy of the model:")
    train_accuracy = int(model.score(X_train, y_train) * 100)
    test_accuracy = int(model.score(X_test, y_test) * 100)
    
    # Display the accuracy
    st.write(f"Train accuracy: {train_accuracy}%")
    st.write(f"Test accuracy: {test_accuracy}%")

    match subject:
        case 'Artificial Intelligence':
            data = pd.read_csv("datasets/csv/ai.csv")
            predict = 'AI3'
        case 'Computer Graphics':
            data = pd.read_csv("datasets/csv/cg.csv")
            predict = 'CG3'
        case 'Computer Networks':
            data = pd.read_csv("datasets/csv/cn.csv")
            predict = 'CN3'
        case 'Management Information System':
            data = pd.read_csv("datasets/csv/mis.csv")
            predict = 'MIS3'
        case 'Web Technology':
            data = pd.read_csv("datasets/csv/web.csv")
            predict = 'WEB3'

    # Displaying the predicted results
    sub1 = data.columns[0]
    sub2 = data.columns[1]

    st.subheader("Student Grades")  
    array = data.to_numpy()
    new_data = np.array(array)
    predictions = model.predict(new_data)
    st.write("Select the number of students to display")
    length = st.slider('Pick', 0, len(predictions))
        
    predictions = pd.DataFrame(predictions)
    test = pd.read_csv("datasets/csv/testing_data.csv")
    
    test = test[['regno', 'name','age', 'gender', sub1, sub2]]
    result = pd.concat([test, predictions], axis=1, join='inner')
    result.rename(columns={0:'Prediction'}, inplace=True)   
        # st.dataframe(result[:length])
    st.table(result[:length])
    
    with st.container():
        st.write("##")
        st.write("##")
        st.write("##")
        st.header("Contact")
        # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
        contact_form = """
        <form action="https://formsubmit.co/shivu9887@gmail.com" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="text" name="name" placeholder="Your name" required>
            <input type="email" name="email" placeholder="Your email" required>
            <textarea name="message" placeholder="Your message here" required></textarea>
            <button type="submit">Send</button>
        </form>
        """
        local_css("style/style.css")
        left_column, right_column = st.columns(2)
        with left_column:
            st.markdown(contact_form, unsafe_allow_html=True)
        with right_column:
            st.empty()
    

    
    # tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Linear Regression', 'Random Forest', 'ElasticNet', 'ExtraTreesRegressor', 'SVR', 'GradientBoostingRegressor'])
    # with tab1: 
    #     x_test = X_test.values[:,0].reshape(-1, 1)
    #     y_pred = model.predict(x_test)
    #     fig = plt.figure(figsize=(10,4))
    #     plt.scatter(x_test, y_test,  color='black')
    #     plt.plot(x_test, y_pred, color='blue', linewidth=3)
    #     st.pyplot(fig)

    # with tab2: 
    #     st.help(RandomForestRegressor)
    # with tab3: 
    #     st.help(ElasticNet)
    # with tab4: 
    #     st.help(ExtraTreesRegressor)
    # with tab5: 
    #     st.help(SVR)
    # with tab6: 
    #     st.help(GradientBoostingRegressor)

    # def display(selected1, selected2):
    #     name = f"""**{selected1} vs {selected2}**"""
    #     st.write(name)

    # st.subheader("Model Comparison")
    # col1, col2 = st.columns(2)
    # # display(selected1, selected2)
    # with col1:
    #     selected1 = st.radio(
    #         "**Select the Model 1**",
    #         ["Linear Regression",
    #          "Random Forest Regressor",
    #          "Elastic Net",
    #          "Extra Trees Regressor",
    #          "SVR",
    #          "Gradient Boosting Regressor"
    #         ]      
    #     )
    #     st.subheader("Accuracy of the model:")
    #     st.write(f"**Train accuracy: {model.score(X_train, y_train):.3f}**")
    #     st.write(f"**Test accuracy: {model.score(X_test, y_test):.3f}**")

    
    # with col2:
    #     selected2 = st.radio(
    #         "**Select the Model 2**",
    #         ["Linear Regression",
    #          "Random Forest Regressor",
    #          "Elastic Net",
    #          "Extra Trees Regressor",
    #          "SVR",
    #          "Gradient Boosting Regressor"
    #         ]      
    #     )
    #     st.subheader("Accuracy of the model:")
    #     st.write(f"**Train accuracy: {model.score(X_train, y_train):.3f}**")
    #     st.write(f"**Test accuracy: {model.score(X_test, y_test):.3f}**")
    


if __name__== "__main__":
    main()
