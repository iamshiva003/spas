# Importing the modules
import streamlit as st
from PIL import Image
import requests
from streamlit_lottie import st_lottie


# setting page configurations
st.set_page_config(page_title='My Project', page_icon=':tada:', layout='wide')
st.sidebar.write("## STUDENT PERFORMANCE ANALYSIS SYSTEM")
st.balloons()


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# ---- LOAD ASSETS ----
lottie_coding = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_q5qeoo3q.json")
img_content_form = Image.open("images/download.jpg")
st.sidebar.image("images/student.jpg", use_column_width=True)


def main():
    with st.container():
        st.subheader("Machine Learning Project")
        st.title("STUDENT PERFORMANCE ANALYSIS SYSTEM")
        st.write("Performance evaluation of students is essential to check the feasibility of improvement. Regular evaluation not only improves the performance of the student but also it helps in understanding where the student is lacking. It takes a lot of manual effort to complete the evaluation process as even one college may contain thousands of students. This paper proposed an automated solution for the performance evaluation of the students using machine learning.")
        

    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.header("Machine Learning Models")
            st.write("1. Linear Regression")
            st.write("2. Random Forest")
            st.write("3. ElasticNet")
            st.write("4. ExtraTreesRegressor")
            st.write("5. SVR")
            st.write("6. GradientBoostingRegressor")       
        with right_column:
            st_lottie(lottie_coding, height=300, key="coding")
    
    with st.container():
        st.write("---")
        st.header("Model Descriptions")
        st.subheader("Linear Regresion")
        st.write("##")
        image_column, text_column = st.columns((1, 2))
        with image_column:
            st.image(img_content_form)
        with text_column:
            st.write("**Description**")
            st.write("""
                    Linear regression analysis is used to predict the value of a variable based on the value of 
                    another variable. The variable you want to predict is called the dependent variable. The 
                    variable you are using to predict the other variable's value is called the independent variable.    
            """)
            st.markdown("[Learn More...](https://www.ibm.com/in-en/topics/linear-regression)")
    
    with st.container():
        st.subheader("Random Forest")
        image_column, text_column = st.columns((1, 2))
        with image_column:
            st.image(Image.open("images/random_forest.jpg"))
        with text_column:
            st.write("**Description**")
            st.write("""
                    As the name suggests, "Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset." Instead of relying on one decision tree, the random forest takes the prediction from each tree and based on the majority votes of predictions, and it predicts the final output.
                    The greater number of trees in the forest leads to higher accuracy and prevents the problem of overfitting.
            """)
            st.markdown("[Learn More...](https://www.javatpoint.com/machine-learning-random-forest-algorithm)")
    
    with st.container():
        st.subheader("Elastic Net")
        image_column, text_column = st.columns((1, 2))
        with image_column:
            st.image(Image.open("images/elasticnet.jpg"))
        with text_column:
            st.write("**Description**")
            st.write("""
                    Elastic net linear regression uses the penalties from both the lasso and ridge techniques to regularize regression models. The technique combines both the lasso and ridge regression methods by learning from their shortcomings to improve the regularization of statistical models.
            """)
            st.markdown("[Learn More...](https://corporatefinanceinstitute.com/resources/data-science/elastic-net/)")
    
    with st.container():
        st.subheader("Extra Trees Regressor")
        image_column, text_column = st.columns((1, 2))
        with image_column:
            st.image(Image.open("images/extra_trees_regressor.jpg"))
        with text_column:
            st.write("**Description**")
            st.write("""
                    The extra trees algorithm, like the random forests algorithm, creates many decision trees, but the sampling for each tree is random, without replacement. This creates a dataset for each tree with unique samples. A specific number of features, from the total set of features, are also selected randomly for each tree. The most important and unique characteristic of extra trees is the random selection of a splitting value for a feature. Instead of calculating a locally optimal value using Gini or entropy to split the data, the algorithm randomly selects a split value. This makes the trees diversified and uncorrelated.
            """)
            st.markdown("[Learn More...](https://pro.arcgis.com/en/pro-app/latest/tool-reference/geoai/how-extra-tree-classification-and-regression-works.htm)")
    
    with st.container():
        st.subheader("Support Vector Regression(SVR)")
        image_column, text_column = st.columns((1, 2))    
        with image_column:
            st.image(Image.open("images/svr.jpg"))
        with text_column:
            st.write("**Description**")
            st.write("""
                    Support Vector Regression as the name suggests is a regression algorithm that supports both linear and non-linear regressions. This method works on the principle of the Support Vector Machine. SVR differs from SVM in the way that SVM is a classifier that is used for predicting discrete categorical labels while SVR is a regressor that is used for predicting continuous ordered variables.
            """)
            st.markdown("[Learn More...](https://www.educba.com/support-vector-regression/)")
    
    with st.container():
        st.subheader("Gradient Boosting Regressor")
        image_column, text_column = st.columns((1, 2))
        with image_column:
            st.image(Image.open("images/gradient_boosting_regressor.jpg"))
        with text_column:
            st.write("**Description**")
            st.write("""
                    Gradient boosting builds an additive mode by using multiple decision trees of fixed size as weak learners or weak predictive models. The parameter, n_estimators, decides the number of decision trees which will be used in the boosting stages. Gradient boosting differs from AdaBoost in the manner that decision stumps (one node & two leaves) are used in AdaBoost whereas decision trees of fixed size are used in Gradient Boosting.
            """)
            st.markdown("[Learn More...](https://vitalflux.com/gradient-boosting-regression-python-examples/)")


if __name__ == "__main__":
    main()