import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
cf.go_offline()
from chart_studio import plotly
import plotly.express as px
import plotly.figure_factory as ff
import time


# Set page configuration
st.set_page_config(page_title='My Project', page_icon=':tada:', layout='wide')
st.sidebar.write("## STUDENT PERFORMANCE ANALYSIS SYSTEM")
st.sidebar.image("images/student.jpg", use_column_width=True)


# Program Begins...
def main():
    with st.container():
        st.subheader("Machine Learning Project")
        st.title("STUDENT PERFORMANCE ANALYSIS SYSTEM")
        st.write("Performance evaluation of students is essential to check the feasibility of improvement. Regular evaluation not only improves the performance of the student but also it helps in understanding where the student is lacking. It takes a lot of manual effort to complete the evaluation process as even one college may contain thousands of students. This paper proposed an automated solution for the performance evaluation of the students using machine learning.")

    st.subheader("Upload the file in the format displayed below...")
    # displaying user demo model to upload the csv file format
    demo = pd.read_csv("datasets/csv/testing_data.csv")
    # demo = demo.drop(['Unnamed: 0'],axis=1)
    st.dataframe(demo.head())
    
    # Taking dataset from the user
    # Allow only .csv and .xlsx files to be uploaded
    st.subheader("Upload your (csv) file here")
    uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx"])


    # Starting the countdown
    col1, col2 = st.columns(2) 
    with col1:  
        st.empty()
    with col2:
        if uploaded_file:
            placeholder = st.empty()
            with placeholder.container():
                N = 5
                readyin = False
                for sec in range(N, -1, -1):
                    ss = sec % 60
                    # st.write("Analysing your data please wait...")
                    placeholder.metric("**Analysing your data please wait...**", f"{ss:02d}")
                    time.sleep(1)

            if ss == 0:
                #This would empty everything inside the container
                placeholder.empty()
                st.balloons()
      
            
    # Check if file was uploaded
    if uploaded_file:
        # Check MIME type of the uploaded file
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    
        df = df[['regno','rollno', 'name', 'gender',  'age', 'internet','address', 'higher', 'studytime', 'absences', 'traveltime', 'freetime', 'failures','AI1','AI2','CG1','CG2','CN1','CN2','WEB1','WEB2','MIS1','MIS2']]
        st.subheader('Uploaded data')
        st.dataframe(df)
        
        st.write("## Insights on your data...")
        
        with st.container():
            st.subheader("Age of the students:")
            col1, col2 = st.columns(2)
            with col1:
                fig = plt.figure(figsize=(10,4))
                sns.countplot(x='age', data=df)
                st.pyplot(fig)

            with col1:
                fig = plt.figure(figsize=(10,4))
                sns.kdeplot(df['age'])
                st.pyplot(fig)
            
            with col1:
                st.subheader("Number of Failures")
                fig = plt.figure(figsize=(10,4))
                sns.countplot(x='failures', data=df)
                st.pyplot(fig)
                
            with col1:
                st.subheader("Male and Female students")
                fig = plt.figure(figsize=(10,4))
                sns.set_style('whitegrid')
                sns.countplot(x='gender', data=df, palette='plasma')
                st.pyplot(fig)
                
            with col1:
                st.subheader("Students prefer for Higher studies")
                fig = plt.figure(figsize=(10,4))
                sns.set_style('whitegrid')
                sns.countplot(x='higher', data=df, palette='plasma')
                st.pyplot(fig)
            
            
            col3, col4 = st.columns(2)
            with col3:
                st.subheader('Number of Male & Female students in different age groups')
                fig = plt.figure(figsize=(10,4))
                b = sns.countplot(x='age',hue='gender', data=df, palette='inferno')
                b.axes.set_title('Number of Male & Female students in different age groups')
                b.set_xlabel("Age")
                b.set_ylabel("Count")
                st.pyplot(fig)
            
            with col3:
                st.subheader('Students residing in Urban and Rural areas')
                fig = plt.figure(figsize=(10,4))
                sns.set_style('whitegrid')
                sns.countplot(x='address',data=df,palette='magma') 
                st.pyplot(fig)
            
            with col3:
                st.subheader('Student scores with respect to their addresses')
                fig = plt.figure(figsize=(10,4))
                sns.countplot(x='address',hue='AI2',data=df,palette='Oranges') 
                st.pyplot(fig)
            
            
            col5, col6 = st.columns(2)
            with col5:
                st.subheader("Do Urban students perform better than Rural students?")
                fig = plt.figure(figsize=(10,4))
                sns.kdeplot(df.loc[df['address'] == 'U', 'CG1'], label='Urban', shade = True)
                sns.kdeplot(df.loc[df['address'] == 'R', 'CG1'], label='Rural', shade = True)
                plt.title('Do urban students score higher than rural students?')
                plt.xlabel('Grade')
                plt.ylabel('Density')
                st.pyplot(fig)
            
            with col5:
                st.subheader("Failures attribute")
                fig = plt.figure(figsize=(10,4))
                b = sns.swarmplot(x=df['failures'],y=df['MIS2'],palette='autumn')
                b.axes.set_title('Previous Failures vs Final Grade')
                st.pyplot(fig)
            
            
            with col5:
                st.subheader("Wish to go for Higher Education Attribute")
                fig = plt.figure(figsize=(10,4))
                b = sns.boxplot(x=df['higher'],y=df['MIS2'],palette='binary')
                b.axes.set_title('Higher Education vs Final Grade')
                st.pyplot(fig)
            
            with col5:
                st.subheader("Students with Internet facility")
                fig = plt.figure(figsize=(10,4))
                sns.kdeplot(df.loc[df['internet'] == 'yes', 'age'], label='Yes', shade = True)
                sns.kdeplot(df.loc[df['internet'] == 'no', 'age'], label='No', shade = True)
                st.pyplot(fig)
                
                
            st.subheader('Does age affect grades?')
            age1, age2 = st.columns(2)
            with age1:
                fig = plt.figure(figsize=(10,4))
                b= sns.boxplot(x='age', y='WEB2',data=df,palette='gist_heat')
                b.axes.set_title('Age vs Final Grade')
                st.pyplot(fig)
            with age2:
                fig = plt.figure(figsize=(10,4))
                b = sns.swarmplot(x='age', y='CN1',hue='gender', data=df,palette='PiYG')
                b.axes.set_title('Does age affect final grade?')
                st.pyplot(fig)
                
                
            # Students marks comparison
            st.subheader("Subjects wise marks distribution")
            tab1, tab2, tab3, tab4, tab5 = st.tabs(['Artificial Intelligence', 'Computer Networks', 'Management Information System', 'WEB Technology', 'Computer Graphics'])
            with tab1:
                st.subheader("Artificial Intelligence")    
                ai1, ai2 = st.columns(2)
                with ai1:
                    fig = plt.figure(figsize=(10,4))
                    sns.distplot(df['AI1'])
                    st.pyplot(fig)
                    
                with ai2:
                    fig = plt.figure(figsize=(10,4))
                    sns.distplot(df['AI2'])
                    st.pyplot(fig)
                
            with tab2:
                st.subheader("Computer Networks")    
                cn1, cn2 = st.columns(2)
                with cn1:
                    fig = plt.figure(figsize=(10,4))
                    sns.distplot(df['CN1'], color='purple')
                    st.pyplot(fig)
                    
                with cn2:
                    fig = plt.figure(figsize=(10,4))
                    sns.distplot(df['CN2'], color='purple')
                    st.pyplot(fig)
              
            with tab3:    
                st.subheader("Management Information System")    
                mis1, mis2 = st.columns(2)
                with mis1:
                    fig = plt.figure(figsize=(10,4))
                    sns.distplot(df['MIS1'], color='violet')
                    sns.color_palette('husl', 8)
                    st.pyplot(fig)
                    
                with mis2:
                    fig = plt.figure(figsize=(10,4))
                    sns.distplot(df['MIS2'], color='violet')
                    st.pyplot(fig)
                
            with tab4:
                st.subheader("WEB Technology")    
                web1, web2 = st.columns(2)
                with web1:
                    fig = plt.figure(figsize=(10,4))
                    sns.distplot(df['WEB1'], color='red')
                    st.pyplot(fig)
                    
                with web2:
                    fig = plt.figure(figsize=(10,4))
                    sns.distplot(df['WEB2'], color='red')
                    st.pyplot(fig)
                
            with tab5:
                st.subheader("Computer Graphics")    
                cg1, cg2 = st.columns(2)
                with cg1:
                    fig = plt.figure(figsize=(10,4))
                    sns.distplot(df['CG1'], color='green')
                    st.pyplot(fig)
                    
                with cg2:
                    fig = plt.figure(figsize=(10,4))
                    sns.distplot(df['CG2'], color='green')
                    st.pyplot(fig)   
           
            
if __name__ == '__main__':
    main()
