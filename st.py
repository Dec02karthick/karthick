import streamlit as st

st.ttitle('Iris predictions')

st.write('Iris data predictions')

sl = st.number_input('Enter the sepal length value')
sw = st.number_input('Enter the sepal width value')
pl = st.number_input('Enter the petal length value')
pw = st.number_input('Enter the petal width value')

if st.button('predict?')

with open('iris_model.pkl','rb')as model:
    main_model = pickle.load(model)

datas = [[sl,sw,pl,pw]]
preds = main_model.predict(datas)
st.write(f'The predicted value is {preds}')
