""""
I think pyspark is not needed because we re working with the lightml pipelien 


"""

# user input lenaaa 
# input me say entities extract karna along with the indices 

import streamlit as st 
import sparknlp
from pyspark.ml import Pipeline
from sparknlp.base import LightPipeline

if 'spark_engine' not in st.session_state:
    st.session_state['spark_engine'] = sparknlp.start()

@st.cache_resource
def init_model():
    pipeline = Pipeline().load("./entity_extraction")
    sample_df = st.session_state['spark_engine'].createDataFrame([[""]]).toDF("text")
    model = LightPipeline(pipeline.fit(sample_df))
    return model

model = init_model()

st.header('Search Engine')
query = st.text_input('Enter your query')

if query:
    response = model.fullAnnotate(query)
    for entity in response[0]['entity']:
        st.markdown(f"**{entity.result}** from {entity.begin} to {entity.end}")