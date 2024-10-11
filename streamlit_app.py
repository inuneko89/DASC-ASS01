import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import io

# Configure Gemini API
genai.configure(api_key='AIzaSyCCQumrGPGSzDgY7_YFSSI5kFzYb-WXFB4')
model = genai.GenerativeModel('gemini-pro')

# Set page config
st.set_page_config(page_title="Data Science Analysis Chatbot", layout="wide")
st.title("Data Science Analysis Chatbot with Gemini")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None

# Function to perform detailed data analysis
def analyze_dataset(df):
    analysis = []
    
    # Basic information
    analysis.append("📊 Basic Dataset Information:")
    analysis.append(f"- Rows: {df.shape[0]}")
    analysis.append(f"- Columns: {df.shape[1]}")
    analysis.append(f"- Memory Usage: {df.memory_usage().sum() / 1024:.2f} KB")
    
    # Data types
    analysis.append("\n📋 Column Data Types:")
    for col in df.columns:
        analysis.append(f"- {col}: {df[col].dtype}")
    
    # Missing values
    missing = df.isnull().sum()
    if missing.any():
        analysis.append("\n⚠️ Missing Values:")
        for col, count in missing[missing > 0].items():
            analysis.append(f"- {col}: {count} missing values ({count/len(df)*100:.2f}%)")
    
    # Numerical analysis
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        analysis.append("\n📈 Numerical Columns Analysis:")
        for col in numerical_cols:
            analysis.append(f"\n{col}:")
            analysis.append(f"- Mean: {df[col].mean():.2f}")
            analysis.append(f"- Median: {df[col].median():.2f}")
            analysis.append(f"- Std: {df[col].std():.2f}")
            analysis.append(f"- Min: {df[col].min():.2f}")
            analysis.append(f"- Max: {df[col].max():.2f}")
    
    # Categorical analysis
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        analysis.append("\n📊 Categorical Columns Analysis:")
        for col in categorical_cols:
            unique_count = df[col].nunique()
            analysis.append(f"\n{col}:")
            analysis.append(f"- Unique values: {unique_count}")
            if unique_count < 10:  # Only show value counts for columns with few unique values
                analysis.append("- Value counts:")
                for val, count in df[col].value_counts().items():
                    analysis.append(f"  • {val}: {count}")
    
    # Correlation analysis for numerical columns
    if len(numerical_cols) > 1:
        analysis.append("\n🔄 Strong Correlations:")
        corr_matrix = df[numerical_cols].corr()
        for i in range(len(numerical_cols)):
            for j in range(i+1, len(numerical_cols)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.5:  # Only show strong correlations
                    analysis.append(f"- {numerical_cols[i]} vs {numerical_cols[j]}: {corr:.2f}")
    
    return "\n".join(analysis)

# Function to generate data science insights
def generate_insights(df, analysis_text):
    prompt = f"""
    As a data scientist, analyze the following dataset and provide insights and recommendations:
    
    Dataset Analysis:
    {analysis_text}
    
    Please provide:
    1. Key insights from the data
    2. Potential issues or data quality concerns
    3. Recommended next steps for deeper analysis
    4. Suggested visualizations that might be useful
    5. Possible machine learning approaches if applicable
    
    Keep the response structured and focused on actionable insights.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating insights: {str(e)}"

# Enhanced plotting function
def create_advanced_plot(df, plot_type, x_col, y_col, title, color_col=None):
    try:
        if plot_type == 'line':
            fig = px.line(df, x=x_col, y=y_col, title=title, color=color_col)
        elif plot_type == 'bar':
            fig = px.bar(df, x=x_col, y=y_col, title=title, color=color_col)
        elif plot_type == 'scatter':
            fig = px.scatter(df, x=x_col, y=y_col, title=title, color=color_col)
        elif plot_type == 'box':
            fig = px.box(df, x=x_col, y=y_col, title=title, color=color_col)
        elif plot_type == 'violin':
            fig = px.violin(df, x=x_col, y=y_col, title=title, color=color_col)
        elif plot_type == 'histogram':
            fig = px.histogram(df, x=x_col, title=title, color=color_col)
        
        fig.update_layout(
            template='plotly_white',
            title_x=0.5,
            margin=dict(t=100),
            height=500
        )
        return fig
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        return None

# File upload
uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df
    
    # Data Preview
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Automatic Analysis
    st.subheader("Automatic Data Analysis")
    analysis_text = analyze_dataset(df)
    with st.expander("View Detailed Analysis", expanded=True):
        st.markdown(analysis_text)
    
    # Generate Insights
    st.subheader("Data Science Insights")
    with st.expander("View AI-Generated Insights", expanded=True):
        insights = generate_insights(df, analysis_text)
        st.markdown(insights)

# Chat interface with enhanced data science capabilities
with st.container():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me about your data..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            if st.session_state.df is not None:
                df = st.session_state.df
                # Create context with data analysis
                analysis_summary = analyze_dataset(df)
                context = f"""
                I am a data science assistant. Here's the current context:
                
                Data Analysis Summary:
                {analysis_summary}
                
                User Question: {prompt}
                
                Please provide a detailed data science perspective in your response.
                """
                
                response = model.generate_content(context).text
                
                # Check for visualization requests
                if any(keyword in prompt.lower() for keyword in ['plot', 'graph', 'chart', 'visualize', 'show']):
                    try:
                        numerical_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numerical_cols) >= 2:
                            st.subheader("Suggested Visualization")
                            fig = create_advanced_plot(
                                df, 
                                'scatter', 
                                numerical_cols[0], 
                                numerical_cols[1], 
                                "Data Visualization"
                            )
                            st.plotly_chart(fig)
                    except Exception as e:
                        st.error(f"Error creating visualization: {str(e)}")
                
                st.markdown(response)
            else:
                st.markdown("Please upload a CSV file first to analyze data.")
                
        st.session_state.messages.append({"role": "assistant", "content": response})

# Enhanced sidebar with more analysis options
with st.sidebar:
    st.header("Advanced Data Analysis Options")
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Data filtering
        st.subheader("Data Filtering")
        selected_columns = st.multiselect("Select columns to analyze", df.columns.tolist())
        
        if selected_columns:
            filtered_df = df[selected_columns]
            
            # Advanced plotting options
            st.subheader("Create Advanced Plot")
            plot_type = st.selectbox(
                "Select Plot Type", 
                ['scatter', 'line', 'bar', 'box', 'violin', 'histogram']
            )
            
            x_col = st.selectbox("Select X-axis", filtered_df.columns)
            
            if plot_type != 'histogram':
                y_col = st.selectbox("Select Y-axis", filtered_df.columns)
            else:
                y_col = None
                
            color_col = st.selectbox("Select Color Variable (optional)", 
                                   ['None'] + filtered_df.columns.tolist())
            
            if st.button("Generate Advanced Plot"):
                if color_col == 'None':
                    color_col = None
                    
                fig = create_advanced_plot(
                    filtered_df, 
                    plot_type, 
                    x_col, 
                    y_col, 
                    f"{plot_type.capitalize()} Plot",
                    color_col
                )
                if fig:
                    st.plotly_chart(fig)

# Footer
st.markdown("---")
st.markdown("🤖 Powered by Gemini AI | 📊 Built for Data Science Analysis")