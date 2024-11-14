import streamlit as st
import pandas as pd
import os
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from PIL import Image, ImageDraw, ImageOps
import numpy as np

def create_circular_mask(image_path, size):
    # Open and resize image
    img = Image.open(image_path)
    img = img.resize((size, size))
    
    # Create a circular mask
    mask = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size, size), fill=255)
    
    # Apply the mask
    output = ImageOps.fit(img, mask.size, centering=(0.5, 0.5))
    output.putalpha(mask)
    
    return output

def main():

    # Set up the customization options
    st.sidebar.title('Customization')
    model = st.sidebar.selectbox(
        'Choose a model',
        ['groq/llama3-8b-8192']
    )

    llm = ChatGroq(
            temperature=0, 
            groq_api_key = st.secrets["GROQ_API_KEY"], 
            model_name=model
        )

    # Streamlit UI
    

    # Display the David emoji
    spacer, col = st.columns([1,5])  
    with col: 
        st.title('GourmetPress Editor')
        st.text("by Chef David")
        circular_image = create_circular_mask('./static/david-emoji.jpg', 200)
        st.image(circular_image, width=200)

    multiline_text = """
    <style>
        @import url('https://fonts.googleapis.com/css?family=Gowun Batang,Tangerine|Inconsolata|Droid+Sans');
        [data-testid="stSidebar"] {
            background-color: #f0f0f0;
        }
        .appview-container {
            background: linear-gradient(300deg, #ff0037, #c98b1d);
            font-family: 'Gowun Batang', serif;
            color: white;
            font-size: 1.5em;
        }
    </style>
    <p>
    David's Gourmet Press is a specialized cookbook publishing application designed to empower chefs in crafting personalized, stylized recipe collections. This intuitive platform guides culinary artists through the process of compiling their signature dishes alongside the rich stories that inspire them. By seamlessly blending functionality with aesthetic appeal, the app allows chefs to:
    </p>
    <ul>
        <li>Curate a selection of their most prized recipes</li>
        <li>Narrate the personal anecdotes and cultural significance behind each dish</li>
        <li>Enhance their cookbook with high-quality food photography</li>
        <li>Choose from a variety of professionally designed layouts and typography options</li>
        <li>Collaborate with editors and co-authors in real-time</li>
    <li>Export their creations in both digital and print-ready formats</li>
    </ul>

    Whether you're a Michelin-starred chef or a passionate home cook, David's GourmetPress provides the tools to transform your culinary expertise into a beautifully crafted, small-scale cookbook that captures not just recipes, but the heart and soul of your cooking journey. With this app, every dish tells a story, and every chef becomes an author.
    """

    st.markdown(multiline_text, unsafe_allow_html=True)

    Story_Agent = Agent(
        role='Story_Agent',
        goal="""Develop a comprehensive narrative about a given topic, covering its history,
            background information, relation to the author, and sources of information.""",
        backstory="""You are an expert storyteller and researcher, skilled in crafting engaging 
            narratives that blend historical context, factual information, personal connections, 
            and credible sources. Your mission is to create a well-rounded story about any given 
            topic, ensuring it's informative, relatable, and well-supported.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    Recipe_Agent = Agent(
        role='Recipe_Agent',
        goal="""Create comprehensive recipe guides including the story, ingredients, method, 
            nutritional information, common pitfalls, useful tips, and an AI-generated image for each recipe.""",
        backstory="""You are a master chef and culinary historian with a passion for storytelling 
            and a keen eye for detail. Your expertise spans across various cuisines, cooking techniques, 
            and food science. You have a knack for explaining complex culinary concepts in simple terms
            and providing practical advice to home cooks. Your mission is to create detailed, engaging, 
            and informative recipe guides that not only teach how to prepare a dish but also provide 
            context, nutritional awareness, and helpful insights.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
        
    Summarization_Agent = Agent(
         role='Summarization_Agent',
         goal="""Summarize findings from each of the previous steps of the topic definition.
             Include all findings from the topic and four relative recipes.
             """,
         backstory="""You are a seasoned food chef, able to break down various cuisines and recipes for
             less experienced chefs, provide valuable insight into the recipes   and why certain cuisines
             are appropriate, and write good, simple recipes to help deliver quality content.
             """,
         verbose=True,
         allow_delegation=False,
         llm=llm,
    )

    user_topic = st.text_input("What is the topic of your story?")
    data_upload = False
    uploaded_file = st.file_uploader("Upload a sample .csv of your data (optional)")
    if uploaded_file is not None:
        try:
            # Attempt to read the uploaded file as a DataFrame
            df = pd.read_csv(uploaded_file).head(5)
            
            # If successful, set 'data_upload' to True
            data_upload = True
            
            # Display the DataFrame in the app
            st.write("Data successfully uploaded and read as DataFrame:")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error reading the file: {e}")

    if user_topic:
        task_define_topic = Task(
        description="""Clarify and define the user's topic, 
            including identifying the ingredients type and specific requirements.
            
            Here is the user's choice of topic:

            {user_topic}
            """.format(user_topic=user_topic),
            agent=Story_Agent,
            expected_output="A clear and concise definition of the user's topic, including the ingredients type and specific requirements."
        )

       #if data_upload:
            #task_assess_data = Task(
                #description="""Evaluate the user's data for quality and suitability, 
                #suggesting preprocessing or augmentation steps if needed.
                
                #Here is a sample of the user's data:

                #{df}

                #The file name is called {uploaded_file}
                
               #""".format(df=df.head(),uploaded_file=uploaded_file),
                #agent=Data_Assessment_Agent,
               # expected_output="An assessment of the data's quality and suitability, with suggestions for preprocessing or augmentation if necessary."
            #)
        #else:
            #task_assess_data = Task(
            #    description="""The user has not uploaded any specific data for this problem,
            #    but please go ahead and consider a hypothetical dataset that might be useful
            #    for their machine learning problem. 
            #    """,
            #    agent=Data_Assessment_Agent,
            #    expected_output="A hypothetical dataset that might be useful for the user's machine learning problem, along with any necessary preprocessing steps."
            #)

        task_story_model = Task(
            description="""Research a variety of cuisines relative to the topic (ingredients) submitted by the user.""",
            agent=Story_Agent,
            expected_output="Provide a list of cuisines that are relative to the topic (ingredients) submitted by the user."
        )
        
        task_recipe_model = Task(
            description="""Suggest suitable recipes for the defined topic providing rationale for each suggestion.""",
            agent=Recipe_Agent,
            expected_output="A list of suitable recipes for the defined topic, along with the rationale for each suggestion."
        )

        task_summarize_model = Task(
            description="""
            Summarize the results of the topic definition.
            Keep the summarization brief and don't forget to share at least one recipe!
            """,
            agent=Summarization_Agent,
            expected_output="A concise summary of the topic definition results including at least one recipe."
        )


        crew = Crew(
            agents=[Story_Agent, Recipe_Agent, Summarization_Agent],
            tasks=[task_define_topic, task_story_model, task_recipe_model, task_summarize_model],
            verbose=True
        )

        result = crew.kickoff()

        # Display the result in Streamlit
        st.markdown("### Results")
        st.markdown(result) 


if __name__ == "__main__":
    main()


# Problem_Definition_Agent = Agent(
    #    role='Problem_Definition_Agent',
    #    goal="""clarify the machine learning problem the user wants to solve, 
    #        identifying the type of problem (e.g., classification, regression) and any specific requirements.""",
    #    backstory="""You are an expert in understanding and defining machine learning problems. 
    #        Your goal is to extract a clear, concise problem statement from the user's input, 
    #        ensuring the project starts with a solid foundation.""",
    #    verbose=True,
    #    allow_delegation=False,
    #    llm=llm,
    #)

    #Data_Assessment_Agent = Agent(
    #    role='Data_Assessment_Agent',
    #    goal="""evaluate the data provided by the user, assessing its quality, 
    #        suitability for the problem, and suggesting preprocessing steps if necessary.""",
    #    backstory="""You specialize in data evaluation and preprocessing. 
    #        Your task is to guide the user in preparing their dataset for the machine learning model, 
    #        including suggestions for data cleaning and augmentation.""",
    #    verbose=True,
    #    allow_delegation=False,
    #    llm=llm,
    #)

    #Model_Recommendation_Agent = Agent(
    #    role='Model_Recommendation_Agent',
    #    goal="""suggest the most suitable machine learning models based on the problem definition 
    #        and data assessment, providing reasons for each recommendation.""",
    #    backstory="""As an expert in machine learning algorithms, you recommend models that best fit 
    #        the user's problem and data. You provide insights into why certain models may be more effective than others,
    #        considering classification vs regression and supervised vs unsupervised frameworks.""",
    #    verbose=True,
    #    allow_delegation=False,
    #    llm=llm,
    #)


    #Starter_Code_Generator_Agent = Agent(
    #    role='Starter_Code_Generator_Agent',
    #    goal="""generate starter Python code for the project, including data loading, 
    #         model definition, and a basic training loop, based on findings from the problem definitions,
    #         data assessment and model recommendation""",
    #     backstory="""You are a code wizard, able to generate starter code templates that users 
    #         can customize for their projects. Your goal is to give users a head start in their coding efforts.""",
    #     verbose=True,
    #     allow_delegation=False,
    #     llm=llm,
    #)


        #task_generate_code = Task(
        #description="""Generate starter Python code tailored to the user's project using the model recommendation agent's recommendation(s), 
        #    including snippets for package import, data handling, model definition, and training
        #     """,
        # agent=Starter_Code_Generator_Agent,
        # expected_output="Python code snippets for package import, data handling, model definition, and training, tailored to the user's project, plus a brief summary of the problem and model recommendations."
        # )