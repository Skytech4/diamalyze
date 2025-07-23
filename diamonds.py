import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import pickle
from tensorflow.keras.models import load_model
import os
import requests


@st.cache_data
def load_data(dataset):
    df = pd.read_csv(dataset)
    return df

def file_download(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="diamonds.csv">Download CSV File</a>'
    return href


def predict_cut(carat, depth, table, price, x, y, z, new_color, new_clarity):
    try:
        with open('Nettoyage/model_diamonds.pkl', 'rb') as file:
                        loaded_model = pickle.load(file)

        nouvelles_donnees = pd.DataFrame({
            "carat": [carat],
            "depth": [depth],
            "table": [table],
            "price": [price],
            "x": [x],
            "y": [y],
            "z": [z],
            "new_color": [new_color],
            "new_clarity": [new_clarity]
        })

        prediction = loaded_model.predict(nouvelles_donnees)
        if prediction[0] == 0:
            return "Fair"
        elif prediction[0] == 1:
            return "Good"
        elif prediction[0] == 2:
            return "Ideal"
        elif prediction[0] == 3:
            return "Very Good"
        elif prediction[0] == 4:
            return "Premium"
        else:
            return "Unknown Prediction for these datas"
    except FileNotFoundError:
        st.error("❌ Machine Learning model file Unfounded")
    except Exception as e:
        st.error(f'⚠️ Error into the Machine Learning Prediction {str(e)}')


def predict_cut_dl(carat, depth, table, price, x, y, z, new_color, new_clarity):
    try:
        model = load_model("Nettoyage/model_ANN_diamonds.h5")

        data_input = np.array([[carat, depth, table, price, x, y, z, new_color, new_clarity]])
        prediction = model.predict(data_input)
        predicted_class = np.argmax(prediction)

        cut_mapping = {0: "Fair", 1: "Good", 2: "Ideal", 3: "Very Good", 4: "Premium"}
        return cut_mapping.get(predicted_class, "Unknown Prediction")

    except FileNotFoundError:
        st.error("❌ Deep Learning model file Unfounded.")
    except Exception as e:
        st.error(f'⚠️ Error into the Deep Learning Prediction : {str(e)}')



# Sidebar image path correction
st.sidebar.image("Ima_projet/diamond4.png", width=180)

def load_local_css(file_path):
    with open(file_path, "r") as f:
        css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

load_local_css("Css_projet/diamonds.css")

def main():

    # Menu options
    menu = ["Home", "Visualization And Analysis Dashboard", "Machine Learning", "Deep Learning", "ChatBot", "Groq"]
    choice = st.sidebar.selectbox("Menu", menu)  

    st.title('DIAMALYZE')

    if choice not in ["Visualization And Analysis Dashboard", "Machine Learning", "Deep Learning", "ChatBot", "Groq"]:
        st.subheader("Predicting Diamond's cut using Machine Learning and Deep Learning")
    
    # Load dataset
    data = load_data("Dataset/diamonds.csv")
    
    if choice == "Home":
        st.write("DIAMALYZE is an app that will analyze diamond data with some Python tools that can optimize decisions.")
        st.subheader("Diamond Information")
        st.write("Diamonds, often referred to as a symbol of luxury and status, are precious gemstones formed deep within the Earth's mantle under extreme pressure and temperature. Globally, the diamond industry is a multi-billion dollar sector, with significant production concentrated in countries like Russia, Botswana, Canada, and Australia. " \
                "These nations are known for their vast diamond mines and advanced extraction technologies. In recent years, the demand for ethically sourced diamonds has grown, leading to the rise of lab-grown diamonds and initiatives aimed at reducing conflict diamonds in trade. Major global organizations have established frameworks to ensure the traceability and ethical sourcing of diamonds.")
        
        st.image("Ima_projet/diamond_origin.jpg", use_container_width=True)

        st.markdown("")
        st.markdown("")
        st.markdown("")

        st.write("Cameroon, while not among the top global producers, has considerable potential in diamond mining. The country is rich in mineral resources, including diamonds, which are primarily found in the eastern regions, particularly in places like the Adamawa and East regions. The government has been working to attract foreign investment to develop its mining sector. " \
                "Challenges in the Cameroonian diamond industry include inadequate infrastructure, regulatory hurdles, and the need for better technology in mining operations. However, with efforts to improve governance and promote sustainable practices, Cameroon aims to harness its diamond resources more effectively. " \
                "In conclusion, while diamonds are a global phenomenon, Cameroon holds a unique position with the potential for growth in its diamond mining industry.")

        st.image("Ima_projet/diamond3.jpeg", use_container_width=True)


    elif choice == "Visualization And Analysis Dashboard":
        st.subheader("Dashboard of Analysis")
  
        cut_filter = st.selectbox("Select one category of cut", pd.unique(data['cut']))
        data = data[data['cut'] == cut_filter]


        avg_carat = np.mean(data['carat'])
        count_color = int(data[(data['color'] == 'E')]['color'].count())
        avg_table = np.mean(data['table'])

        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric(label="Average Carat ✨", value=round(avg_carat), delta=round(avg_carat))
        kpi2.metric(label="Count of Color 🎨", value=count_color, delta=round(count_color))
        kpi3.metric(label="Average Table ⏳", value=f'{round(avg_table,2)}', delta=f'{round(avg_table,2)}')


        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Pie Chart of top 4 frequent values by dimension")
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != 'Unnamed: 0']

            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Select a numeric column", ['x', 'y', 'z'])
                top_values = data[selected_col].value_counts().nlargest(4)
                fig1 = plt.figure(figsize=(2,2))
                plt.pie(top_values.values, labels=top_values.index, autopct='%1.1f%%', startangle=90)
                plt.title(f'Distribution of top 4 frequent values of {selected_col}')
                plt.axis("equal")
                st.pyplot(fig1)


            st.subheader('Boxplot of all clarities by price')
            fig2 = plt.figure()
            sns.boxplot(x="clarity", y="price", data=data, palette='viridis')
            st.pyplot(fig2)


        with col2:
            st.subheader("CountPlot of top 10 carat frequent values")
            fig3 = plt.figure(figsize=(6,4.78))
            top_values = data['carat'].value_counts().nlargest(10).index
            filtered_data = data[data['carat'].isin(top_values)]
            sns.countplot(data=filtered_data, x="carat", palette='muted')
            st.pyplot(fig3)

            st.subheader("ScatterPlot of depth by table")
            fig4 = plt.figure(figsize=(6,4.38))
            sns.scatterplot(x="depth", y="table", data=data, color='#e516f8')
            st.pyplot(fig4)



    elif choice == "Machine Learning":
        tab1,tab2,tab3 = st.tabs([":clipboard: Check-Up of Data", ":question: Why this analysis", ":gem: Analysis"])
        with tab1:
            st.subheader("Dataset - Viewer")
            st.write(data.head())
            st.subheader("Descriptive Statistics ")
            st.write(data.describe())
            st.subheader("Columns Dataset")
            st.write(data.columns.tolist())

        with tab2:
            st.subheader("Understanding the Diamond Cut Prediction Process with Machine Learning")

            st.markdown("""The prediction of a diamond's *cut* quality is performed using a supervised machine learning model trained 
                        on a labeled dataset of diamond characteristics. This model analyzes several numerical and categorical attributes,
                        such as carat (weight), depth, table (top width), price, and physical dimensions (x, y, z), along with encoded color 
                        and clarity levels. These features are statistically correlated with the *cut* quality, which ranges from Fair to Premium. 
                        Once a user inputs these values through the interface, the system forms a structured data sample that is passed to the 
                        trained classification model. Internally, the model applies the patterns it has learned during training to evaluate where 
                        this new diamond fits among the different cut categories. The result is a classification label indicating the predicted cut
                        grade. A clear and reliable prediction is crucial because the cut is one of the most significant factors influencing a 
                        diamond’s brilliance, market value, and desirability. Unlike other attributes, the cut reflects the craftsmanship of how 
                        well a diamond has been shaped and faceted, directly impacting how light reflects through it. In professional valuation and 
                        trading environments, a misjudgment in cut quality can lead to significant financial implications. Therefore, having a dependable 
                        automated tool helps buyers, sellers, and gemologists make more informed, consistent decisions with reduced bias or human error.""")
            
            st.image("Ima_projet/diamond2.jpg", use_container_width=True)

        with tab3:    
            st.subheader("Machine Learning Prediction")
        
            # Sliders pour les valeurs numériques
            st.subheader("Enter values for the prediction")
            stab1, stab2 = st.columns(2)
            with stab1:
                carat = st.slider("Carat", min_value=0.0, max_value=30.0, value=0.0, step=0.01)
                depth = st.slider("Depth", min_value=0.0, max_value=80.0, value=0.0, step=0.01)
                table = st.slider("Table", min_value=0.0, max_value=80.0, value=0.0, step=0.01)
                new_color = st.selectbox("Color", options=[0, 1, 2, 3, 4, 5, 6])

            with stab2:
                x = st.slider("x", min_value=0.0, max_value=10.0, value=0.0, step=0.01)
                y = st.slider("y", min_value=0.0, max_value=10.0, value=0.0, step=0.01)
                z = st.slider("z", min_value=0.0, max_value=10.0, value=0.0, step=0.01)
                new_clarity = st.selectbox("Clarity", options=[0, 1, 2, 3, 4, 5, 6])
            
            price = st.number_input("Price", min_value=100, max_value=1000, step=10)
            
            button = st.button("Prediction with Machine Learning")
            if button:
                new_cut = predict_cut(carat, depth, table, price, x, y, z, new_color, new_clarity)
                if new_cut:
                    st.success(f"✅ The result of Machine Learning Prediction is : **{new_cut}**")
                    
                    # Sauvegarde des données dans session_state pour le ChatBot
                    st.session_state['derniere_prediction_ml'] = {
                        "Carat": carat,
                        "Depth": depth,
                        "Table": table,
                        "Price": price,
                        "x": x,
                        "y": y,
                        "z": z,
                        "new_color": new_color,
                        "new_clarity": new_clarity,
                        "Resultat": new_cut
                    }


    elif choice == "Deep Learning":
        tab1,tab2 = st.tabs([":question: Why this analysis", ":gem: Analysis"])
        with tab1:
            st.subheader("Understanding the Diamond Cut Prediction Process with Deep Learning")
            st.markdown("The deep learning model employed for diamond cut prediction is based on a neural network architecture trained on a dataset of diamond characteristics. " \
                        "This model is designed to learn complex, non-linear relationships between a diamond’s physical and categorical features and its cut quality. " \
                        "During training, the model receives input features such as:")
            st.markdown("- Carat (weight of the diamond) ;")            
            st.markdown("- Depth and Table (proportions of the diamond) ;")
            st.markdown("- Price ;")
            st.markdown("- Physical dimensions: x, y and z ;")
            st.markdown("- Encoded categorical variables: color and clarity ;")
            st.markdown("These inputs are passed through multiple interconnected layers of artificial neurons. Each neuron applies a mathematical transformation (usually a weighted " \
                        "sum followed by an activation function), allowing the network to detect intricate patterns within the data. Over time, using backpropagation and optimization " \
                        "techniques like stochastic gradient descent, the model adjusts its internal parameters to minimize prediction error. Once trained, the deep learning model can take" \
                        "a new, unseen data sample (a diamond's characteristics) and output a predicted cut class. The model’s output is a probability distribution over the possible cut categories "
                        "(e.g., Fair, Good, Very Good, Premium, Ideal), and the class with the highest probability is selected as the final prediction. This approach is particularly effective because " \
                        "deep learning excels at uncovering hidden relationships in large and complex datasets. It enables a more accurate and reliable prediction of diamond quality attributes, supporting " \
                        "better valuation and decision-making in the gemstone industry.")
            
            st.image("Ima_projet/diamond1.jpg", use_container_width=True)

        with tab2:    
            st.subheader("Deep Learning Prediction")
        
            # Sliders pour les valeurs numériques
            st.subheader("Enter values for the prediction")
            stab1, stab2 = st.columns(2)
            with stab1:
                carat = st.slider("Carat", min_value=0.0, max_value=30.0, value=0.0, step=0.01)
                depth = st.slider("Depth", min_value=0.0, max_value=80.0, value=0.0, step=0.01)
                table = st.slider("Table", min_value=0.0, max_value=80.0, value=0.0, step=0.01)
                new_color = st.selectbox("Color", options=[0, 1, 2, 3, 4, 5, 6])

            with stab2:
                x = st.slider("x", min_value=0.0, max_value=10.0, value=0.0, step=0.01)
                y = st.slider("y", min_value=0.0, max_value=10.0, value=0.0, step=0.01)
                z = st.slider("z", min_value=0.0, max_value=10.0, value=0.0, step=0.01)
                new_clarity = st.selectbox("Clarity", options=[0, 1, 2, 3, 4, 5, 6])

            price = st.number_input("Price", min_value=100, max_value=1000, step=10)

            if st.button("Prediction with Deep Learning"):
                new_cut_dl = predict_cut_dl(carat, depth, table, price, x, y, z, new_color, new_clarity)
                if new_cut_dl:
                    st.success(f"✅ Your result of Deep Learning Prediction is: **{new_cut_dl}**")


                    st.session_state['derniere_prediction_dl'] = {
                        "Carat": carat,
                        "Depth": depth,
                        "Table": table,
                        "Price": price,
                        "x": x,
                        "y": y,
                        "z": z,
                        "new_color": new_color,
                        "new_clarity": new_clarity,
                        "Resultat": new_cut_dl
                    }

    elif choice == "ChatBot":
        col1, col2 = st.columns([0.06, 0.94])  # Ajuste les proportions selon la taille de l’image
        with col1:
            st.image("Ima_projet/bot.png", width=60)

        with col2:
            st.markdown('<h2 style="margin: 0; padding: 0; color:#06d9e0">Discuss with DIAMABOT<h2>', unsafe_allow_html=True)
        

        st.markdown("""
            <style>
            .typewriter-container {
                text-align: center;
                margin-top: 10px;
                margin-bottom: 1px;
                font-family: Times New Roman;
                font-size: 20px;
                color: #ffffff;
            }

            .typewriter-text {
                display: inline-block;
                overflow: hidden;
                white-space: nowrap;
                animation:
                    typing 3s steps(50, end),
                    fadeBlur 1.5s ease-out forwards;
                filter: blur(6px);
            }

            @keyframes typing {
                from { width: 0 }
                to { width: 100% }
            }

            @keyframes fadeBlur {
                0% { filter: blur(6px); }
                100% { filter: blur(0px); }
            }
            </style>

            <div class="typewriter-container">
                <div class="typewriter-text">👋 Good Morning, I am Diamabot ! How can I help you ?</div>
            </div>
            """, unsafe_allow_html=True)
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        
        # Liste des mots-clés autorisés pour filtrer les questions
        allowed_keywords = [
            "cut", "new_cut", "carat", "depth", "profondeur", "table",
            "x", "y", "z", "dimensions", "new_color", "new_clarity", "couleur", "clarté",
            "prédiction", "prediction", "résultat", "result", "avis"
        ]

        def is_question_relevant(question):
            return any(keyword.lower() in question.lower() for keyword in allowed_keywords)


        API_URL = "https://router.huggingface.co/hf-inference/models/microsoft/Phi-3.5-mini-instruct/v1/chat/completions"
        headers = {
            "Authorization": "Bearer {st.secrets['Token1']}",
        }

        def query_hf_model(message, prediction_data=None):
            context = "Tu es un expert en diamant. Réponds uniquement aux questions concernant " \
                      "les diamants en général ou les résultats de prédiction liés aux diamants. Donne des réponses détaillées, " \
                      "claires et faciles à comprendre pour un client ou un professionnel du milieu diamantaire."

            if prediction_data:
                context += f"""
                Voici les dernières données du diamant à considérer pour le contexte :
                Carat: {prediction_data.get("Carat", "N/A")}
                Depth: {prediction_data.get("Depth", "N/A")}
                Table: {prediction_data.get("Table", "N/A")}
                Price: {prediction_data.get("Price", "N/A")}
                x: {prediction_data.get("x", "N/A")}
                y: {prediction_data.get("y", "N/A")}
                z: {prediction_data.get("z", "N/A")}
                new_color: {prediction_data.get("new_color", "N/A")}
                new_clarity: {prediction_data.get("new_clarity", "N/A")}
                Résultat de la prédiction : {prediction_data.get("Resultat", "N/A")}
                """

            payload = {
                "messages": [
                    {"role": "system", "content": context},
                    {"role": "user", "content": message}
                ],
                "model": "microsoft/Phi-3.5-mini-instruct",
                "temperature": 0.4,
                "max_tokens": 768
            } 

            try:
                response = requests.post(API_URL, headers=headers, json=payload)
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    return f"Erreur d'appel API : {response.status_code}"
            except Exception as e:
                return f"Erreur : {str(e)}"
        




        # Récupérer les données de la dernière prédiction
        prediction_data = st.session_state.get('derniere_prediction_ml', {}) or st.session_state.get('derniere_prediction_dl')

        user_message = st.chat_input("Please ask your question (ex: What does Depth mean ?)")

        if user_message:
            if is_question_relevant(user_message):
                with st.spinner("DIAMABOT is thinking..."):
                    bot_reply = query_hf_model(user_message, prediction_data)

                # Stocker la conversation
                st.session_state.messages.append({"role": "user", "content": user_message})
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            else:
                st.warning("⚠️ I cannot answer this question. Please try again")

        # Affichage des messages avec style bulle
        for msg in st.session_state.messages:
            clean_content = msg["content"].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            
            if msg["role"] == "user":
                st.markdown(f"""
                    <div style='display: flex; justify-content: flex-end; margin: 10px 0;'>
                        <div style='position: relative; background-color: #f0f0f0; color: black; padding: 10px 14px; border-radius: 16px; max-width: 70%;'>
                            <div style='white-space: pre-wrap;'>{clean_content}</div>
                            <div style="
                                content: '';
                                position: absolute;
                                right: -10px;
                                top: 10px;
                                width: 0;
                                height: 0;
                                border-top: 10px solid transparent;
                                border-bottom: 10px solid transparent;
                                border-left: 10px solid #f0f0f0;
                            "></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style='display: flex; justify-content: flex-start; margin: 10px 0;'>
                        <div style='position: relative; background-color: #e516f8; color: white; padding: 10px 14px; border-radius: 16px; max-width: 70%;'>
                            <div style='white-space: pre-wrap;'>{clean_content}</div>
                            <div style="
                                content: '';
                                position: absolute;
                                left: -10px;
                                top: 10px;
                                width: 0;
                                height: 0;
                                border-top: 10px solid transparent;
                                border-bottom: 10px solid transparent;
                                border-right: 10px solid #e516f8;
                            "></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
