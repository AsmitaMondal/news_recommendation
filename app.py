import streamlit as st
import pandas as pd
from trying import scrape_category
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from streamlit_lottie import st_lottie
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import warnings
warnings.filterwarnings("ignore")
import json
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
import time
import webbrowser

# Set the page configuration to dark theme
# st.set_page_config(
#     page_title="News Recommendation",
#     page_icon=":news:",  # You can choose an icon if you'd like
#     layout="centered",        # Optional, for centered layout
#     initial_sidebar_state="expanded",  # Optional
# )

# import streamlit as st

# Set the page configuration
st.set_page_config(
    page_title="FeedLY",
    page_icon=":newspaper:",
    layout="centered",
    initial_sidebar_state="expanded"
)




# Add custom CSS for grid layout
st.markdown("""
<style>
.grid-item {
    border: 1px solid #e0e0e0;
    padding: 15px;
    margin-bottom: 15px;
    margin-top: 10px;
    border-radius: 5px;
    border-color: red;
    height: 325px;
    overflow: hidden;

}



.grid-item h3 {
    margin-bottom: 10px;
    font-size: 19px;
    word-wrap: break-word;
    white-space: normal;
}

.grid-item p {
    overflow: hidden;
    font-size: 16px;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;

}
/* Styling for the <a> tag inside .grid-item */
.grid-item a {
    color: #00bfff; /* Set link color */
    text-decoration: none; /* Remove underline */
    font-weight: bold; /* Make link text bold */
    transition: color 0.3s ease; /* Smooth color change on hover */
}

.grid-item a:hover {
    color: red; /* Change link color on hover */
    cursor: pointer; /* Change cursor to pointer when hovering */

}
</style>

""", unsafe_allow_html=True)
# Function to scrape and load data dynamically
def preload_data():
    """
    Scrapes data for all categories and loads them into a dictionary.
    Returns:
        dict: A dictionary with category names as keys and dataframes as values.
    """
    # Updated categories with proper category URLs for scraping
    categories = {
        "sci-tech/science": "the_hindu_articles_science_pages.csv",
        "sci-tech/health": "the_hindu_articles_health_pages.csv",
        "news/national": "the_hindu_articles_india_pages.csv",
        "news/international": "the_hindu_articles_world_pages.csv",
        "entertainment/movies":"the_hindu_articles_movies_pages.csv",
        "business":"the_hindu_articles_business_pages.csv"
    }

    data = {}

    # Check if the combined output CSV exists
    output_csv = "combined_news_articles.csv"
    if not os.path.exists(output_csv):  # If the combined file doesn't exist, scrape
        print("Combined output CSV does not exist. Scraping all categories...")
        scrape_category(categories, output_csv, pages=20)  # Scrape all categories
    else:
        print(f"Combined output CSV ({output_csv}) already exists. Skipping scrape.")
        
    categories = {
        "science": "the_hindu_articles_science_pages.csv",
        "health": "the_hindu_articles_health_pages.csv",
        "india": "the_hindu_articles_india_pages.csv",
        "world": "the_hindu_articles_world_pages.csv",
        "movies":"the_hindu_articles_movies_pages.csv",
        "business":"the_hindu_articles_business_pages.csv"
    }
    # Now load the scraped data into the dictionary
    for category, csv_file in categories.items():
        data[category] = pd.read_csv(csv_file)

    return data

# Load dataset
dataset_path = "combined_news_articles.csv"
if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)
    print("Dataset loaded successfully.")
else:
    preload_data()
    
# Deep Q-Network for News Recommendation
class NewsRecommenderDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NewsRecommenderDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class NewsRecommendationAgent:
    def __init__(self, df, state_size, action_size):
        self.df = df
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.learning_rate = 0.001
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Neural Networks
        self.model = NewsRecommenderDQN(state_size, 64, action_size)
        self.target_model = NewsRecommenderDQN(state_size, 64, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Replay Memory
        self.memory = []
        self.batch_size = 32
        self.max_memory_size = 2000
    
    def preprocess_state(self, articles):
        # Convert articles to feature vector
        features = []
        for _, article in articles.iterrows():
            # Example feature extraction (customize based on your data)
            feature = [
                len(article['title']),
                len(article['subtitle']) if pd.notna(article['subtitle']) else 0,
                # Add more meaningful features like category encoding, etc.
                hash(article['category']) % 1000,  # Simple category encoding
            ]
            features.append(feature)
        
        # Normalize and convert to tensor
        state = torch.FloatTensor(np.array(features).flatten())
        return state
    
    def choose_action(self, state):
    # Epsilon-greedy action selection with more refined exploration
        if np.random.rand() <= self.epsilon:
            # Instead of purely random, prioritize actions with higher Q-values even during exploration
            with torch.no_grad():
                q_values = self.model(state)
                probabilities = torch.softmax(q_values, dim=0)  # Convert Q-values to a probability distribution
                action = torch.multinomial(probabilities, 1).item()  # Sample action based on probabilities
            return action
        
        # If exploiting, choose the action with the highest Q-value
        with torch.no_grad():
            q_values = self.model(state)  # Predict Q-values for all actions
            best_action = torch.argmax(q_values).item()  # Select the best action based on highest Q-value
            return best_action
    
    def remember(self, state, action, reward, next_state, done):
        # Store experience in replay memory
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)
        
        # Limit memory size
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)
    
    def replay(self):
        # Not enough memories to replay
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch data
        states = torch.FloatTensor([b[0].numpy() for b in batch])
        actions = torch.LongTensor([b[1] for b in batch])
        rewards = torch.FloatTensor([b[2] for b in batch])
        next_states = torch.FloatTensor([b[3].numpy() for b in batch])
        dones = torch.FloatTensor([b[4] for b in batch])
        
        # Compute Q-values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_model(self):
        # Soft update of target network
        self.target_model.load_state_dict(self.model.state_dict())

# Display statistics and graphs
def display_statistics_and_graphs():
    # Metrics: likes, dislikes, reads, and not reads
    likes = sum([1 for idx in range(len(st.session_state.history)) if st.session_state.history[idx]["reward"] > 0])
    dislikes = sum([1 for idx in range(len(st.session_state.history)) if st.session_state.history[idx]["reward"] < 0])
    reads = sum([1 for idx in range(len(st.session_state.history)) if st.session_state.history[idx]["reward"] != 0])
    not_reads = len(st.session_state.history) - reads
    
    col1,col2=st.columns(2)
    with col1:
        st.write(f" ✅ Likes: {likes}")
        st.write(f" ❌ Dislikes: {dislikes}")
    with col2:
        st.write(f" 📜 Total Articles Read: {reads}")
        st.write(f" 🗞️ Total Articles Not Read: {not_reads}")    
    
    # Plot graphs
    metrics = ['Likes', 'Dislikes', 'Read', 'Not Read']
    values = [likes, dislikes, reads, not_reads]
    
    # Plot bar chart for article statistics
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(metrics, values, color=['#210c21c7', '#750e0ac7', '#16084fc7', '#3b3940c7'])
    ax.set_title('Article Interactions')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    
    # Plot training progress: Total score over time
    scores_over_time = [entry['reward'] for entry in st.session_state.history]
    total_score_over_time = np.cumsum(scores_over_time)

    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.plot(total_score_over_time, label='Total Score Over Time', color='purple')
    ax2.set_title('Training Progress')
    ax2.set_xlabel('Interaction Steps')
    ax2.set_ylabel('Total Score')
    st.pyplot(fig2)
 


def generate_wordcloud(content):
    
    # Tokenize the text
    tokens = word_tokenize(content)
    
    # Convert tokens to lowercase and remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    
    # Generate the word cloud
    processed_text = " ".join(filtered_tokens)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(processed_text)
    
    # Plot the word cloud
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title("Frequently Occuring Words in Your Top Articles", fontsize=12, pad=10)
    st.pyplot(fig)
 
               
# Preload data when the app starts
with st.spinner("Gettin Things Ready For You 🪄🪄 ..."):
    category_data = preload_data()

# Display an image in the sidebar
st.sidebar.image("images/about2.jpg", use_container_width=True)

# Sidebar title
st.sidebar.title("Navigation")


page = st.sidebar.radio("Go to", ["About", "News","Recommendation"])

def create_about_page():
    # Load the Lottie animation
    with open("images/lottie.json", "r") as f:
        lottie_animation = json.load(f)

    # Display the app title
    st.markdown("## 📰 FeedLY: News Recommendation App", unsafe_allow_html=True)

    # Create two columns
    col1, col2 = st.columns([1, 1])  # Adjust proportions as needed

    # Display Lottie animation in one column
    with col1:
        st_lottie(lottie_animation, height=300, key="lottie")

    # Display image in the other column
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Adds two line spaces

        st.markdown("""
        ### How It Works
        Our Reinforcement Learning model analyzes:
        - Your interaction patterns such as likes and reads
        - Your preferred categories such as science or international news
        
        🤖 Smart. Adaptive. Personalized.
        """, unsafe_allow_html=True)
        # st.image("images/about1.jpg", use_container_width=True)



    # col1, col2 = st.columns([1, 1])
    
    
    st.markdown(
    """
    <div style="
        border: 2px solid red;
        padding: 10px;
        padding-left:30px;
        border-radius: 10px;">
        <h4>Key Features:</h4>
        <p>🌐 <b>Diverse Content:</b> News across multiple categories scraped in real time from top channels.<br>
        🧠 <b>RL-Powered Recommendations:</b> Algorithms tailored to your interactions and interests.<br>  
        🔍 <b>Continuous Learning:</b> Recommendations improve with feedback to suggest better content.<br>
        🎯 <b>Result Display:</b> Statistics and Articles curated just for you to discover more.<br>
    </div>
    """,
    unsafe_allow_html=True,
)

    
    # with col2:
        
        
        
    #     st.markdown("""
    #     ### How It Works
    #     Our advanced Reinforcement Learning model analyzes:
    #     - Your reading history
    #     - Interaction patterns
    #     - Preferred topics
        
    #     🤖 Smart. Adaptive. Personalized.
    #     """, unsafe_allow_html=True)
        
# About Page
if page == "About":
    st.sidebar.markdown(
    """
    <div style="border: 2px solid red; padding: 10px; border-radius: 10px; background-color: rgba(0, 0, 0, 0.1);">
        <h3>Welcome to FeedLY!</h3>
        <p>
            👋 Hi there! Get ready to discover news articles based on your unique preferences. <br><br>
            At <b>FeedLY</b>, we personalize your reading experience with news that matters most to you!
        </p>
        
    </div>
    """,
    unsafe_allow_html=True,
)

    create_about_page()

# News Page
elif page == "News":
    
    # Dropdown for category selection
    categories = {
        "National": "india",
        "World": "world",
        "Health": "health",
        "Science": "science",
        "Movies":"movies",
        "Business":"business"
    }
   
    

    # Apply the CSS

    selected_category = st.sidebar.selectbox("Select News Category", list(categories.keys()))
    category = categories[selected_category]
    
    st.sidebar.markdown(
    """
    <div style="border: 2px solid red; padding: 10px;padding-top:15px; border-radius: 10px; background-color: rgba(0, 0, 0, 0.1);">
        <ul style="padding-left: 0;">
            <li><strong>Choose Category:</strong> Select any category from the dropdown.</li>
            <li><strong>Real Time Scraping:</strong> You get to read <b>30</b> latest news instantly!</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
    # print(category)
    # Get preloaded data for the selected category
    df1 = category_data[category]
    if(selected_category=='National'):
        em='🎆'
    elif(selected_category=='World'):
        em='🌏'
    elif(selected_category=='Health'):
        em='🚑'
    elif(selected_category=='Science'):
        em='🔬'
    elif(selected_category=='Movies'):
        em='🎞️'
    elif(selected_category=='Business'):
        em='💸'
    # Display articles in a 2-column layout
    st.title(f"News: {selected_category} {em}")

    # Define the number of articles per page and total pages
    articles_per_page = 6
    max_articles = 30  # Limit to top 30 latest articles
    df1 = df1.head(max_articles)  # Select the latest 30 articles
    total_pages = min((len(df1) + articles_per_page - 1) // articles_per_page, 5)
    
    
    
    # Radio buttons for page navigation
    page_number = st.radio(
        "Page Navigation",
        range(1, total_pages + 1),
        horizontal=True,
        key="page_navigation",
        label_visibility="collapsed"  # Hides the label
    )


    # Calculate the starting index based on the selected page number
    start_idx = (page_number - 1) * articles_per_page
    end_idx = start_idx + articles_per_page

    # Get the articles for the selected page
    page_data = df1.iloc[start_idx:end_idx]

    # Display articles in a 2-column layout
    cols = st.columns(2)  # Create 2 columns for the grid layout
    for idx, row in page_data.iterrows():
        col = cols[idx % 2]  # Select the column for this article
        with col:
            st.markdown(f'''
            <div class="grid-item">
                <h3>{row['title']}</h3>
                <p>{row['subtitle']}</p>
                <p><strong>Author</strong>: {row['author']}</p>
                <a href="{row['link']}"
                  target="_blank">Read Full Article</a>
            </div>
            ''', unsafe_allow_html=True)

elif page == "Recommendation":
    st.sidebar.markdown(
    """
    <div style="border: 2px solid red; padding: 10px; border-radius: 10px; background-color: rgba(0, 0, 0, 0.1);">
        <ul style="padding-left: 0;">
            <li><strong>Give Feedback:</strong> Like or dislike news articles.</li>
            <li><strong>Continuous Learning:</strong> With every interaction, the app adapts to your preferences.</li>
            <li><strong>Top Picks Unlocked:</strong> Once preference score reaches <strong>100</strong>, your top picks will be displayed!</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

    st.title("Try Our Recommender 🥸")
    
    # Initialize session state
    if 'agent' not in st.session_state:
        # Prepare state and action spaces
        state_size = 5 * 3  # 5 articles, 3 features per article
        action_size = 5  # Number of articles to choose from
        st.session_state.agent = NewsRecommendationAgent(df, state_size, action_size)
    
    # Initialize other session states
    if 'total_score' not in st.session_state:
        st.session_state.total_score = 0
    if 'current_articles' not in st.session_state:
        st.session_state.current_articles = df.sample(5).reset_index(drop=True)
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'category_scores' not in st.session_state:
        st.session_state.category_scores = {category: 0 for category in df['category'].unique()}
    
    # Scoring function
    def calculate_reward(liked, read,elapsed):
        if liked == "Yes" and read == "Yes" and elapsed <60:
            return 15
        if liked == "Yes" and read == "Yes" and elapsed >60:
            return 20
        elif liked == "Yes" and read == "No":
            return 10
        elif liked == "No" and read == "Yes" and elapsed > 60:
            return -10
        elif liked == "No" and read == "Yes" and elapsed < 60:
            return -5
        elif liked == "No" and read == "No":
            return -2
        elif liked == "Neutral" and read == "Yes":
            return 5
        return 0
    st.markdown("<br>", unsafe_allow_html=True)  

    # Display current articles
    st.subheader("Check Out These Articles ...")
    current_articles = st.session_state.current_articles
    # st.session_state['start_time']=0.0
    # Collect user feedback
    rewards = []
    flag=0
    
    for idx, row in current_articles.iterrows():
        try:
            st.markdown(f"""
            <div style="border: 2px solid red; border-radius: 8px; padding: 15px; margin-bottom: 10px;">
                <h3>{row['title']}</h3>
                <p><strong>Subtitle</strong>: {row['subtitle']}</p>
                <!-- <p><a href="{row['link']}" target="_blank">Read Full Article</a></p> -->
            </div>
            """, unsafe_allow_html=True)

            # Initialize session state for each article
            if f'start_time_{idx}' not in st.session_state:
                st.session_state[f'start_time_{idx}'] = None
            if f'elapsed_time_{idx}' not in st.session_state:
                st.session_state[f'elapsed_time_{idx}'] = None
            if f'timer_calculated_{idx}' not in st.session_state:
                st.session_state[f'timer_calculated_{idx}'] = False  # To avoid recalculation
            if st.button(f"📰 Wish To Full Article {idx + 1} ?", key=f"button_{idx}"):
                st.session_state[f'start_time_{idx}'] = time.time()
                st.markdown(f"[Click here to read full article](<{row['link']}>)", unsafe_allow_html=True)
                
            # if st.button(f"📰 Read Full Article {idx + 1}", key=f"button_{idx}"):
            #     st.session_state[f'start_time_{idx}'] = time.time()
            #     webbrowser.open(row['link'])

            col1, col2 = st.columns(2)

            with col1:
                liked = st.radio(
                    f"Did you like this news?",
                    ["Select an option","Yes", "No", "Neutral"],
                    key=f"like_{idx}"
                )

            with col2:
                key_name = f"read_{idx}"

                read = st.radio(
                    f"Did you read this news?",
                    ["Select an option", "Yes", "No"],
                    key=key_name
                )

                if read == "Yes" and not st.session_state[f'timer_calculated_{idx}']:
                    if st.session_state[f'start_time_{idx}'] is not None:
                        st.session_state[f'elapsed_time_{idx}'] = time.time() - st.session_state[f'start_time_{idx}']
                        st.session_state[f'timer_calculated_{idx}'] = True  # Mark as calculated


                if st.session_state[f'elapsed_time_{idx}'] is not None:
                    st.write(f"Time taken for article {idx + 1}: {st.session_state[f'elapsed_time_{idx}']:.2f} seconds")
            # Calculate reward
            elapsed=st.session_state[f'elapsed_time_{idx}']    
            # print(elapsed)
            reward = calculate_reward(liked, read,elapsed)
            rewards.append(reward)
            
            # Update category score
            category = row['category']
            st.session_state.category_scores[category] += reward
        except TypeError as e:
            st.write("⚠️ You did not read the article. You can't click on Yes")
            elapsed=0
            rewards.append(-5)


    
    # Submit feedback and learn
    if st.button("Submit Feedback"):
        # Preprocess current state
        current_state = st.session_state.agent.preprocess_state(current_articles)
        
        # Calculate total reward
        total_reward = sum(rewards)
        st.session_state.total_score += total_reward
        
        # Choose next action (article set)
        action = st.session_state.agent.choose_action(current_state)
        
        # Prepare next state (sample new articles based on action)
        next_articles = df.sample(5).reset_index(drop=True)
        next_state = st.session_state.agent.preprocess_state(next_articles)
        
        # Store experience in memory
        st.session_state.agent.remember(
            current_state, 
            action, 
            total_reward, 
            next_state, 
            st.session_state.total_score >= 100
        )
        
        # Learning step
        st.session_state.agent.replay()
        
        # Update target network periodically
        if st.session_state.total_score % 50 == 0:
            st.session_state.agent.update_target_model()
        
        # Update current articles
        st.session_state.current_articles = next_articles
        
        # Update history
        for idx, row in current_articles.iterrows():
            st.session_state.history.append({
                "title": row['title'],
                "category": row['category'],
                "link": row['link'],
                "content":row['content'],
                "reward": rewards[idx]
            })
        
        # Rerun to refresh
        st.rerun()
    
    # Display total score
    st.write(f"### Total Score: {st.session_state.total_score}")
    st.markdown("<hr>", unsafe_allow_html=True)

    # Final recommendation if score reaches threshold
    if st.session_state.total_score >= 100:
        st.write("## Your Top 5 Prefered News")
        top_articles = sorted(
            st.session_state.history, 
            key=lambda x: x['reward'], 
            reverse=True
        )[:5]
        all_content = " ".join(article['content'] for article in top_articles if 'content' in article)

        for article in top_articles:
            st.markdown(f"🏷️ &nbsp;&nbsp;<span style='font-size: 15px;'>{article['title']}</span><br><a href='{article['link']}' style='color: red; text-decoration: none;'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> Read Again</a>", unsafe_allow_html=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.write("## Your Interactions")
        display_statistics_and_graphs()
        generate_wordcloud(all_content)

        st.markdown("<hr>", unsafe_allow_html=True)
        # Show top category
        top_category = max(st.session_state.category_scores, key=st.session_state.category_scores.get)
        st.write(f"###  🥇 Top Category : {top_category.upper()}")
        st.markdown("<hr>", unsafe_allow_html=True)
        

        st.write(f"### Here are some news from the **{top_category.upper()}** category that you might like:")

        # Filter articles by the top category and exclude already shown articles
        # Filter articles by the top category and exclude already shown articles
        remaining_articles = df[df['category'] == top_category]
        displayed_articles_titles = [article['title'] for article in top_articles]
        remaining_articles = remaining_articles[~remaining_articles['title'].isin(displayed_articles_titles)]

        # Display 5 articles from the top category
        remaining_articles_to_display = remaining_articles.head(5)

        # Create a grid layout for the articles, 3 columns in the first row
        columns = st.columns(3)  # 3 columns for the first row
        for idx, (_, article) in enumerate(remaining_articles_to_display.iterrows()):
            if idx < 3:  # First row (3 articles)
                col = columns[idx]
                col.markdown(f"""
                <div style="border: 2px solid red; border-radius: 8px; padding: 15px; margin-bottom: 25px;margin-top: 30px;">
                    <h6>{article['title']}</h6>
                    <a href="{article['link']}" target="_blank" style="color: red; text-decoration: none;">🔖 Read Here</a>
                </div>
                """, unsafe_allow_html=True)
            
        # Now create a new row for the remaining 2 articles, ensuring alignment
        columns = st.columns(2)  # 2 columns for the second row
        for idx, (_, article) in enumerate(remaining_articles_to_display.iloc[3:].iterrows()):
            col = columns[idx]  # Assign the current article to the column in the second row
            col.markdown(f"""
            <div style="border: 2px solid red; border-radius: 8px; padding: 15px; margin-bottom: 30px;">
                <h6>{article['title']}</h6>
                <a href="{article['link']}" target="_blank" style="color: red; text-decoration: none;">🔖 Read Here</a>
            </div>
            """, unsafe_allow_html=True)


        # Add Try Again Button to reset the session
        if st.button("Use Again"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            # Rerun the app to reset
            st.experimental_rerun()
