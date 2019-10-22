import streamlit as st

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
import requests
import difflib

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
import unidecode

import time
import warnings
warnings.simplefilter('ignore')

from wordcloud import WordCloud

pd.options.display.max_columns=999
pd.options.display.max_rows=999

def plot_wcs(topic_words):
    c=0
    for topic in topic_words:
        fig, ax = plt.subplots(1,2,figsize=(18,6))
        merged = {}
        for d in list(topic.values())[0]:
            merged.update(d)
        # Word cloud of top 10 words
        wc = WordCloud(background_color='white').generate_from_frequencies(merged)
        ax[0].set_title(f'Word Cloud',size=25,y=1.1)
        ax[0].imshow(wc,interpolation='bilinear')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].axis('off')
        # Bar graph of top 10 words and their TF-IDF score
        ax[1].set_title('Top 10 TF-IDF Vectorizer',size=25)
        ax[1].bar(merged.keys(),merged.values(),edgecolor='black',linewidth=[1])
        ax[1].set_xticklabels(merged.keys(), rotation=45)
        ax[1].set_ylabel('TF-IDF Score',size=15)
        ax[1].set_xlabel('Tokens',size=15)
        fig.suptitle(f'Topic {c}',size=30,y=1.03)
        c += 1

def display_topics(model, feature_names, no_top_words):
    topics = []
    # Get weights of each word and match them up to feature_names
    for topic_idx, topic in enumerate(model.components_):
        topics.append({topic_idx:[{feature_names[i]:topic[i]}
                                  for i in topic.argsort()[:-no_top_words-1:-1]]})
#         print ("Topic %d:" % (topic_idx),end='\t')
#         print ("\n\t\t".join([feature_names[i]
#                         for i in topic.argsort()[:-no_top_words - 1:-1]]),end='\n\n')
    return topics

def get_topics(corpa,n_topics=10,visuals=False):
    # Create my own stop words, starting with the english stop words as a base.
    stop = list(set(stopwords.words('english')))
    names = [unidecode.unidecode(card_name.lower().replace(',','  ').replace('-',' ')).split() \
                     for card_name in corpa['name']]
    for name in names:
        stop.extend(name)
    stop.extend(['adiyah', 'adventurers', 'aliban', 'aquitect', 'arenson', 'arguel', 'arnjlot',
                 'ashnod', 'ata', 'avenant', 'barl', 'barrin', 'belbe', 'briber', 'cathars', 
                 'cogworker', 'conqueror', 'cosi', 'coursers', 'curtains', 'debtor', 'debtors', 
                 'delif', 'descendants', 'dispeller', 'drafna', 'eladamri', 'embalmer', 'fa', 
                 'farrel', 'feroz', 'feudkiller', 'fireforger', 'fool', 'forebear', 'geistcatcher',
                 'geomancer', 'ghalma', 'glassblower', 'grafdigger', 'guildmages', 'hadana',
                 'horncaller', 'hurkyl', 'ihsan', 'inventors', 'iona', 'ixalan', 'ixalli', 'ja', 
                 'jabari', 'jandor', 'journeyer', 'kaboom', 'kanar', 'kinjalli', 'kuar', 'leovold',
                 'liar', 'loreseeker', 'love', 'magewright', 'martyrs', 'mercadia', 'metalspinner', 
                 'minions', 'misfortune', 'mourner', 'murderer', 'nevinyrral', 'nightbird', 'orim', 
                 'outlaws', 'pemmin', 'primevals', 'rivals', 'rofellos', 'roilmage', 'rrik', 
                 'runechanter', 'scalebane', 'seafarer', 'seht', 'sek', 'shaila', 'shiv', 'sinstriker',
                 'skywhaler', 'slayers', 'soulcatchers', 'sunbird', 'sunbringer', 'suq', 'talara',
                 'telim', 'terashi', 'thespian', 'thieves', 'tilonalli', 'tormod', 'tovolar', 'trapfinder',
                 'trapmaker', 'ula', 'vance', 'whitesun', 'wit', 'wizards', 'wolfcaller', 'wolfhunter', 
                 'wolfrider', 'woodcutter', 'woodweaver', 'wren', 'zephid','target'])
    stop = list(set(stop))
    sents = [text.strip().lower().split('\n') for text in corpa['oracle_text'].fillna('').tolist()]
    temp = []
    for sent in sents:
        temp.extend(sent)
    sents = temp
    # Vectorize my texts by effect lines and find the topics of these effects.
    # Vectorized by effect lines instead of full card text because a card can have multiple different effects.
    vec = TfidfVectorizer(max_df=0.9, min_df=10, stop_words=stop,ngram_range=(1,2))
    sent_vec_ft = vec.fit_transform(sents)
    sent_vec_ft_words = vec.get_feature_names()
    lda=LatentDirichletAllocation(n_components=n_topics,random_state=30,verbose=1,n_jobs=-1,max_iter=25)
    lda.fit(sent_vec_ft)
    
    # Vectorize full card texts so that the LDA model can predict the overall topic of the card.
    cards = [text.strip().lower().replace('\n',' ') for text in corpa['oracle_text'].fillna('').tolist()]
    card_vec_ft = vec.transform(cards)
    
    topic_df = lda.transform(card_vec_ft)
    
    # Plot the word clouds and distribution of each topic's top 10 words
    if visuals:
        topic_words = display_topics(lda, sent_vec_ft_words, 10)
        plot_wcs(topic_words)
    
    return pd.DataFrame(topic_df,index=corpa.index)

def deck_builder(commander):
    card_pool = pd.read_json('Data/commander_legal.json')
    card_pool.drop(columns=['all_parts','artist','color_indicator','games','highres_image','lang','layout',
                            'oversized','preview','printed_text','printed_type_line','promo','released_at','reprint',
                            'reserved','set','set_name','set_search_uri','textless'],inplace=True)
    card_pool['mana_cost'].fillna('',inplace=True)
    cmdr = card_pool[card_pool['name']==commander]
    # For every card `c`, check that every color `a` in its color identity 
    # is also in the color identity of the commander. Filters out cards that don't fit
    # the commander's color identity.
    card_pool = card_pool[[all([a in cmdr['color_identity'].tolist()[0] for a in c])
                               for c in card_pool['color_identity']]]
    
    # Get the topic scores for the legal cards
    print('Getting Topics...',end=': ')
    topics_df = get_topics(card_pool,n_topics=10)
    print('Done!')
    card_pool = pd.merge(card_pool,topics_df,left_index=True,right_index=True,how='outer')
    # Initialize a DataFrame to store the deck
    print('Building Deck...',end=': ')
    deck = pd.DataFrame(columns=card_pool.columns)
    deck = deck.append(cmdr)
    card_pool.drop(cmdr.index,inplace=True)
    # Look for the correct column of booleans for staples
    colors = set(map(lambda x: x.lower(),set(cmdr['color_identity'].tolist()[0])))
    if len(colors)==0:
        colors = 'colorless'
    else:
        colors = [c for c in card_pool.columns[16:48] if set(c)==colors][0]
    
    # Set aside the deck staples. These will always be included so I don't want their topic scores to skew
    # the recommender.
    staples = card_pool[card_pool[colors]]
    card_pool.drop(staples.index.tolist(),inplace=True)
    
    # A deck usually has 37-40 lands, cutting 1 land for every 3 mana generating artifacts
    # Filled with the staple lands first and then the rest will be filled with basic lands
    # based on the distribution of the mana costs in the deck.
    manacount = 40
    manacount -= len(staples[staples['type_line'].str.contains('Land')])
    manacount -= len(staples[(staples['type_line'].str.contains('Artifact'))&(staples['oracle_text'].str.lower().str.contains('add '))])//3
    
    # A deck usually has at least 10 draw/search effects.
    draws = 0
    draws += len(staples[staples['oracle_text'].str.lower().str.contains('draw')])
    draws += len(staples[staples['oracle_text'].str.lower().str.contains('search')])
    # If not enough draw/search effects, include the top draw/search cards from the remaining cards
    if draws < 10:
        draw_cards = card_pool[(card_pool['oracle_text'].str.lower().str.contains('draw'))
                                 |(card_pool['oracle_text'].str.lower().str.contains('search'))]
        draw_cards = draw_cards.sort_values('edhrec_rank').head(10-draws)
        draws += len(draw_cards)
        staples = staples.append(draw_cards)
        card_pool.drop(draw_cards.index.tolist(),inplace=True)
    
    creatures = 40
    # Include the number of creatures in 
    creatures -= len(staples[staples['type_line'].str.contains('Creature')])
    
    # Give each card in the card pool a score based on its 'distance'
    # from the deck's average topic score, weighted by the card's edhrec rank.
    # Include the top card and repeat until deck is full.
    while(len(deck)+len(staples)+manacount<100):
        deck_avg = deck[deck.columns[48:]].mean()
        scores = (card_pool[card_pool.columns[48:]]-deck_avg).abs().sum(axis=1)
        # Create weights for creatures proportional to the number of creatures already included.
        # Meaning the less creatures there are, the lower the scores of creatures will be brought.
        creature_weight = pd.Series([(0.5/(creatures+1)) if 'Creature' in c else 2 for c in card_pool['type_line']],index=card_pool.index)
        # Consistent decks will have a mana curve centered at 3. This weight penalizes cards based on how far from
        # 3 its Converted Mana Cost (CMC) is.
        cmc_weight = pd.Series([(np.absolute(3-cost)+1)/100 for cost in card_pool['cmc']],index=card_pool.index)
        scores = scores.mul(creature_weight).mul(cmc_weight).add(card_pool['edhrec_rank']/20000).sort_values()
        if 'land' in card_pool.loc[scores.index[0]]['type_line'].lower():
            manacount -= 1
        elif 'creature' in card_pool.loc[scores.index[0]]['type_line'].lower():
            creatures -= 1
        deck = deck.append(card_pool.loc[scores.index[0]])
        card_pool.drop(scores.index[0],inplace=True)
    # Add in the staples we set aside
    deck = deck.append(staples)
    # Fill in the rest of the deck with basic lands
    mana_dist = ''.join(deck['mana_cost'].tolist())
    basic_lands = [('{W}','Plains'),
                  ('{U}','Island'),
                  ('{B}','Swamp'),
                  ('{R}','Mountain'),
                  ('{G}','Forest'),
                  ('{C}','Wastes')]
    color_counts = [mana_dist.count(color[0]) for color in basic_lands]
    color_counts = [c/sum(color_counts) for c in color_counts]
    # Finds the distribution of colored mana costs in the deck
    color_counts = [int(round(c*(manacount))) for c in color_counts]
    for idx in range(6):
        for n in range(color_counts[idx]):
            deck = deck.append(card_pool[card_pool['name']==basic_lands[idx][1]])
    deck = sort_deck(deck)
    print('Done!')
    return deck

# Sort the deck for displace purposes.
# First card is always the commander.
# Sort the rest by mana cost, then by color, and finally alphabetically.
# Lands are at the very end.
def sort_deck(deck_df):
    new_deck = deck_df.copy()
    new_deck['color_identity'] = [''.join(colors) for colors in new_deck['color_identity']]
    cmdr = new_deck.iloc[[0]]
    lands = new_deck[new_deck['type_line'].str.contains('Land')]
    new_deck.drop(cmdr.index,inplace=True)
    new_deck.drop(lands.index,inplace=True)
    new_deck = new_deck.sort_values(['cmc','color_identity','name'])
    new_deck = new_deck.append(lands.sort_values(['color_identity','name']))
    new_deck = cmdr.append(new_deck)
    return new_deck

def deck_specs(deck_df):
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    # Mana Curve
    ax[1] = st.bar_chart(deck_df[~deck_df['type_line'].str.contains('Land')]['cmc'].tolist())
    ax[1].set_xlim(0,deck_df['cmc'].max())
    ax[1].set_title('Mana Curve',size=20)
    ax[1].set_xlabel('Converted Mana Cost')
    
    # Type Distribution
    types = dict.fromkeys(['Creature','Artifact','Planeswalker','Sorcery','Instant','Land'],0)
    for typ in types.keys():
        types[typ] = len(deck_df[deck_df['type_line'].str.contains(typ)])
    ax[0] = st.bar_chart(types)
    ax[0].set_xticklabels(types.keys(),rotation=45)
    ax[0].set_title('Type Distribution',size=20)
    ax[0].set_xlabel('Card Type')
    ax[0].set_ylabel('count')

    plt.autoscale(enable=True,axis='x',tight='Tight')
    st.pyplot()
    st.write('EDHREC rank:',int(deck['edhrec_rank'].sum()))
    st.write('Removal options:',len(deck_df[(deck_df['oracle_text'].str.lower().str.contains('exile'))|
                                         (deck_df['oracle_text'].str.lower().str.contains('destroy'))]))
    st.write('Draw/search options:',len(deck_df[(deck_df['oracle_text'].str.lower().str.contains('draw'))|
                                           (deck_df['oracle_text'].str.lower().str.contains('search'))]))

def show_deck(deck_df):
    fig, ax = plt.subplots(10,10,figsize=(48,68))
    plt.subplots_adjust(wspace=0.05,hspace=0)
    for i in range(100):
        curr_ax = ax[i//10][i%10]
        curr_ax.axis('off')
        try:
            response = requests.get(deck_df.iloc[i].image_uris['normal'])
            if response.status_code!=200:
                print(response.status_code, 'for card',deck_df.iloc[i].name)
                curr_ax.imshow(plt.imread(r'placeholder_card.jpeg',format='jpeg'))
                continue
            curr_ax.imshow(plt.imread(BytesIO(response.content),format='jpg'))
            time.sleep(0.1)
        except TypeError:
            curr_ax.imshow(plt.imread(r'placeholder_card.jpeg',format='jpeg'))
    plt.savefig('new_deck.png',bbox_inches='tight')
    
def show_card(name=None,show=False):
    if name is None:
        card_name = input('Enter the name of a commander: ').strip()
        if card_name.lower()=='exit':
            return
    else:
        card_name = name
    try:
        card = commanders[commanders.index.str.lower()==card_name.lower()]
        response = requests.get(card.image_uris.tolist()[0]['normal'])
    except:
        closest = difflib.get_close_matches(card_name,commanders.index.tolist(),len(commanders.index.tolist()),0)[0]
        print(f"Could not find {card_name}. Instead showing {closest}.")
        card = commanders[commanders.index==closest]
        response = requests.get(card.image_uris.tolist()[0]['normal'])
    img = Image.open(BytesIO(response.content))
    if show==True:
        display(img)
    return card.image_uris.tolist()[0]['normal']

def recommend_by_commander(inpt):
    topics = cmdr_topics.columns[:-1]
    inpt_row = cmdr_topics.loc[inpt]
    # Calculate a score for each card, with the input having a default score of -1
    # All scores will be a 'distance' from the input row, lowest distance means most similar
    scores = pd.Series(index=cmdr_topics.index)
    scores.loc[inpt] = -1
    for idx, row in cmdr_topics.drop(index=inpt).iterrows():
        scores.loc[idx] = sum([np.absolute(row.loc[topic]-inpt_row.loc[topic]) for topic in topics])
    # Scores are weighted by their popularity/rank in the edhrec database
    scores = (scores*100000)+cmdr_topics['edhrec_rank']
    scores = scores.sort_values().index.tolist()[1:]
    # Display the recommendations in order
    print(f"Showing commanders most similar to: {inpt}")
    offset = 1
    while(1):
        fig, ax = plt.subplots(1,5,figsize=(28,14))
        for idx in range(offset,offset+5):
            ax[idx-offset].set_title(str(idx)+'. '+scores[idx-1],size=15)
            ax[idx-offset].axis('off')
            try:
                response = requests.get(show_card(scores[idx-1]))
                if response.status_code!=200:
                    print(response.status_code,'for',scores[idx-1])
                    ax[idx-offset].imshow(plt.imread(r'placeholder_card.jpeg',format='jpeg'))
                    continue
                ax[idx-offset].imshow(plt.imread(BytesIO(response.content),format='jpg'))
                time.sleep(0.1)
            except TypeError:
                ax[idx-offset].imshow(plt.imread(r'placeholder_card.jpeg',format='jpeg'))
        st.pyplot()
        move = input("(N)ext or (P)rev? ").lower().strip()
        if move == 'next' or move == 'n':
            offset += 5
        elif move == 'prev' or move == 'p':
            offset -= 5
        else:
            return
        if offset < 0 or offset+5 > len(scores):
            print("End of list.")
            return

def recommend_by_topic(likes):
    if len(likes)==0:
        return
    top_recs = (cmdr_topics[likes]-(1/len(likes))).abs().sum(axis=1)
    top_recs = top_recs[top_recs < 0.4]
    top_recs = (top_recs*100000) + cmdr_topics['edhrec_rank']
    top_recs = top_recs.sort_values().dropna().index.tolist()
    print("--------------------------------------------------------------------")
    print(f"I recommend these {len(top_recs)} commanders for {', '.join(likes)}")
    offset = 1
    while(1):
        fig,ax = plt.subplots(1,5,figsize=(28,14))
        for idx in range(offset,offset+5):
            ax[idx-offset].set_title(str(idx)+'. '+top_recs[idx-1],size=15)
            ax[idx-offset].axis('off')
            try:
                response = requests.get(show_card(top_recs[idx-1]))
                if response.status_code!=200:
                    print(response.status_code,'for',top_recs[idx-1])
                    ax[idx-offset].imshow(plt.imread(r'placeholder_card.jpeg',format='jpeg'))
                    continue
                ax[idx-offset].imshow(plt.imread(BytesIO(response.content),format='jpg'))
                time.sleep(0.1)
            except TypeError:
                ax[idx-offset].imshow(plt.imread(r'placeholder_card.jpeg',format='jpeg'))
        st.pyplot()
        move = input("(N)ext or (P)rev? ").lower().strip()
        if move == 'next' or move == 'n':
            offset += 5
        elif move == 'prev' or move == 'p':
            offset -= 5
        else:
            return
        if offset < 0 or offset+5 > len(top_recs):
            print("End of list.")
            return
def recommender():
    print("How would you like to be recommended a new Commander? (Enter the number)")
    print("\t1. There's a commander I liked playing before!")
    print("\t2. I want some general playstyles to chooose from.")
    print("\t3. WHAT IS COMMANDER???")
    while(1):
        how = input().strip()
        if how == '':
            return
        try:
            how = int(how)
            if how == 3:
                print("""
-----------------------------------------------------------------------------------
   "Commander is an exciting, unique way to play Magic that is all about awesome 
    legendary creatures, big plays, and battling your friends in epic multiplayer 
    games! In Commander, each player chooses a legendary creature as the commander 
    of their deck. They then play with a 99-card deck that contains only cards of 
    their commander's colors. Also, other than basic lands, each deck can only use 
    one copy of any card. During the game, you can cast your commander multiple 
    times, meaning your favorite Legendary Creature can come back again and again 
    to lead the charge as you battle for victory!"
        Taken from Wizards of the Coast site.
-----------------------------------------------------------------------------------
""")
            elif how in [1,2]:
                break
            else:
                print('Enter 1, 2, or 3')
        except ValueError:
            print('Enter 1, 2, or 3.')
    if how == 1:
        recommend_by_commander()
    elif how == 2:
        recommend_by_topic()

commanders = pd.read_json('Data/commander_legal.json')
commanders = commanders[commanders.type_line.str.contains('Legendary Creature')]
commanders = commanders[commanders.layout=='normal']
commanders = commanders[~commanders.oracle_text.str.contains('Partner')]
commanders.drop_duplicates('name','last',inplace=True)
commanders.set_index('name',inplace=True)
cmdr_topics = pd.read_json('Data/tfidf-LDA-commander-topics.json')
"""
# What is Commander?
-----------------------------------------------------------------------------------
Commander is an exciting, unique way to play Magic that is all about awesome legendary creatures, big plays, and battling your friends in epic multiplayer games! In Commander, each player chooses a legendary creature as the commander of their deck. They then play with a 99-card deck that contains only cards of their commander's colors. Also, other than basic lands, each deck can only use one copy of any card. During the game, you can cast your commander multiple times, meaning your favorite Legendary Creature can come back again and again to lead the charge as you battle for victory!
    
Wizards of the Coast
    
-----------------------------------------------------------------------------------
"""

options = ["Select option","There's a commander I liked playing before!","I want some general playstyles to chooose from.","Build me a Deck!"]
choice = st.selectbox('How would you like to be recommended a new Commander?',options)
if choice == options[1]:
    option = st.selectbox(
        'Which commander did you play before?',
             ["Select a Commander"]+cmdr_topics.index.tolist())
    if option!="Select a Commander":
          recommend_by_commander(option)
elif choice == options[2]:
    option = st.multiselect(
        'What playstyle(s) do you like?',
            cmdr_topics.columns[:-1])
    recommend_by_topic(option)
elif choice == options[3]:
    option = st.selectbox(
        'Which commander do you want a deck for?',
            ["Select a Commander"]+cmdr_topics.index.tolist())
    if option != "Select a Commander":
          st.write('Building...')
          deck = deck_builder(option)
          st.write('Built!')
          deck_specs(deck)
          show_deck(deck)