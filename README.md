# Commander-Deck-Builder
Commander recommender and deck builder for the Commander format in Magic: The Gathering (MTG).

## How can I get more people interested in this niche but growing format?
There are over 18,000 cards ever printed in MTG. The Commander format requires you to choose one legendary creature out of a possible 1000 to be the commander and then build a 99-card deck of unique cards themed around this commander. This is probably the hardest part of getting started in Commander and all the learning and fun starts after this step is completed. So to expidite this process, I decided to help people choose their legendary creature commander and then build a functioning and well-rounded deck around their chosen commander.

## The Data
I gathered 260,000+ MTG cards from the Scryfall API with 30 features per card. These features describe everything that can be seen on a physical card.

![card-layout](/Images/MTG-card-layout.jpg "Card structure")

I also web-scrapped EDHrec.com for popularity statistics on a card. Every card will have a `True` or `False` value for each color combination in MTG signifying if this particular card is popular within that color combination.

Of the 260,000+ cards that I gathered from Scryfall, many of them are reprints of cards or different language prints of cards. There are also cards included which are banned from the Commander format. After removing the duplicate cards and banned cards from my dataset, I am left with around 18,000 cards. I chose the most recent printing of a card to keep.

## Choosing a Commander
The first step to building a deck for Commander is to choose a single legendary creature to be your commander. Since I want beginners to be able to use this as well, I needed to find a way to categorize cards in such a way that doesn't require any knowledge of game mechanics.
### Topic Modeling with NLP
I used **Latent Dirichlet Allocation** (LDA), an unsupervised method of Natural Language Processing, to form 5 general topics to categorize each card. I chose 5 because I wanted a fair amount of overlap within topics so that some cards will have an even split between multiple topics. This would help in recommending cards based on multiple topics. 

After subsetting my card pool to only include legendary creatures, I created a corpa of individual card lines from the card text of each of these creatures. I chose to split the text into individual lines because a single card can have multiple different effects. I then vectorized my corpa using Term Frequency-Inverse Document Frequency (TF-IDF) from the Scikit-learn library. The vectorizer had parameters that allowed me to create bi-grams from my sentences because bi-grams like `opponent creature` would mean a lot more than just the word `creature`. I also set the minimum and maximum frequency a word was allowed to have before it would be automatically included into the stop words. I wanted tokens to appear in at least 5 documents but no more than 95% of the documents.

I also created my own list of stopwords for the vectorizer to ignore by iteratively vectorizing the corpa, modeling the topics, and then including words that were describing 4 or more of the 5 topics. This was from looking at the feature importance of the LDA model.

![Army-Maker](/Images/Army_Maker_Topic.png)

![Counters-Everywhere](/Images/Counter_Everywhere_Topic.png)

I trained the model on 80% of the text lines and tested it on the remaining 20%. For each legendary creature, the model returned how likely a card would belong to a certain topic. Putting these topic scores into a DataFrame returns something that looks like this:

![topic-scores](/Images/topic_scores.png "topic scores dataframe")

The topics were originally just numbered 0 to 4. I created those names myself based on the important words and the top 10 cards in that specific topic. I also had to make sure the topic names didn't include any mechanics that beginners would not be familiar with. The distribution of the topics is fairly even.

![topic-distribution](/Images/Commander-LDA.png)

### Recommending the Commander
I didn't have any user-data so I would only do a content-based recommendation system. If the person getting a recommendation was a complete beginner, they would be asked to pick any number of the 5 pre-generated topics. I would then create the 'perfect' commander for them that would have evenly distributed scores amongst the topics they chose. Using the topic scores that my LDA model generated, I would recommend cards based on shortest euclidean distance to this 'perfect' commander. I also used each card's popularity ranking from EDHrec.com to weigh its distance; the more popular a card was, the less their distance would be increased.

If the person has played magic before and wanted a new commander to play with, they would instead enter a legendary creature that they enjoyed and that card's topic scores would be used instead of creating a 'perfect' commander.

## Building a Deck
After we have chosen a commander to build our deck around, we can start building the deck. I wanted to build a well-rounded deck that could handle many situations, instead of a one-dimensional deck that can only do a single thing well.

The approach I took was to build the deck one card at a time, choosing the next card based on how well it fit with the deck's current theme. This was again done by creating topics with LDA for the entire card pool. This time I wanted to be more specific with the topics so I created 15 different topics to categorize the cards into.

I also set aside room for lands and staple cards that fit the commander's color identity. These will not be included into the deck until the very end because I did not want them to affect the deck's overall topic score when building the deck.

After setting aside room for the lands and staple cards, the rest of the deck will be built one card at a time. Starting with only the chosen commander, I calculate the deck's average topic score to use as the center of the deck. Then I find the euclidean distance between this center and every card still in the card pool. This distance is then weighted by its EDHrec ranking, the current mana curve of the deck, the current number of draw effects the deck currently has, and the number of creature cards the deck currently has. Each deck should have a certain number of draw effects and a certain number of creature cards, so the cards that the deck currently lacks will have its distance slightly depending on how many cards of that type the deck still needs.

Here are the statistics for two decks that the builder generated:

**Niv-Mizzet Reborn Deck**

![niv-mizzet-reborn](/Images/Niv-Mizzet-Reborn.png)

**Kozilek, the Great Distortion Deck**

![kozilek](/Images/Kozilek-Great-Distortion.png)

## Showing the Deck
I also want to be able to show the generated deck to the person right away. The Scryfall API includes high resolution images of each card that I can stitch together and plot using matplotlib's subplots. However, storing 18,000+ high resolution images on my own machine would be a huge waste of space. Instead, I opted to make live API calls when creating the deck image to **save storage space at the expense of run time**. The deck can still be generated in under a minute with less than decent internet connection, so the amount of time sacrificed is well worth the space saved.

## Conclusions
This was meant to be a way to introduce people to the Commander format in MTG. This is by no means the best possible deck but is a good starting point to play around and experiment with. Most of the fun comes in playing your deck against other people's decks and seeing what cool interactions occur when there are 4-6 people playing at once.

## Improvements and Next Steps
* Make API calls to TcgPlayer for card prices. Hope to use price as a feature so that different decks can be built for different budgets.
* Change the popularity ranking from edhrec from binary `true` or `false` to a percentage representing the number of user submitted decks the card appears in.
* Adjust stopwords, parameters for topic model, and weights for scores.
* Create a front end using Flask or Streamlit.
